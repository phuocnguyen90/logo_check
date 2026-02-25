
import os
import json
import argparse
import torch
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn

# Add project root to path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.retrieval.vector_store import VectorStore
from logo_similarity.retrieval.dataset import TrademarkInferenceDataset

# 🚀 Fix for WSL "shared memory" error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def fast_build_index():
    parser = argparse.ArgumentParser(description="🚀 High-Throughput Index Builder for RTX 3060")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (256 is safer for 16GB RAM)")
    parser.add_argument("--workers", type=int, default=8, help="CPU workers (8 is safer for 16GB RAM)")
    parser.add_argument("--force", action="store_true", help="Clear existing index for this checkpoint")
    parser.add_argument("--toy", action="store_true", help="Run on toy dataset")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (safer if model is unstable)")
    parser.add_argument("--build-only", action="store_true", help="Skip extraction, only build FAISS from existing chunks")
    args = parser.parse_args()

    # 1. Setup Paths
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Try relative to checkpoints dir
        ckpt_path = paths.CHECKPOINTS_DIR / args.checkpoint
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            return

    ckpt_stem = ckpt_path.stem
    index_output_dir = paths.INDEXES_DIR / "embeddings" / ckpt_stem
    
    if args.force and index_output_dir.exists():
        print(f"🗑️ Clearing existing index: {index_output_dir}")
        shutil.rmtree(index_output_dir)
    
    index_output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = index_output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    # 2. Setup Data
    metadata_path = paths.TOY_DATASET_METADATA if args.toy else paths.DATASET_METADATA
    print(f"📂 Loading metadata: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Resumability logic
    processed_ids = set()
    if not args.build_only:
        for chunk_file in chunks_dir.glob("*.npz"):
            try:
                with np.load(chunk_file) as data:
                    processed_ids.update(data['ids'].tolist())
            except:
                chunk_file.unlink()
        
        initial_count = len(metadata)
        metadata = [m for m in metadata if (m.get('file') or m.get('image')) not in processed_ids]
        print(f"🔄 Resume: {len(processed_ids)} already done. {len(metadata)} remaining.")
    else:
        print("⏭️ Build-only mode: Skipping extraction stage.")
        metadata = []

    if not metadata:
        print("✅ Everything already processed.")
    else:
        # 3. Model Setup
        print("🤖 Loading model to GPU...")
        device = "cuda"
        embedder = EfficientNetEmbedder().to(device).eval()
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q.'):
                new_state_dict[k.replace('encoder_q.', '')] = v
            elif k.startswith('module.encoder_q.'):
                new_state_dict[k.replace('module.encoder_q.', '')] = v
        
        if new_state_dict:
            embedder.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Loaded weights from {ckpt_path.name}")
        else:
            print("⚠️ No MoCo keys found, using defaults.")

        # 4. DataLoader Setup
        # Using pin_memory and high prefetch for RTX 3060 throughput
        dataset = TrademarkInferenceDataset(metadata)
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True
        )

        # 5. Extraction Loop
        print(f"🚀 Starting extraction (Batch Size: {args.batch_size}, Workers: {args.workers})")
        chunk_size = 5000
        cur_embs, cur_ids = [], []
        
        existing_chunks = [int(p.stem.split('_')[1]) for p in chunks_dir.glob("chunk_*.npz")]
        next_chunk_id = max(existing_chunks) + 1 if existing_chunks else 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=not args.no_amp):
            for batch_imgs, batch_ids, batch_valid in tqdm(loader, desc="Indexing"):
                valid_mask = batch_valid.bool()
                if not valid_mask.any(): continue
                
                imgs = batch_imgs[valid_mask].to(device, non_blocking=True)
                ids = np.array(batch_ids)[valid_mask.cpu().numpy()]
                
                embs = embedder(imgs).cpu().numpy()
                cur_embs.append(embs)
                cur_ids.append(ids)
                
                if sum(len(x) for x in cur_ids) >= chunk_size:
                    save_emb = np.vstack(cur_embs)
                    save_ids = np.concatenate(cur_ids)
                    np.savez_compressed(chunks_dir / f"chunk_{next_chunk_id}.npz", embeddings=save_emb, ids=save_ids)
                    next_chunk_id += 1
                    cur_embs, cur_ids = [], []

            # Save last chunk
            if cur_ids:
                save_emb = np.vstack(cur_embs)
                save_ids = np.concatenate(cur_ids)
                np.savez_compressed(chunks_dir / f"chunk_{next_chunk_id}.npz", embeddings=save_emb, ids=save_ids)

    # 6. Final FAISS Construction
    print("🏗️ Consolidating chunks into FAISS index...")
    
    # Cleanup memory before large-scale FAISS build
    if 'embedder' in locals(): del embedder
    if 'loader' in locals(): del loader
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    chunk_files = sorted(chunks_dir.glob("*.npz"))
    if not chunk_files:
        print("❌ No data to index.")
        return

    # Use VectorStore for standard HNSW/IP index
    dim = 1280 # b0 dim
    store = VectorStore(dimension=dim, index_type="hnsw")
    full_id_list = []
    
    for chunk_file in tqdm(chunk_files, desc="Building FAISS"):
        with np.load(chunk_file) as data:
            emb = data['embeddings'].astype('float32')
            ids = data['ids']
            start_idx = len(full_id_list)
            indices = list(range(start_idx, start_idx + len(ids)))
            store.add(emb, indices)
            full_id_list.extend(ids.tolist())

    store.save(index_output_dir / "faiss_index.bin")
    with open(index_output_dir / "id_map.json", "w") as f:
        json.dump(full_id_list, f)

    build_metadata = {
        "checkpoint": str(ckpt_path),
        "total_images": len(full_id_list),
        "built_at": datetime.now().isoformat(),
        "embedding_dim": dim,
        "hardware": "i5-12400 / RTX 3060 12GB"
    }
    with open(index_output_dir / "build_metadata.json", "w") as f:
        json.dump(build_metadata, f, indent=2)

    print(f"✨ Indexing Complete! Total images: {len(full_id_list)}")
    print(f"📂 Location: {index_output_dir}")

if __name__ == "__main__":
    fast_build_index()
