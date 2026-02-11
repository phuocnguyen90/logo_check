import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.embeddings.pca_reducer import PCAReducer
from logo_similarity.retrieval.vector_store import VectorStore
from logo_similarity.retrieval.dataset import TrademarkInferenceDataset

def build_index():
    """
    Builds global embedding index with atomic checkpoints and resumability.
    Optimized with DataLoader for parallel preprocessing.
    """
    parser = argparse.ArgumentParser(description="Build global embedding index.")
    parser.add_argument("--toy", action="store_true", help="Run on toy dataset (pilot mode)")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    args = parser.parse_args()

    logger.info("Starting index building process...")
    
    # 1. Initialization
    metadata_path = paths.TOY_DATASET_METADATA if args.toy else paths.DATASET_METADATA
    
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    logger.info(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    embedder = EfficientNetEmbedder().cuda().eval()
    
    if args.checkpoint:
        logger.info(f"Loading trained weights from {args.checkpoint}...")
        try:
            checkpoint = torch.load(args.checkpoint)
            # MoCo helper saves model_state_dict. 
            # If standard key names match, strict=True works.
            # MoCo usually wraps encoder with 'encoder_q.' prefix or similar if DataParallel/MoCo class
            # But EfficientNetEmbedder is the base.
            # Let's check keys in Step 04: model.state_dict(). model is MoCo class.
            # MoCo class has self.encoder_q = base_encoder.
            # So keys will be 'encoder_q.model._conv_stem.weight' etc.
            # EfficientNetEmbedder keys are 'model._conv_stem.weight' etc.
            
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder_q.'):
                    new_state_dict[k.replace('encoder_q.', '')] = v
                    
            if not new_state_dict:
                 logger.warning("No 'encoder_q' keys found. Trying raw load.")
                 new_state_dict = state_dict

            embedder.load_state_dict(new_state_dict, strict=False)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return
    
    # Chunking and Checkpointing
    chunk_size = 5000
    checkpoint_dir = paths.EMBEDDING_INDEX_DIR / "chunks"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Resumability Check
    processed_ids = set()
    for chunk_file in checkpoint_dir.glob("*.npz"):
        try:
            with np.load(chunk_file) as data:
                processed_ids.update(data['ids'].tolist())
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {chunk_file}: {e}. It might be corrupt. Deleting.")
            chunk_file.unlink()
            
    logger.info(f"Resume: Found {len(processed_ids)} already processed images.")
    
    # Filter metadata to only include unprocessed items
    # We check if item['file'] is in processed_ids.
    # Note depending on extension fixes, we might have mismatch, but let's assume item['file'] is the key.
    # Toy dataset logic fixed keys in metadata itself.
    initial_count = len(metadata)
    metadata = [m for m in metadata if m['file'] not in processed_ids]
    logger.info(f"Processing {len(metadata)} remaining images (skipped {initial_count - len(metadata)}).")
    
    if not metadata:
        logger.info("No images to process. Proceeding to consolidation.")
    
    # 3. Setup DataLoader
    dataset = TrademarkInferenceDataset(metadata)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=False
    )
    
    # 4. Batch Processing
    current_chunk_embeddings = []
    current_chunk_ids = []
    
    # Since we filter metadata, chunk_id calculation needs to be continuous from previous state?
    # Or just use unique filenames. Let's use simple incremental chunking for new data.
    # We can start naming chunks based on existing max chunk ID + 1.
    
    existing_chunks = [int(p.stem.split('_')[1]) for p in checkpoint_dir.glob("chunk_*.npz")]
    next_chunk_id = max(existing_chunks) + 1 if existing_chunks else 0
    
    # We need to track total processed so we know when to save
    items_in_current_chunk = 0
    
    logger.info(f"Starting processing with {args.workers} workers, batch size {args.batch_size}...")
    
    with torch.no_grad():
        for batch_imgs, batch_ids, batch_valid in tqdm(loader, desc="Indexing"):
            # batch_imgs: [B, 3, 224, 224]
            # batch_ids: tuple of strings
            # batch_valid: tensor of bools
            
            # Filter valid only logic
            # Although simpler to just process all and filter result?
            # Or filter before forward pass to save GPU.
            
            valid_mask = batch_valid.bool()
            if not valid_mask.any():
                continue
                
            imgs = batch_imgs[valid_mask].cuda(non_blocking=True)
            valid_ids = np.array(batch_ids)[valid_mask.cpu().numpy()]
            
            try:
                # Extract embeddings [B_valid, 1280]
                # Use forward() which handles batch processing
                embeddings = embedder(imgs).cpu().numpy()
                
                current_chunk_embeddings.append(embeddings)
                current_chunk_ids.append(valid_ids)
                items_in_current_chunk += len(valid_ids)
                
                # 5. Atomic Checkpoint Save
                if items_in_current_chunk >= chunk_size:
                    # Consolidate lists
                    chunk_emb = np.vstack(current_chunk_embeddings)
                    chunk_ids = np.concatenate(current_chunk_ids)
                    
                    temp_path = checkpoint_dir / f"chunk_{next_chunk_id}.tmp.npz"
                    final_path = checkpoint_dir / f"chunk_{next_chunk_id}.npz"
                    
                    np.savez_compressed(
                        temp_path, 
                        embeddings=chunk_emb, 
                        ids=chunk_ids
                    )
                    temp_path.replace(final_path)
                    logger.debug(f"Saved chunk {next_chunk_id} with {len(chunk_ids)} items.")
                    
                    next_chunk_id += 1
                    current_chunk_embeddings = []
                    current_chunk_ids = []
                    items_in_current_chunk = 0
                    
                    # Explicit GC to free large numpy arrays
                    import gc
                    gc.collect()

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # We skip this batch, effectively dropping these images from the index.
                # In production, we might want to retry or log individual IDs.
                continue

    # Save final partial chunk
    if items_in_current_chunk > 0:
        chunk_emb = np.vstack(current_chunk_embeddings)
        chunk_ids = np.concatenate(current_chunk_ids)
        
        temp_path = checkpoint_dir / f"chunk_{next_chunk_id}.tmp.npz"
        final_path = checkpoint_dir / f"chunk_{next_chunk_id}.npz"
        
        np.savez_compressed(temp_path, embeddings=chunk_emb, ids=chunk_ids)
        temp_path.replace(final_path)
        logger.debug(f"Saved final chunk {next_chunk_id} with {len(chunk_ids)} items.")
    
    # 6. Global PCA & Index Construction
    dataset_total = initial_count
    
    logger.info("Consolidating chunks for PCA reduction and Indexing...")
    
    all_embeddings = []
    all_ids = []
    
    # Load all chunks to build final index
    # Note: If dataset is huge (millions), loading all into RAM might be tight (1280 floats).
    # 1M * 1280 * 4 bytes = ~5GB. OK for 32GB RAM.
    # But we use IncrementalPCA, so we can iterate.
    # However, for FAISS, we need all data to add() if we build one index.
    # Or we can add to index incrementally (if not training index).
    # FAISS HNSW adds incrementally!
    
    # Let's adapt to be memory efficient.
    # But PCAReducer needs training data.
    
    # Phase 2a: Train PCA (on subset or all)
    embed_dim = 1280
    reduced_dim = settings.REDUCED_DIM
    
    # Collect paths
    chunk_files = sorted(checkpoint_dir.glob("*.npz"))
    if not chunk_files:
        logger.error("No chunks found to index.")
        return

    logger.info("Fitting IncrementalPCA...")
    reducer = PCAReducer(n_components=reduced_dim)
    
    for chunk_file in tqdm(chunk_files, desc="PCA Fitting"):
        with np.load(chunk_file) as data:
            emb = data['embeddings']
            reducer.partial_fit(emb)
            
    reducer.save(paths.MODELS_DIR / "pca_model.joblib")
    
    # Phase 2b: Build FAISS Index
    logger.info("Building FAISS index...")
    store = VectorStore(dimension=reduced_dim, index_type="hnsw")
    
    full_id_list = []
    
    for chunk_file in tqdm(chunk_files, desc="Indexing"):
        with np.load(chunk_file) as data:
            emb = data['embeddings'] # [N, 1280]
            ids = data['ids'] # [N]
            
            # Reduce
            reduced_emb = reducer.transform(emb) # [N, 64]
            
            # Add to store
            # Store expects integer IDs. We maintain a mapping.
            start_idx = len(full_id_list)
            # Create range of IDs
            batch_indices = list(range(start_idx, start_idx + len(ids)))
            
            store.add(reduced_emb, batch_indices)
            
            full_id_list.extend(ids.tolist())
            
    store.save(paths.EMBEDDING_INDEX_DIR / "faiss_index.bin")
    
    # Save ID mapping
    with open(paths.EMBEDDING_INDEX_DIR / "id_map.json", "w") as f:
        json.dump(full_id_list, f)
        
    logger.info(f"Index building complete with {len(full_id_list)} vectors!")

if __name__ == "__main__":
    build_index()
