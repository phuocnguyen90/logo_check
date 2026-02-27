
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
from datetime import datetime
import onnx
import faiss
from onnxruntime.quantization import quantize_dynamic, QuantType

# Add project root to path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.embeddings.pca_reducer import PCAReducer
from logo_similarity.retrieval.vector_store import VectorStore

def vps_optimize(checkpoint_name=None):
    """
    Optimizes the model and index for limited resource VPS:
    1. Exports to ONNX (Float32) and Quantized INT8.
    2. Trains PCA on existing chunks to reduce vector size (1280 -> 256).
    3. Builds a lightweight FAISS index.
    """
    logger.info("🚀 Starting VPS Optimization...")

    # 1. Setup Model Paths
    if not checkpoint_name:
        # Find latest/best checkpoint
        ckpts = sorted(list(paths.CHECKPOINTS_DIR.glob("*semantic_epoch_*.pth")))
        if not ckpts:
            logger.error("❌ No checkpoints found to optimize.")
            return
        ckpt_path = ckpts[-1]
    else:
        ckpt_path = paths.CHECKPOINTS_DIR / checkpoint_name
        if not ckpt_path.exists():
            logger.error(f"❌ Checkpoint not found: {ckpt_path}")
            return

    ckpt_stem = ckpt_path.stem
    logger.info(f"📍 Target Checkpoint: {ckpt_path.name}")

    # 2. Export to ONNX
    onnx_dir = paths.MODELS_DIR / "onnx" / ckpt_stem
    onnx_dir.mkdir(parents=True, exist_ok=True)
    f32_path = onnx_dir / "model.onnx"
    int8_path = onnx_dir / "model_quant.onnx"

    logger.info("📦 Exporting to ONNX...")
    embedder = EfficientNetEmbedder().eval()
    
    # Load weights
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder_q.'):
            new_state_dict[k.replace('encoder_q.', '')] = v
        elif k.startswith('module.encoder_q.'):
            new_state_dict[k.replace('module.encoder_q.', '')] = v
        else:
            new_state_dict[k] = v
    embedder.load_state_dict(new_state_dict, strict=False)

    dummy_input = torch.randn(1, 3, settings.IMG_SIZE, settings.IMG_SIZE)
    torch.onnx.export(
        embedder, dummy_input, f32_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    logger.info("⚖️ Quantizing to INT8 (CPU Optimized)...")
    quantize_dynamic(f32_path, int8_path, weight_type=QuantType.QUInt8)
    
    # 3. PCA Training (1280 -> REDUCED_DIM)
    index_dir = paths.INDEXES_DIR / "embeddings" / ckpt_stem
    chunks_dir = index_dir / "chunks"
    
    if not chunks_dir.exists():
        logger.error(f"❌ No embedding chunks found at {chunks_dir}. Build index first.")
        return

    chunk_files = sorted(list(chunks_dir.glob("*.npz")))
    reduced_dim = settings.REDUCED_DIM 
    
    logger.info(f"📉 Training PCA: 1280 -> {reduced_dim}...")
    reducer = PCAReducer(n_components=reduced_dim)
    
    # Use a subset if dataset is massive, or use partial_fit on everything
    # 50 chunks is usually enough for a stable PCA
    for chunk_file in tqdm(chunk_files[:50], desc="Fitting PCA"):
        with np.load(chunk_file) as data:
            reducer.partial_fit(data['embeddings'])
            
    pca_path = onnx_dir / "pca_model.joblib"
    reducer.save(pca_path)

    # 4. Build Lightweight Index
    logger.info(f"🏗️ Building optimized FAISS index (dim={reduced_dim})...")
    store = VectorStore(dimension=reduced_dim, index_type="hnsw")
    full_id_list = []
    
    for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
        with np.load(chunk_file) as data:
            emb = data['embeddings']
            ids = data['ids']
            
            # Apply PCA
            reduced_emb = reducer.transform(emb).astype('float32')
            faiss.normalize_L2(reduced_emb)
            
            # Add to FAISS
            start_idx = len(full_id_list)
            indices = list(range(start_idx, start_idx + len(ids)))
            store.add(reduced_emb, indices)
            full_id_list.extend(ids.tolist())

    idx_path = onnx_dir / "vps_index.bin"
    id_map_path = onnx_dir / "vps_id_map.json"
    
    store.save(idx_path)
    with open(id_map_path, "w") as f:
        json.dump(full_id_list, f)

    # 5. Metadata
    build_meta = {
        "checkpoint": str(ckpt_path),
        "total_images": len(full_id_list),
        "reduced_dim": reduced_dim,
        "built_at": datetime.now().isoformat(),
        "files": {
            "model": "model_quant.onnx",
            "pca": "pca_model.joblib",
            "index": "vps_index.bin"
        }
    }
    with open(onnx_dir / "vps_metadata.json", "w") as f:
        json.dump(build_meta, f, indent=2)

    logger.info(f"✨ VPS Optimization Complete! Directory: {onnx_dir}")
    logger.info(f"   - Model: {os.path.getsize(int8_path)/1e6:.1f} MB (vs 45MB path)")
    logger.info(f"   - Index: {os.path.getsize(idx_path)/1e6:.1f} MB (Reduced by PCA)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint filename")
    args = parser.parse_args()
    vps_optimize(args.checkpoint)
