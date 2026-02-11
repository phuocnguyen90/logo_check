#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.retrieval.vector_store import VectorStore
from logo_similarity.reranking.reranker import ReRanker
from logo_similarity.reranking.composite_scorer import CompositeScoringPipeline
from logo_similarity.preprocessing.pipeline import PreprocessingPipeline

def test_full_pipeline():
    """
    Demonstrates the end-to-end multi-stage retrieval pipeline.
    """
    logger.info("Starting Full Pipeline Test (Stage 1 to 5)")

# 1. Load Data/Model
    checkpoint_path = paths.CHECKPOINTS_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = paths.CHECKPOINTS_DIR / "latest.pth"
        
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {paths.CHECKPOINTS_DIR}")
        return

    # Add numpy scalars to torch safe globals for newer torch versions
    try:
        import torch.serialization
        import numpy as np
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np._core.multiarray.scalar if hasattr(np, '_core') else np.core.multiarray.scalar,
            np.float64, np.int64
        ])
    except Exception as e:
        logger.warning(f"Failed to set torch safe globals: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = EfficientNetEmbedder().to(device)
    
    # Fix for numpy scalar loading in newer torch
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Strip 'encoder_q.' prefix if it exists (checkpoints from MoCo trainer)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder_q.'):
            new_state_dict[k.replace('encoder_q.', '')] = v
        else:
            new_state_dict[k] = v
            
    embedder.load_state_dict(new_state_dict, strict=False) # strict=False to handle missing/unexpected keys
    embedder.eval()
    logger.info("Loaded model weights (stripped MoCo prefixes).")

    # 2. Load Index
    index_path = paths.EMBEDDING_INDEX_DIR / "faiss_index.bin"
    id_map_path = paths.EMBEDDING_INDEX_DIR / "id_map.json"
    
    if not index_path.exists() or not id_map_path.exists():
        logger.error("Index or ID map missing. Run 03_build_index.py first.")
        return

    # Assuming we used standard HNSW for local build (1280-dim)
    index_dim = 1280 if not settings.USE_PCA else settings.REDUCED_DIM
    store = VectorStore(dimension=index_dim, index_type="hnsw")
    store.load(index_path)
    
    with open(id_map_path, "r") as f:
        full_id_list = json.load(f)
    logger.info(f"Loaded index with {len(full_id_list)} vectors (dim={store.dimension}).")

    # 3. Load Metadata (for text similarity)
    metadata_path = paths.DATASET_METADATA
    if not metadata_path.exists():
        # Fallback to toy metadata if raw doesn't exist (though usually it should)
        metadata_path = paths.TOY_DATASET_METADATA
        
    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)
    
    # Create dict for fast lookup - handle both 'image' and 'file' keys
    metadata_map = {}
    for item in metadata_list:
        key = item.get('image') or item.get('file')
        if key:
            metadata_map[key] = item
            
    logger.info(f"Loaded metadata for {len(metadata_map)} images from {metadata_path.name}.")


    # 4. Initialize Pipeline Components
    reranker = ReRanker(embedder)
    composite_pipeline = CompositeScoringPipeline()
    preprocessing = PreprocessingPipeline(config={'skip_text_removal': True})
    
    # Load PCA Reducer for search
    from logo_similarity.embeddings.pca_reducer import PCAReducer
    
    reducer = None
    if settings.USE_PCA:
        pca_path = paths.MODELS_DIR / "pca_model.joblib"
        if pca_path.exists():
            reducer = PCAReducer.load(pca_path)
        else:
            logger.warning("PCA model not found but USE_PCA=True. Search may fail if index is reduced.")
    else:
        logger.info("PCA disabled (USE_PCA=False).")

    # 5. Pick a Query (e.g. from toy validation)
    toy_val_path = paths.TOY_VALIDATION_DIR / "similar_pairs.json"
    if not toy_val_path.exists():
        logger.error("Toy validation pairs not found.")
        return

    with open(toy_val_path, "r") as f:
        pairs = json.load(f)
    
    # Take first pair
    test_pair = pairs[0] 
    query_img_name = test_pair['image1']
    target_img_name = test_pair['image2']
    
    logger.info(f"Query: {query_img_name} | Target: {target_img_name}")

    # Process Query
    query_path = paths.RAW_DATASET_DIR / "images" / query_img_name
    # Handle flat structure
    if not query_path.exists():
        query_path = paths.RAW_DATASET_DIR / query_img_name

    preprocessed = preprocessing.process_on_the_fly(str(query_path))
    if not preprocessed:
        logger.error("Failed to preprocess query.")
        return

    # STAGE 1: Global Search
    q_emb = embedder.get_embedding(preprocessed.normalized, device=device)
    
    # Reduce dimensionality if needed
    if reducer:
        q_search = reducer.transform(q_emb.reshape(1, -1))
    else:
        q_search = q_emb.reshape(1, -1)
        
    distances, indices = store.search(q_search, k=500)
    
    candidates = []
    found_target_s1 = False
    
    # store.search returns (distances_row, indices_row)
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx == -1: continue
        img_name = full_id_list[idx]
        cand_path = paths.RAW_DATASET_DIR / "images" / img_name
        if not cand_path.exists():
            cand_path = paths.RAW_DATASET_DIR / img_name
            
        candidates.append({
            "image": img_name,
            "path": str(cand_path),
            "global_score": float(dist),
            "metadata": metadata_map.get(img_name, {})
        })
        if img_name == target_img_name:
            found_target_s1 = True
            logger.info(f"Target found in Stage 1 at rank {i+1} (score={dist:.4f})")

    if not found_target_s1:
        logger.warning("Target NOT found in Top 500 Global Search.")

    # STAGE 2: Spatial Re-ranking
    logger.info("Running Stage 2: Spatial Re-ranking (Top 50)...")
    refined_candidates = reranker.rerank_candidates(
        preprocessed.normalized, 
        candidates, 
        top_k=50
    )

    # STAGE 5: Composite Scoring (Text + Color)
    logger.info("Running Stage 5: Composite Scoring...")
    final_candidates = composite_pipeline.score_results(
        metadata_map.get(query_img_name, {}),
        preprocessed.original,
        refined_candidates
    )

    # Print Top 10
    print("\n" + "="*50)
    print("FINAL RESULTS (TOP 10)")
    print("="*50)
    for i, cand in enumerate(final_candidates[:10]):
        status = "TARGET" if cand['image'] == target_img_name else ""
        print(f"{i+1:2d}. {cand['image']} | Score: {cand['final_score']:.4f} {status}")
        print(f"    - Global: {cand.get('global_score', 0):.4f}")
        print(f"    - Spatial: {cand.get('spatial_score', 0):.4f}")
        print(f"    - Text:    {cand.get('text_similarity', 0):.4f} ('{cand.get('metadata', {}).get('text', '')}')")
        print(f"    - Color:   {cand.get('color_score', 0):.4f}")
    print("="*50)

if __name__ == "__main__":
    test_full_pipeline()
