import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.preprocessing.pipeline import PreprocessingPipeline
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.embeddings.pca_reducer import PCAReducer
from logo_similarity.retrieval.vector_store import VectorStore

def load_metadata() -> List[Dict[str, Any]]:
    """Load dataset results.json."""
    if not paths.DATASET_METADATA.exists():
        logger.error(f"Metadata file not found: {paths.DATASET_METADATA}")
        return []
    with open(paths.DATASET_METADATA, "r") as f:
        return json.load(f)

def build_index():
    """
    Builds global embedding index with atomic checkpoints and resumability.
    As per Revised Plan v2.
    """
    logger.info("Starting index building process...")
    
    # 1. Initialization
    metadata = load_metadata()
    if not metadata:
        return
        
    pipeline = PreprocessingPipeline()
    embedder = EfficientNetEmbedder().cuda().eval()
    
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
    
    # 3. Batch Processing
    all_embeddings = []
    current_chunk_embeddings = []
    current_chunk_ids = []
    
    pbar = tqdm(total=len(metadata))
    
    for i, item in enumerate(metadata):
        img_id = item['file']
        if img_id in processed_ids:
            pbar.update(1)
            continue
            
        img_path = str(paths.RAW_DATASET_DIR / "images" / item['file'])
        
        # On-the-fly preprocessing
        preprocessed = pipeline.process_on_the_fly(img_path)
        if preprocessed is None:
            pbar.update(1)
            continue
            
        # Global embedding extraction
        try:
            with torch.no_grad():
                embedding = embedder.get_embedding(preprocessed.normalized)
                current_chunk_embeddings.append(embedding)
                current_chunk_ids.append(img_id)
        except Exception as e:
            logger.error(f"Embedding failed for {img_id}: {e}")
            
        # 4. Atomic Checkpoint Save
        if len(current_chunk_ids) >= chunk_size or i == len(metadata) - 1:
            if current_chunk_ids:
                chunk_id = i // chunk_size
                temp_path = checkpoint_dir / f"chunk_{chunk_id}.tmp.npz"
                final_path = checkpoint_dir / f"chunk_{chunk_id}.npz"
                
                np.savez_compressed(
                    temp_path, 
                    embeddings=np.array(current_chunk_embeddings), 
                    ids=np.array(current_chunk_ids)
                )
                # Atomic rename
                temp_path.replace(final_path)
                
                current_chunk_embeddings = []
                current_chunk_ids = []
                
        pbar.update(1)
    
    pbar.close()
    
    # 5. Global PCA & Index Construction
    logger.info("Consolidating chunks for PCA reduction and Indexing...")
    
    all_embeddings = []
    all_ids = []
    for chunk_file in sorted(checkpoint_dir.glob("*.npz")):
        with np.load(chunk_file) as data:
            all_embeddings.append(data['embeddings'])
            all_ids.append(data['ids'])
            
    if not all_embeddings:
        logger.error("No embeddings extracted!")
        return

    all_embeddings = np.vstack(all_embeddings)
    all_ids = np.concatenate(all_ids).tolist()
    
    # Fit IncrementalPCA
    logger.info("Fitting IncrementalPCA...")
    reducer = PCAReducer(n_components=settings.REDUCED_DIM)
    # partial_fit in batches to be memory safe
    pca_batch_size = 10000
    for i in range(0, len(all_embeddings), pca_batch_size):
        reducer.partial_fit(all_embeddings[i : i + pca_batch_size])
        
    reducer.save(paths.MODELS_DIR / "pca_model.joblib")
    
    # Reduce dimensions
    logger.info("Reducing dimensions...")
    reduced_embeddings = reducer.transform(all_embeddings)
    
    # Build FAISS Index
    logger.info("Building FAISS index...")
    # Convert string IDs back to integers for FAISS IndexIDMap internally? 
    # Or just use row indices. Let's use simple row indices for this baseline.
    faiss_ids = list(range(len(all_ids)))
    
    store = VectorStore(dimension=settings.REDUCED_DIM, index_type="hnsw")
    store.add(reduced_embeddings, faiss_ids)
    store.save(paths.EMBEDDING_INDEX_DIR / "faiss_index.bin")
    
    # Save ID mapping
    with open(paths.EMBEDDING_INDEX_DIR / "id_map.json", "w") as f:
        json.dump(all_ids, f)
        
    logger.info("Index building complete!")

if __name__ == "__main__":
    build_index()
