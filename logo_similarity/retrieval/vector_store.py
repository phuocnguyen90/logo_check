import faiss
import numpy as np
import json
from typing import List, Tuple, Optional
from ..utils.logging import logger

class VectorStore:
    """
    ANN Search using FAISS with IndexIDMap for persistent identifier tracking.
    """
    
    def __init__(self, dimension: int, index_type: str = "hnsw"):
        self.dimension = dimension
        self.index_type = index_type
        self._set_index(index_type)
        logger.info(f"Initialized VectorStore with {index_type} (dim={dimension})")

    def _set_index(self, index_type: str):
        """Create the underlying FAISS index."""
        if index_type == "hnsw":
            sub_index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        elif index_type == "l2":
            sub_index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "cosine":
            sub_index = faiss.IndexFlatIP(self.dimension)
        else:
            logger.warning(f"Unknown index type {index_type}, falling back to FlatL2")
            sub_index = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap2(sub_index)

    def add(self, embeddings: np.ndarray, ids: List[int]):
        """Add embeddings with explicitly provided integer IDs."""
        try:
            embeddings = np.ascontiguousarray(embeddings, dtype='float32')
            if self.index_type == "cosine":
                faiss.normalize_L2(embeddings)
            ids_arr = np.array(ids).astype('int64')
            self.index.add_with_ids(embeddings, ids_arr)
            logger.debug(f"Added {len(ids)} vectors to index.")
        except Exception as e:
            logger.error(f"Failed to add vectors to index: {e}")

    def search(self, query: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors."""
        try:
            if query is None: return np.array([]), np.array([])
            # Ensure query is float32 and [1, D]
            query = query.astype('float32')
            if len(query.shape) == 1:
                query = query.reshape(1, -1)
            
            # Dimension check to prevent cryptic FAISS crashes
            if query.shape[1] != self.index.d:
                logger.error(f"Search dimension mismatch! Query: {query.shape[1]}, Index: {self.index.d}")
                return np.array([]), np.array([])

            if self.index_type == "cosine":
                faiss.normalize_L2(query)
                
            distances, indices = self.index.search(query, k)
            return distances[0], indices[0]
        except Exception as e:
            import traceback
            logger.error(f"Search failed: {e}")
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])

    def save(self, path: str):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, str(path))
            logger.info(f"Saved FAISS index to {path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load(self, path: str, use_mmap: bool = False):
        """Load FAISS index from disk. Supports MMAP for production."""
        try:
            if use_mmap:
                logger.info(f"Using MMAP for FAISS index: {path}")
                self.index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
            else:
                self.index = faiss.read_index(str(path))
            
            self.dimension = self.index.d
            
            # Try to move to GPU if available (for local dev)
            try:
                # MMAP indices cannot be directly moved to standard GPU indices 
                # without copying, which defeats the purpose of MMAP.
                if not use_mmap and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info(f"Moved FAISS index to GPU 0")
            except Exception:
                pass
                
            logger.info(f"Loaded FAISS index from {path} (size={self.index.ntotal}, dim={self.dimension})")
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            raise