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
            # M=32 is a good balance for search speed/accuracy
            sub_index = faiss.IndexHNSWFlat(self.dimension, 32)
        elif index_type == "l2":
            sub_index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "cosine":
            sub_index = faiss.IndexFlatIP(self.dimension)
        else:
            logger.warning(f"Unknown index type {index_type}, falling back to FlatL2")
            sub_index = faiss.IndexFlatL2(self.dimension)
            
        # Wrap with IndexIDMap2 to store 64-bit integer IDs (or indices)
        # We'll map our string UUIDs to integer indices internally
        self.index = faiss.IndexIDMap2(sub_index)

    def add(self, embeddings: np.ndarray, ids: List[int]):
        """Add embeddings with explicitly provided integer IDs."""
        try:
            embeddings = embeddings.astype('float32')
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
            query = query.astype('float32').reshape(1, -1)
            if self.index_type == "cosine":
                faiss.normalize_L2(query)
                
            distances, indices = self.index.search(query, k)
            return distances[0], indices[0]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return np.array([]), np.array([])

    def save(self, path: str):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, str(path))
            logger.info(f"Saved FAISS index to {path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load(self, path: str):
        """Load FAISS index from disk."""
        try:
            self.index = faiss.read_index(str(path))
            self.dimension = self.index.d
            logger.info(f"Loaded FAISS index from {path} (size={self.index.ntotal})")
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            raise
