import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from typing import Optional
from ..utils.logging import logger

class PCAReducer:
    """
    Dimensionality reduction using IncrementalPCA.
    Optimized for large datasets as requested in Plan v2 (robustness).
    """
    
    def __init__(self, n_components: int = 512, batch_size: Optional[int] = None):
        self.n_components = n_components
        self.batch_size = batch_size
        self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        logger.info(f"Initialized PCAReducer (dim={n_components})")

    def partial_fit(self, embeddings: np.ndarray):
        """Fit on a batch of embeddings."""
        try:
            self.pca.partial_fit(embeddings)
        except Exception as e:
            logger.error(f"PCA partial fit failed: {e}")

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensions of embeddings."""
        try:
            return self.pca.transform(embeddings)
        except Exception as e:
            logger.error(f"PCA transform failed: {e}")
            return embeddings

    def save(self, path: str):
        """Save fitted PCA model to disk."""
        try:
            joblib.dump(self.pca, path)
            logger.info(f"Saved PCA model to {path}")
        except Exception as e:
            logger.error(f"Failed to save PCA model: {e}")

    @classmethod
    def load(cls, path: str) -> "PCAReducer":
        """Load fitted PCA model from disk."""
        try:
            reducer = cls()
            reducer.pca = joblib.load(path)
            reducer.n_components = reducer.pca.n_components
            logger.info(f"Loaded PCA model from {path} (dim={reducer.n_components})")
            return reducer
        except Exception as e:
            logger.error(f"Failed to load PCA model from {path}: {e}")
            raise
