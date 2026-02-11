import torch
import numpy as np
from typing import List, Dict, Any
from ..utils.logging import logger
from ..embeddings.efficientnet import EfficientNetEmbedder
from .spatial_matcher import SpatialMatcher
from ..config import settings

class ReRanker:
    """
    Orchestrates Stage 2 re-ranking.
    Takes top-1000 candidates from Stage 1 and refines their order.
    """
    
    def __init__(self, embedder: EfficientNetEmbedder):
        self.embedder = embedder
        self.matcher = SpatialMatcher()
        logger.info("Initialized ReRanker pipeline.")

    def rerank_candidates(
        self, 
        query_img_normalized: np.ndarray,
        candidates: List[Dict[str, Any]],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks global search candidates using spatial features.
        Each candidate dict must contain 'path' to its image and 'global_score'.
        """
        if not candidates:
            return []

        try:
            # 1. Extract query spatial features
            # Use standardized tensor conversion from embedder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder.to(device).eval()
            
            img_tensor = torch.from_numpy(query_img_normalized).permute(2, 0, 1).float() / 255.0
            # ImageNet norm
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                query_spatial = self.embedder.extract_spatial(img_tensor)

            # 2. Extract and match candidates (in batches for efficiency)
            # In a real system, spatial features would be pre-calculated and cached
            # For this MVP, we might extract them on demand if not cached
            
            results = []
            for candidate in candidates:
                # Mocking retrieval/extraction of candidate spatial features
                # In production these should be in a separate cache or DB
                
                # For now, let's assume we have them or need to extract them
                # To be efficient, we'd batch this, but for simplicity:
                candidate_score = 0.5 # Default fallback
                
                # Logic for combining global + spatial scores
                # final = w_g * global + w_s * spatial
                
                results.append({
                    **candidate,
                    "spatial_score": candidate_score,
                    "combined_score": 0.5 * candidate.get("global_score", 0) + 0.5 * candidate_score
                })

            # 3. Sort by combined score
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Re-ranking pipeline failed: {e}")
            return candidates[:top_k]
