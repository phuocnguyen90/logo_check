import torch
import numpy as np
from typing import List, Dict, Any
from ..utils.logging import logger
from ..embeddings.efficientnet import EfficientNetEmbedder
from .spatial_matcher import SpatialMatcher
from ..config import settings

from ..preprocessing.pipeline import PreprocessingPipeline
from .scoring import CompositeScorer

class ReRanker:
    """
    Orchestrates Stage 2 re-ranking.
    Takes top-K candidates from Stage 1 and refines their order.
    """
    
    def __init__(self, embedder: EfficientNetEmbedder):
        self.embedder = embedder
        self.matcher = SpatialMatcher()
        self.pipeline = PreprocessingPipeline(config={'skip_text_removal': True})
        self.scorer = CompositeScorer()
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder.to(device).eval()
            
            # 1. Extract query spatial features
            img_tensor = torch.from_numpy(query_img_normalized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                query_spatial = self.embedder.extract_spatial(img_tensor)

            # 2. Extract and match candidates
            results = []
            
            # Re-rank strictly top_k (or fewer)
            to_process = candidates[:top_k * 2] # Process more than final k to allow ranking shift
            
            for candidate in to_process:
                cand_path = candidate.get("path")
                if not cand_path:
                    results.append({**candidate, "spatial_score": 0.0, "combined_score": candidate.get("global_score", 0.0)})
                    continue
                
                # Load and normalize candidate image
                preprocessed = self.pipeline.process_on_the_fly(cand_path)
                
                if preprocessed is None:
                    results.append({**candidate, "spatial_score": 0.0, "combined_score": candidate.get("global_score", 0.0)})
                    continue
                
                # Extract candidate spatial features
                c_img = preprocessed.normalized
                c_tensor = torch.from_numpy(c_img).permute(2, 0, 1).float() / 255.0
                c_tensor = (c_tensor - mean) / std
                c_tensor = c_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    cand_spatial = self.embedder.extract_spatial(c_tensor)
                
                # Match
                spatial_score = self.matcher.compute_similarity(query_spatial, cand_spatial)
                
                # Combine using weights from settings (Partial score for Stages 1+2)
                w_g = settings.WEIGHT_GLOBAL
                w_s = settings.WEIGHT_SPATIAL
                total_w = w_g + w_s
                combined = (w_g * candidate.get("global_score", 0.0) + w_s * spatial_score) / total_w
                
                # Compute color similarity
                color_score = self.scorer.compute_color_similarity(query_img_normalized, c_img)
                
                results.append({
                    **candidate,
                    "spatial_score": spatial_score,
                    "color_score": color_score,
                    "combined_score": combined
                })

            # 3. Final Sort
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Re-ranking pipeline failed: {e}")
            return candidates[:top_k]
