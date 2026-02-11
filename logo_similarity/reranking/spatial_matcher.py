import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from ..utils.logging import logger

class SpatialMatcher:
    """
    Implements Stage 2 re-ranking using deep spatial feature matching.
    Calculates a 49x49 cosine similarity matrix between query and candidate patches.
    """
    
    def __init__(self):
        logger.info("Initialized SpatialMatcher for Stage 2 re-ranking.")

    def compute_similarity(
        self, 
        query_spatial: torch.Tensor, 
        candidate_spatial: torch.Tensor
    ) -> float:
        """
        Compute max-pooled pairwise similarity score.
        Input tensors: [1, 1280, 7, 7]
        """
        try:
            # Flatten spatial dimensions: [1, 1280, 7, 7] -> [49, 1280]
            q = query_spatial.squeeze(0).permute(1, 2, 0).reshape(49, 1280)
            c = candidate_spatial.squeeze(0).permute(1, 2, 0).reshape(49, 1280)
            
            # L2 Normalize patches
            q_norm = F.normalize(q, p=2, dim=1)
            c_norm = F.normalize(c, p=2, dim=1)
            
            # Compute 49x49 pairwise cosine similarity matrix
            # Each cell (i, j) = similarity between query patch i and candidate patch j
            sim_matrix = torch.mm(q_norm, c_norm.T) # [49, 49]
            
            # Max pooling per query patch (translation invariance)
            # For each query patch, find its best matching patch in the candidate
            max_per_query = sim_matrix.max(dim=1).values # [49]
            
            # Average of max scores = alignment score
            alignment_score = max_per_query.mean().item()
            
            return alignment_score
        except Exception as e:
            logger.error(f"Spatial similarity computation failed: {e}")
            return 0.0

    def batch_compute_similarity(
        self, 
        query_spatial: torch.Tensor, 
        candidates_spatial: torch.Tensor
    ) -> torch.Tensor:
        """
        Batched version of compute_similarity for higher throughput.
        Input query: [1, 1280, 7, 7]
        Input candidates: [N, 1280, 7, 7]
        """
        try:
            N = candidates_spatial.shape[0]
            q = query_spatial.squeeze(0).permute(1, 2, 0).reshape(49, 1280) # [49, 1280]
            c = candidates_spatial.permute(0, 2, 3, 1).reshape(N, 49, 1280) # [N, 49, 1280]
            
            q_norm = F.normalize(q, p=2, dim=1) # [49, 1280]
            c_norm = F.normalize(c, p=2, dim=2) # [N, 49, 1280]
            
            # Batch matrix multiplication: [49, 1280] x [N, 1280, 49] -> [N, 49, 49]
            # Reshape q for bmm: [1, 49, 1280] -> repeat N times
            q_batch = q_norm.unsqueeze(0).expand(N, -1, -1) # [N, 49, 1280]
            
            # bmm( [N, 49, 1280], [N, 1280, 49] )
            sim_matrices = torch.bmm(q_batch, c_norm.transpose(1, 2)) # [N, 49, 49]
            
            # Max pooling per query patch: [N, 49]
            max_per_query = sim_matrices.max(dim=2).values # [N, 49]
            
            # Average alignment scores
            alignment_scores = max_per_query.mean(dim=1) # [N]
            
            return alignment_scores
        except Exception as e:
            logger.error(f"Batch spatial similarity computation failed: {e}")
            return torch.zeros(candidates_spatial.shape[0])
