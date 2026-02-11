import cv2
import numpy as np
from typing import Dict, Any
from ..utils.logging import logger

from ..config import settings

class CompositeScorer:
    """
    Final scoring logic combining multiple similarity dimensions.
    Global Embedding + Spatial Alignment + Phonetic Similarity + Color Palette.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "global": settings.WEIGHT_GLOBAL,
            "spatial": settings.WEIGHT_SPATIAL,
            "text": settings.WEIGHT_TEXT,
            "color": settings.WEIGHT_COLOR
        }
        logger.info(f"Initialized CompositeScorer with weights: {self.weights}")

    def compute_color_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simple color histogram correlation."""
        try:
            # Convert to HSV for better color matching
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            
            # Simple 1D Hue histogram
            hist1 = cv2.calcHist([hsv1], [0], None, [180], [0, 180])
            hist2 = cv2.calcHist([hsv2], [0], None, [180], [0, 180])
            
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        except Exception as e:
            logger.warning(f"Color similarity computation failed: {e}")
            return 0.0

    def calculate_final_score(
        self, 
        global_sim: float, 
        spatial_sim: float, 
        text_sim: float = 0.0, 
        color_sim: float = 0.0
    ) -> float:
        """Combine all scores into a single similarity metric [0, 1]."""
        score = (
            self.weights["global"] * global_sim +
            self.weights["spatial"] * spatial_sim +
            self.weights["text"] * text_sim +
            self.weights["color"] * color_sim
        )
        return float(np.clip(score, 0, 1))
