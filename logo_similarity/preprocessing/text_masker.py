import cv2
import numpy as np
from typing import List, Tuple
from ..utils.logging import logger

class TextMasker:
    """Inpaints detected text regions to isolate figurative elements."""
    
    def __init__(self, inpaint_method: str = "telea"):
        self.inpaint_method = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS
        logger.debug(f"Initialized TextMasker with method: {inpaint_method}")

    def create_mask(self, image_shape: Tuple[int, ...], boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Create a binary mask where text regions are white (255)."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for (x, y, w, h) in boxes:
            # Add a small padding to ensure edges is covered
            pad = 2
            cv2.rectangle(
                mask, 
                (max(0, x - pad), max(0, y - pad)), 
                (min(image_shape[1], x + w + pad), min(image_shape[0], y + h + pad)), 
                255, 
                -1
            )
        return mask

    def mask_text(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Inpaint text regions in the image."""
        if not boxes:
            return image.copy()
            
        try:
            mask = self.create_mask(image.shape, boxes)
            # 3 is the inpaint radius
            inpainted = cv2.inpaint(image, mask, 3, self.inpaint_method)
            return inpainted
        except Exception as e:
            logger.error(f"Text masking failed: {e}")
            return image.copy()
