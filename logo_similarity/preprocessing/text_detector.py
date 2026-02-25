import cv2
import numpy as np
from typing import List, Tuple, Dict
from ..utils.logging import logger

class TextDetector:
    """Detects text regions in images using Tesseract OCR."""
    
    def __init__(self, method: str = "tesseract"):
        self.method = method
        if method != "tesseract":
            logger.warning(f"Detection method '{method}' requested, but only 'tesseract' is currently implemented.")

    def detect_text(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text boxes in an image.
        Returns a list of (x, y, w, h) bounding boxes.
        """
        try:
            # Lazy loading to avoid torch/tesseract dependency in production unless needed
            try:
                import pytesseract
            except ImportError:
                logger.warning("Pytesseract not installed. Use 'pip install pytesseract' for Stage 5 text removal. Skipping detection.")
                return []
            
            # Tesseract expects RGB or Grayscale
            # Use data output to get bounding boxes directly
            # Config: --psm 11 (Sparse text. Find as much text as possible in no particular order.)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config='--psm 11')
            
            boxes = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                # Filter by confidence > 0 (Tesseract uses -1 for layout elements)
                if int(data['conf'][i]) > 0:
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    boxes.append((x, y, w, h))
            
            if boxes:
                logger.debug(f"Detected {len(boxes)} text boxes.")
            return boxes
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []

    def visualize_detections(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw bounding boxes on a copy of the image."""
        vis = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return vis