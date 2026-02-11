import cv2
import numpy as np
from typing import Dict
from ..utils.logging import logger

class ImageNormalizer:
    """Normalizes images (resize, pad, RGB conversion) for deep learning backbone."""
    
    def __init__(self, target_size: int = 224, background_color: int = 255):
        self.target_size = target_size
        self.background_color = background_color

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image: resize with aspect ratio maintenance and padding.
        Converts to RGB if necessary.
        """
        try:
            # Convert BGR (OpenCV default) to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image.copy()

            h, w = image_rgb.shape[:2]
            
            # Calculate scaling factor
            scale = self.target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize
            resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create white canvas
            canvas = np.full((self.target_size, self.target_size, 3), self.background_color, dtype=np.uint8)
            
            # Paste resized image onto center of canvas
            y_offset = (self.target_size - new_h) // 2
            x_offset = (self.target_size - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            # Return a blank canvas as fallback
            return np.full((self.target_size, self.target_size, 3), self.background_color, dtype=np.uint8)

    def to_binary_structure(self, image: np.ndarray) -> np.ndarray:
        """Convert image to a shape-focused binary structure (grayscale + threshold)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Use Otsu's thresholding for structure extraction
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
