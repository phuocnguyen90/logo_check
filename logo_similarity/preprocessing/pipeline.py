import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from functools import lru_cache
import hashlib
from pathlib import Path

from .text_detector import TextDetector
from .text_masker import TextMasker
from .image_normalizer import ImageNormalizer
from ..utils.logging import logger
from ..config import settings

class PreprocessedImage:
    """Container for preprocessed image data."""
    def __init__(self, original: np.ndarray, masked: np.ndarray, normalized: np.ndarray, binary: np.ndarray):
        self.original = original      # Original loaded image
        self.masked = masked        # Text removed
        self.normalized = normalized  # Resized, padded, RGB
        self.binary = binary        # Binarized structure

class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing flow with LRU caching.
    Supports on-the-fly processing as per Revised Plan v2.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detector = TextDetector(method=settings.TEXT_DETECTION_METHOD)
        self.masker = TextMasker(inpaint_method=settings.INPAINT_METHOD)
        self.normalizer = ImageNormalizer(target_size=settings.IMG_SIZE)
        
        # We handle caching manually for more control over memory
        # but decorator @lru_cache is useful for the compute-heavy parts
        self._cache_size = self.config.get('cache_size', settings.LRU_CACHE_SIZE)
        self._skip_text_removal = self.config.get('skip_text_removal', False)
        self._cache = {}

    def _get_image_hash(self, image_path: str) -> str:
        """Simple hash for cache key."""
        return hashlib.md5(image_path.encode()).hexdigest()

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from disk with case-insensitive extension fallback."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try case conversion for extension
                path = Path(image_path)
                if path.suffix.lower() == '.jpg':
                    # Swap case: .jpg -> .JPG or .JPG -> .jpg
                    alt_suffix = '.JPG' if path.suffix == '.jpg' else '.jpg'
                    alt_path = path.with_suffix(alt_suffix)
                    if alt_path.exists():
                        image = cv2.imread(str(alt_path))
                
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def process_on_the_fly(self, image_path: str) -> Optional[PreprocessedImage]:
        """
        Runs the full pipeline on a single image.
        Checks cache first.
        """
        img_id = self._get_image_hash(image_path)
        if img_id in self._cache:
            return self._cache[img_id]

        img = self.load_image(image_path)
        if img is None:
            return None

        try:
            # 1. Detect & Mask Text (Expensive)
            if not self._skip_text_removal:
                boxes = self.detector.detect_text(img)
                masked = self.masker.mask_text(img, boxes)
            else:
                masked = img
            
            # 2. Normalize (Resize/Pad/RGB)
            normalized = self.normalizer.normalize(masked)
            
            # 3. Binarize (Structure)
            binary = self.normalizer.to_binary_structure(normalized)
            
            result = PreprocessedImage(
                original=img,
                masked=masked,
                normalized=normalized,
                binary=binary
            )
            
            # Simple LRU eviction
            if len(self._cache) >= self._cache_size:
                # Evict oldest entry (not true LRU but simple enough)
                self._cache.pop(next(iter(self._cache)))
            
            self._cache[img_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed for {image_path}: {e}")
            return None

    def process_batch(self, image_paths: List[str]) -> List[PreprocessedImage]:
        """Process a list of images (serial for now, can be multiprocessing)."""
        results = []
        for path in image_paths:
            res = self.process_on_the_fly(path)
            if res:
                results.append(res)
        return results
