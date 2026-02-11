from typing import Dict, Any, Optional
import pytesseract
import numpy as np
from ..utils.logging import logger

class TextExtractor:
    """Extracts trademark text from metadata or via OCR fallback."""
    
    def __init__(self):
        logger.debug("Initialized TextExtractor.")

    def extract(self, metadata: Dict[str, Any], image: Optional[np.ndarray] = None) -> str:
        """
        Main entry point for text extraction.
        Prioritizes structured metadata, falls back to OCR if image provided.
        """
        # 1. Check metadata (90.1% of L3D has this)
        text = metadata.get("text")
        if text and isinstance(text, str) and text.strip():
            return text.strip()
            
        # 2. OCR Fallback if metadata is missing/null
        if image is not None:
            try:
                ocr_text = pytesseract.image_to_string(image).strip()
                if ocr_text:
                    logger.debug(f"Extracted OCR text: {ocr_text[:30]}...")
                    return ocr_text
            except Exception as e:
                logger.error(f"OCR text extraction failed: {e}")
                
        return ""
