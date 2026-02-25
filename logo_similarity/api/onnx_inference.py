import onnxruntime as ort
import numpy as np
import joblib
from typing import List, Optional, Union
import cv2
from ..utils.logging import logger
from ..config import settings
from pathlib import Path

class ONNXInference:
    """
    Portable inference engine for ONNX models.
    Supports CPU auto-fallback for VPS and optional PCA dimensionality reduction.
    """
    
    def __init__(self, model_path: str, pca_path: Optional[str] = None):
        try:
            # Smart provider selection: try CUDA first, fallback to CPU
            available_providers = ort.get_available_providers()
            
            preferred_providers = []
            if 'CUDAExecutionProvider' in available_providers:
                preferred_providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB limit for VPS safety
                }))
            
            preferred_providers.append('CPUExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=preferred_providers)
            self.input_name = self.session.get_inputs()[0].name
            
            # Load PCA if provided
            self.pca = None
            if pca_path and Path(pca_path).exists():
                self.pca = joblib.load(pca_path)
                logger.info(f"Loaded PCA from {pca_path}")
            
            logger.info(f"Initialized ONNXInference with providers: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Full image preprocessing: BGR -> RGB -> Resize -> Normalize -> Transpose.
        Matches the training pipeline exactly.
        """
        # 1. BGR to RGB
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Resize
        img = cv2.resize(img, (settings.IMG_SIZE, settings.IMG_SIZE))
        
        # 3. Scale and Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # 4. HWC to CHW and add batch dim
        return img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def run(self, img_bgr: np.ndarray, apply_pca: bool = True) -> np.ndarray:
        """
        Run end-to-end inference: Preprocess -> ONNX -> PCA (if loaded).
        """
        try:
            batch = self.preprocess(img_bgr)
            outputs = self.session.run(None, {self.input_name: batch})
            embedding = outputs[0] # [1, 1280]
            
            # Apply PCA if requested and available
            if apply_pca and self.pca is not None:
                embedding = self.pca.transform(embedding).astype('float32')
            
            return embedding
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
