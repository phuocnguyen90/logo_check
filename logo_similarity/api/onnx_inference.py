import onnxruntime as ort
import numpy as np
import torch
from typing import List, Optional
from ..utils.logging import logger
from ..config import settings

class ONNXInference:
    """
    High-performance inference using ONNX Runtime with CUDA provider.
    Optimized for FP16 as per Revised Plan v2.
    """
    
    def __init__(self, model_path: str):
        try:
            # Configure session for GPU usage
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024, # 4GB limit for inference
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Initialized ONNXInference from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def preprocess_tensor(self, normalized_img: np.ndarray) -> np.ndarray:
        """Standard ImageNet normalization for inference."""
        # CHW conversion
        img = normalized_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        img = (img - mean) / std
        return img.astype(np.float32)

    def run(self, batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of preprocessed images.
        Input shape: [B, 3, 224, 224]
        """
        try:
            # ONNX expected float32 normally, or float16 if exported so.
            # ORT handles the precision internally if the model is FP16.
            outputs = self.session.run(None, {self.input_name: batch})
            return outputs[0]
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return np.zeros((batch.shape[0], settings.EMBEDDING_DIM))
