import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import numpy as np
from typing import Optional
from ..utils.logging import logger
from ..config import settings

class EfficientNetEmbedder(nn.Module):
    """
    EfficientNet-B0 backbone for global embedding extraction.
    Supports both PyTorch and future ONNX export.
    """
    
    def __init__(self, model_name: str = "efficientnet-b0", pretrained: bool = True):
        super(EfficientNetEmbedder, self).__init__()
        try:
            if pretrained:
                self.model = EfficientNet.from_pretrained(model_name)
            else:
                self.model = EfficientNet.from_name(model_name)
            
            # Remove the classification head (fc)
            # EfficientNet-B0 final layer is 1280-dim
            self.embed_dim = self.model._fc.in_features
            self.model._fc = nn.Identity()
            
            logger.info(f"Initialized EfficientNetEmbedder with {model_name} (dim={self.embed_dim})")
        except Exception as e:
            logger.error(f"Failed to initialize EfficientNet: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global average pooled features.
        Input: [B, 3, 224, 224]
        Output: [B, 1280]
        """
        # EfficientNet extract_features returns [B, 1280, 7, 7]
        features = self.model.extract_features(x)
        
        # Global Average Pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        return pooled.squeeze(-1).squeeze(-1)

    def extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps for Stage 2 re-ranking.
        Output: [B, 1280, 7, 7]
        """
        return self.model.extract_features(x)

    @torch.no_grad()
    def get_embedding(self, normalized_img: np.ndarray, device: str = "cuda") -> np.ndarray:
        """
        Convenience method for single image inference.
        Handles tensor conversion and normalization.
        """
        self.eval()
        self.to(device)
        
        # Convert HWC uint8 [0, 255] to torch CHW float [0, 1]
        img_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization (standard for EfficientNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        embedding = self.forward(img_tensor)
        return embedding.cpu().numpy().squeeze()
