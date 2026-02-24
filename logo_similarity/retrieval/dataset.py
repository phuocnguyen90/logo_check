import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
# from torchvision import transforms # Removed to avoid NMS operator error
from PIL import Image
from typing import List, Dict, Any, Tuple
from ..utils.logging import logger
from ..config import paths
from ..preprocessing.pipeline import PreprocessingPipeline

class TrademarkInferenceDataset(Dataset):
    """
    Dataset for efficient inference/indexing.
    Optimized for high-throughput batch processing.
    """
    
    def __init__(
        self, 
        metadata: List[Dict[str, Any]], 
        transform=None,
        img_size: int = 224
    ):
        self.metadata = metadata
        self.transform = transform
        self.img_size = img_size
        
        # We manually handle normalization to be faster than the full pipeline
        from ..preprocessing.image_normalizer import ImageNormalizer
        self.normalizer = ImageNormalizer(target_size=img_size)
        
        # Standard ImageNet normalization coefficients
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, bool]:
        item = self.metadata[idx]
        img_id = item['file']
        
        # Construct path
        img_path = paths.RAW_DATASET_DIR / "images" / img_id
        
        # Handle case-insensitive extensions
        if not img_path.exists():
             stem = img_path.stem
             ext = img_path.suffix
             alt_ext = ext.swapcase()
             alt_path = img_path.with_suffix(alt_ext)
             if alt_path.exists():
                 img_path = alt_path
        
        try:
            # 1. Fast Load with OpenCV
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                return torch.zeros((3, self.img_size, self.img_size)), img_id, False
            
            # 2. Minimal Normalization (Resize + Pad + BGR2RGB)
            normalized = self.normalizer.normalize(img_bgr)
            
            # 3. Convert to Tensor
            if self.transform:
                # transform usually expects PIL
                img_pil = Image.fromarray(normalized)
                img_tensor = self.transform(img_pil)
            else:
                # Manual ToTensor + Normalize
                img_np = normalized.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # HWC -> CHW
                img_tensor = (img_tensor - self.mean) / self.std
                
            return img_tensor, img_id, True
            
        except Exception as e:
            return torch.zeros((3, self.img_size, self.img_size)), img_id, False
