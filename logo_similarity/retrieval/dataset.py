import torch
from torch.utils.data import Dataset
from torchvision import transforms
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
        transform: transforms.Compose = None,
        img_size: int = 224
    ):
        self.metadata = metadata
        self.transform = transform
        # Disable cache for one-pass indexing to prevent OOM with multiple workers
        pipeline_config = {'cache_size': 1}
        self.pipeline = PreprocessingPipeline(config=pipeline_config)
        self.img_size = img_size
        
        # Standard normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, bool]:
        item = self.metadata[idx]
        img_id = item['file']
        
        # Construct path
        img_path = paths.RAW_DATASET_DIR / "images" / img_id
        
        # Handle simple extension mismatch (e.g. metadata has .JPG but file is .jpg)
        # This is a common issue with older datasets or mixed OS origins
        if not img_path.exists():
             if img_path.suffix == '.JPG':
                 alt_path = img_path.with_suffix('.jpg')
                 if alt_path.exists():
                     img_path = alt_path
             elif img_path.suffix == '.jpg':
                 alt_path = img_path.with_suffix('.JPG')
                 if alt_path.exists():
                     img_path = alt_path
        
        try:
            # Pipeline returns PreprocessedImage(original, normalized, mask, etc.)
            # We specifically want 'normalized' which is (224, 224, 3) uint8 np array
            prep = self.pipeline.process_on_the_fly(str(img_path))
            
            if prep is None:
                # Return dummy tensor and False flag
                return torch.zeros((3, self.img_size, self.img_size)), img_id, False
                
            img = Image.fromarray(prep.normalized)
            
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = self.normalize(img)
                
            return img_tensor, img_id, True
            
        except Exception as e:
            # Log rarely to avoid spamming if mass failure
            logger.warning(f"Failed to process {img_id}: {e}")
            return torch.zeros((3, self.img_size, self.img_size)), img_id, False
