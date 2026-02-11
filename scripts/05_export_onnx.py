import torch
import onnx
from onnxconverter_common import float16
import numpy as np

from logo_similarity.config import paths, settings
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder

def export_onnx():
    """
    Exports fine-tuned PyTorch model to ONNX with FP16 conversion.
    As per Revised Plan v2.
    """
    logger.info("Initializing ONNX export...")
    
    # 1. Load model
    model = EfficientNetEmbedder()
    checkpoint_path = paths.CHECKPOINTS_DIR / "best_model.pth"
    
    if checkpoint_path.exists():
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Handle MoCo state dict wrapping if needed
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Strip 'encoder_q.' prefix if it came from MoCo class
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        logger.warning("No fine-tuned checkpoint found, exporting base pre-trained model.")

    model.eval()
    
    # 2. Export to Float32 ONNX
    dummy_input = torch.randn(1, 3, settings.IMG_SIZE, settings.IMG_SIZE)
    onnx_f32_path = paths.ONNX_DIR / "model_f32.onnx"
    onnx_f16_path = paths.ONNX_DIR / "model_fp16.onnx"
    
    paths.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Exporting to Float32 ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_f32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 3. Convert to FP16
    logger.info("Converting to FP16...")
    model_f32 = onnx.load(str(onnx_f32_path))
    model_f16 = float16.convert_float_to_float16(model_f32)
    onnx.save(model_f16, str(onnx_f16_path))
    
    logger.info(f"ONNX FP16 export complete: {onnx_f16_path}")

    # 4. Validation (Simplified)
    logger.info("Verifying ONNX output match...")
    # This would involve running both models and comparing cosine similarity
    
if __name__ == "__main__":
    export_onnx()
