import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Add project root to path so we can import logo_similarity
sys.path.append(str(Path(__file__).parent.parent))

from logo_similarity.utils.s3 import s3_service

def upload_best_model():
    """Uploads the best_model ONNX bundle to the production Railway bucket."""
    load_dotenv()
    
    # Priority: RAILWAY_BUCKET_NAME -> MINIO_BUCKET -> 'l3d'
    bucket = os.getenv("RAILWAY_BUCKET_NAME") or os.getenv("MINIO_BUCKET", "l3d")
    model_dir = Path("models/onnx/best_model")
    
    if not model_dir.exists():
        logger.error(f"❌ Directory {model_dir} not found! Please ensure you have generated the best_model bundle.")
        return

    # Files composing the production bundle
    files_to_upload = [
        "model.onnx",
        "model.onnx.data",
        "pca_model.joblib",
        "vps_id_map.db",
        "vps_index.bin",
        "vps_metadata.json"
    ]

    logger.info(f"🚀 Starting upload to bucket: {bucket}")
    logger.info(f"Endpoint: {os.getenv('RAILWAY_BUCKET_URL', 'default')}")

    success_count = 0
    for filename in files_to_upload:
        file_path = model_dir / filename
        if file_path.exists():
            # Store in 'models/best_model/' prefix in the bucket
            object_name = f"models/best_model/{filename}"
            
            # For the .bin and .onnx.data files, this might take a while
            file_size_gb = file_path.stat().st_size / (1024**3)
            logger.info(f"📤 Uploading {filename} ({file_size_gb:.2f} GB)...")
            
            if s3_service.upload_file(str(file_path), bucket, object_name):
                logger.info(f"✅ Successfully uploaded {filename}")
                success_count += 1
            else:
                logger.error(f"❌ Failed to upload {filename}")
        else:
            logger.warning(f"⚠️ Skipping {filename} (file does not exist in {model_dir})")

    logger.info(f"🏁 Upload complete. {success_count}/{len(files_to_upload)} files transferred.")

if __name__ == "__main__":
    upload_best_model()
