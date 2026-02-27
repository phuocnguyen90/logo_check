import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from logo_similarity.utils.s3 import s3_service
from logo_similarity.utils.logging import logger

def sync_models_to_railway():
    load_dotenv()
    
    bucket = os.getenv("RAILWAY_BUCKET_NAME")
    model_id = "best_model"
    local_dir = Path("models/onnx") / model_id
    
    if not bucket:
        print("❌ Error: RAILWAY_BUCKET_NAME not set")
        return

    print(f"🚀 Syncing model '{model_id}' to Railway bucket: {bucket}")
    
    # Get list of existing keys to skip (checking specific prefixes to avoid 1000-limit miss)
    existing_keys = set()
    try:
        for prefix in ['models/', 'data/']:
            resp = s3_service.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in resp:
                existing_keys.update({obj['Key'] for obj in resp['Contents']})
        print(f"🔎 Found {len(existing_keys)} existing objects in models/ and data/ folders.")
    except Exception as e:
        print(f"⚠️ Could not fetch existing keys: {e}. Will upload all.")


    files_to_sync = [
        "model.onnx",
        "model.onnx.data",
        "pca_model.joblib",
        "vps_index.bin",
        "vps_id_map.db",
        "vps_id_map.json"
    ]
    
    for filename in files_to_sync:
        local_path = local_dir / filename
        s3_key = f"models/{model_id}/{filename}"
        
        if not local_path.exists():
            print(f"⚠️ Warning: Local file {local_path} missing, skipping.")
            continue
            
        if s3_key in existing_keys:
            print(f"⏭️ Skipping {s3_key} (already exists)")
            continue

        print(f"📤 Uploading {filename} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)...")
        success = s3_service.upload_file(str(local_path), bucket, s3_key)
        if success:
            print(f"✅ Successfully uploaded to {s3_key}")
        else:
            print(f"❌ Failed to upload {filename}")

    # --- Sync Master DB ---
    master_db = Path("data/metadata_v2.db")
    if master_db.exists():
        s3_key = f"data/{master_db.name}"
        if s3_key in existing_keys:
            print(f"⏭️ Skipping Master DB {s3_key} (already exists)")
        else:
            print(f"\n📦 Syncing Master Metadata DB...")
            print(f"📤 Uploading {master_db.name} ({master_db.stat().st_size / 1024 / 1024:.2f} MB)...")
            s3_service.upload_file(str(master_db), bucket, s3_key)
            print(f"✅ Master DB uploaded to {s3_key}")
    else:
        print("\n⚠️ Master Metadata DB (metadata_v2.db) not found locally.")


if __name__ == "__main__":
    sync_models_to_railway()
