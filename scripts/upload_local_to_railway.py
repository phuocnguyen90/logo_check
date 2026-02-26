import os
import threading
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ---- Config ----
MAX_WORKERS = 100

def get_railway_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv("RAILWAY_BUCKET_URL"),
        aws_access_key_id=os.getenv("RAILWAY_ACCESS_ID"),
        aws_secret_access_key=os.getenv("RAILWAY_ACCESS_KEY"),
        config=Config(signature_version='s3v4', max_pool_connections=MAX_WORKERS + 10),
        region_name="auto"
    )

def list_bucket_keys(s3_client, bucket, prefix="images/") -> set:
    """Paginate through the bucket and return the full set of existing keys."""
    logger.info(f"🌐 Querying bucket '{bucket}' for existing keys under '{prefix}'...")
    existing = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    with tqdm(desc="Listing bucket", unit=" keys") as pbar:
        for page in pages:
            for obj in page.get("Contents", []):
                existing.add(obj["Key"])
            pbar.update(len(page.get("Contents", [])))
    logger.info(f"   ↳ Found {len(existing):,} files already in bucket.")
    return existing

def upload_worker(file_info, s3_client, bucket):
    """Upload a single file to the bucket."""
    local_path, s3_key = file_info
    try:
        s3_client.upload_file(
            str(local_path), bucket, s3_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        return "uploaded"
    except Exception as e:
        return f"error: {str(e)}"

def main():
    s3_client = get_railway_client()
    railway_bucket = os.getenv("RAILWAY_BUCKET_NAME")
    local_root = Path(os.getenv("RAW_DATASET_DIR"))

    if not all([railway_bucket, local_root]):
        logger.error("RAILWAY_BUCKET_NAME or RAW_DATASET_DIR missing in .env")
        return

    # 1. Build the deterministic, sorted local file list
    logger.info(f"🔍 Scanning local directory: {local_root}")
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_files = sorted([
        (p, f"images/{p.name.lower()}")
        for ext in extensions
        for p in local_root.rglob(f"*{ext}")
    ], key=lambda x: x[1])  # Sort by S3 key for determinism
    logger.info(f"   ↳ {len(all_files):,} total images found locally.")

    # 2. Query bucket to find what's already uploaded
    existing_keys = list_bucket_keys(s3_client, railway_bucket)
    pending = [(p, k) for p, k in all_files if k not in existing_keys]

    already_done = len(all_files) - len(pending)
    logger.info(f"📋 {already_done:,} already uploaded. {len(pending):,} remaining.")

    if not pending:
        logger.info("🎉 All files already uploaded!")
        return

    # 3. Upload remaining files
    logger.info(f"🚀 Uploading {len(pending):,} files with {MAX_WORKERS} workers...")
    stats = {"uploaded": 0, "errors": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upload_worker, fi, s3_client, railway_bucket): fi
            for fi in pending
        }
        with tqdm(total=len(pending), desc="Uploading", unit="img") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res == "uploaded":
                    stats["uploaded"] += 1
                else:
                    stats["errors"] += 1
                    logger.debug(f"❌ {futures[future][0].name}: {res}")
                pbar.update(1)

    total_done = already_done + stats["uploaded"]
    logger.info("✅ Session Summary:")
    logger.info(f"   Uploaded this run : {stats['uploaded']:,}")
    logger.info(f"   Errors this run   : {stats['errors']:,}")
    logger.info(f"   Total in bucket   : {total_done:,} / {len(all_files):,}")

if __name__ == "__main__":
    main()
