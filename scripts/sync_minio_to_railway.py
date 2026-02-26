import os
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

# Load credentials
load_dotenv()

def get_clients():
    # Local MinIO Client
    local_s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("MINIO_HOST"),
        aws_access_key_id=os.getenv("MINIO_USERNAME"),
        aws_secret_access_key=os.getenv("MINIO_PASSWORD"),
        config=Config(signature_version='s3v4'),
        region_name="us-east-1"
    )
    
    # Railway Production Client (Tigris/S3)
    railway_s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("RAILWAY_BUCKET_URL"),
        aws_access_key_id=os.getenv("RAILWAY_ACCESS_ID"),
        aws_secret_access_key=os.getenv("RAILWAY_ACCESS_KEY"),
        config=Config(signature_version='s3v4'),
        region_name="us-east-1"
    )
    
    return local_s3, railway_s3

def sync_worker(key, local_s3, railway_s3, local_bucket, railway_bucket):
    """Worker to transfer a single object from local MinIO to Railway."""
    try:
        # 1. Quick existence check on Railway to skip already synced files
        try:
            railway_s3.head_object(Bucket=railway_bucket, Key=key)
            return "skipped"
        except:
            pass # Object missing, proceed to transfer
            
        # 2. Get from Local
        response = local_s3.get_object(Bucket=local_bucket, Key=key)
        data = response['Body'].read()
        content_type = response.get('ContentType', 'image/jpeg')
        
        # 3. Put to Railway
        railway_s3.put_object(
            Bucket=railway_bucket, 
            Key=key, 
            Body=data,
            ContentType=content_type
        )
        return "synced"
    except Exception as e:
        return f"error: {str(e)}"

def main():
    local_s3, railway_s3 = get_clients()
    local_bucket = os.getenv("MINIO_BUCKET", "l3d")
    railway_bucket = os.getenv("RAILWAY_BUCKET_NAME")
    
    if not all([local_bucket, railway_bucket]):
        logger.error("Bucket names missing in .env")
        return

    logger.info(f"🔄 Syncing from Local ({local_bucket}) to Railway ({railway_bucket})...")
    
    # Use paginator for large datasets (770k is too big for a single list call)
    paginator = local_s3.get_paginator('list_objects_v2')
    
    # Track stats
    stats = {"synced": 0, "skipped": 0, "errors": 0}
    
    # Concurrent execution for speed
    max_workers = 32 # Adjust based on your upload bandwidth
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Iterate over all objects in 'images/' prefix
        for page in paginator.paginate(Bucket=local_bucket, Prefix='images/'):
            if 'Contents' not in page:
                continue
                
            contents = page['Contents']
            futures = {
                executor.submit(sync_worker, obj['Key'], local_s3, railway_s3, local_bucket, railway_bucket): obj['Key'] 
                for obj in contents
            }
            
            for future in tqdm(as_completed(futures), total=len(contents), desc="Processing batch"):
                res = future.result()
                if res == "synced":
                    stats["synced"] += 1
                elif res == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["errors"] += 1
                    logger.debug(f"Error syncing {futures[future]}: {res}")

    logger.info("✅ Sync Operation Summary:")
    logger.info(f"   - Total Synced: {stats['synced']}")
    logger.info(f"   - Total Skipped (Already there): {stats['skipped']}")
    logger.info(f"   - Total Errors: {stats['errors']}")

if __name__ == "__main__":
    main()
