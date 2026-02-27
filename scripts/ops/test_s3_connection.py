import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from logo_similarity.utils.s3 import s3_service
from logo_similarity.utils.logging import logger

def test_railway_bucket():
    load_dotenv()
    
    bucket = os.getenv("RAILWAY_BUCKET_NAME")
    print(f"\n🔍 Testing Railway Bucket: {bucket}")
    print(f"🔗 Endpoint: {os.getenv('RAILWAY_BUCKET_URL')}")
    
    if not bucket:
        print("❌ Error: RAILWAY_BUCKET_NAME not set in .env")
        return

    try:
        # List objects in the bucket
        print(f"📂 Listing files in {bucket}...")
        response = s3_service.client.list_objects_v2(Bucket=bucket)
        
        # List only models specifically
        print(f"📂 Specifically checking for 'models/' prefix...")
        models_resp = s3_service.client.list_objects_v2(Bucket=bucket, Prefix='models/')
        
        if 'Contents' in models_resp:
            print(f"✅ Found {len(models_resp['Contents'])} objects in models/ folder:")
            for obj in models_resp['Contents']:
                print(f"  - {obj['Key']} ({obj['Size'] / 1024 / 1024:.2f} MB)")
        else:
            print("❌ No files found with 'models/' prefix.")
            
        if 'Contents' in response:
            print(f"\n✅ Total objects in root listing (limit 1000): {len(response['Contents'])}")

                
        else:
            print(f"⚠️ Bucket {bucket} is empty or not accessible.")
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_railway_bucket()
