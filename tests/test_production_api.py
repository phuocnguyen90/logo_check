import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# --- Configuration ---
BASE_URL = os.getenv("RAILWAY_SERVICE_URL", "https://logocheck-production.up.railway.app")
API_KEY = os.getenv("LOGO_API_KEY", "l3d_qEldZyPbHgcFlZAMC5xNca88kQGZyaDbSkOJ1aj09S0")
# Relative path from project root to a test image
TEST_IMAGE = Path(__file__).resolve().parent.parent / "temp_uploads" / "nike.png"

def test_api_health():
    """Verify the production server is up and models are listed."""
    print(f"🔍 Testing Health: {BASE_URL}/health")
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BASE_URL}/health", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        assert data["status"] == "ready"
        print("✅ Health check passed!")
        print(f"   Models Memory: {data.get('models_in_memory', [])}")
        print(f"   Enabled Models: {data.get('enabled_models', [])}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_api_search():
    """Perform a search and validate result structure."""
    if not TEST_IMAGE.exists():
        print(f"⚠️ Skipping search test: Sample image not found at {TEST_IMAGE}")
        return False

    print(f"🔍 Testing Search: {BASE_URL}/v1/search")
    headers = {"X-API-Key": API_KEY}
    params = {"top_k": 5}
    
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": (TEST_IMAGE.name, f, "image/png")}
            response = requests.post(
                f"{BASE_URL}/v1/search", 
                headers=headers, 
                params=params, 
                files=files,
                timeout=20
            )
            
        response.raise_for_status()
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) > 0
        
        # Check first result fields
        best_match = data["results"][0]
        required_fields = ["score", "filename", "proxied_url"]
        for field in required_fields:
            assert field in best_match
            
        print(f"✅ Search test passed! Top match: {best_match['filename']} (Score: {best_match['score']:.4f})")
        return True
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Production API Integration Tests...")
    health_ok = test_api_health()
    search_ok = test_api_search()
    
    if health_ok and search_ok:
        print("\n✨ All production tests passed successfully!")
    else:
        print("\n❌ One or more tests failed.")
        exit(1)
