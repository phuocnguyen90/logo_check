# 🚀 Logo Similarity Inference API Guide

This document describes how to use the Logo Similarity Inference API and how to interact with the underlying S3-compatible storage (Tigris on Railway) for displaying search results.

---

## 📍 API Base URL
- **Production**: `https://logocheck-production.up.railway.app`
- **Health Check**: `GET /health`

---

## 🔒 Security & Authentication

All endpoints (except `/health`) require an API key passed in the request header.

- **Header Name**: `X-API-Key`
- **Value**: Your secret API key (configured in environment variables).

---

---

## 🛠 Endpoints

### 1. Health Check (Public)
Checks if models and their respective indexes are loaded.

- **URL**: `/health`
- **Method**: `GET`
- **Response**:
    ```json
    {
      "status": "ready",
      "models_loaded": ["best_model", "semantic_v1_semantic_epoch_30"],
      "details": {
        "best_model": {"ready": true, "count": 769674}
      }
    }
    ```

### 2. Image Search (Protected)
- **URL**: `/v1/search`
- **Method**: `POST`
- **Headers**: `X-API-Key: <YOUR_KEY>`
- **Query Parameters**: 
    - `top_k` (Optional): Number of results.
    - `model` (Optional): Model ID (e.g. `best_model`).
    - `include_url` (Optional, `bool`): If `true`, returns a short-lived presigned URL for each result.
- **Parameters**: `file` (Binary image)
- **Response**:
    ```json
    {
      "query_id": "apple.png",
      "model": "best_model",
      "results": [
        {
          "id": "775fc06d-0088-4e2f-a4f6.jpg",
          "score": 0.8521
        }
      ]
    }
    ```
    *`id` is the unique filename stored in S3. Use this to perform retrieval in your consuming service.*

### 3. Incremental Indexing (Protected)
- **URL**: `/v1/index`
- **Method**: `POST`
- **Headers**: `X-API-Key: <YOUR_KEY>`
- **Parameters**: 
    - `file`: Image to index.
    - `model` (Optional): Target model/index. Default is `best_model`.

---

## 🎯 Model Selection
The system supports multiple specialized models:
1. `best_model`: Optimized for **Visual Similarity** (shape, color, icons).
2. `semantic_v1_semantic_epoch_30`: Optimized for **Figurative/Semantic Similarity** (contextual meaning).

Example usage for semantic search:
`POST /v1/search?model=semantic_v1_semantic_epoch_30`

---

## 💻 Code Examples (Authorized)

### 🐍 Python
```python
import requests

headers = {"X-API-Key": "YOUR_SECRET_KEY"}
files = {'file': open('your_image.jpg', 'rb')}
# Example with query parameters
# url = "https://logocheck-production.up.railway.app/v1/search?top_k=5&model=best_model&include_url=true"
url = "https://logocheck-production.up.railway.app/v1/search"

response = requests.post(url, headers=headers, files=files)
if response.status_code == 200:
    results = response.json()
    print(f"Query ID: {results['query_id']}")
    print(f"Model: {results['model']}")
    for item in results['results']:
        print(f"  ID: {item['id']}, Score: {item['score']}")
        if 'url' in item: # if include_url=true was used
            print(f"  URL: {item['url']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### 🌐 JavaScript
```javascript
const url = 'https://logocheck-production.up.railway.app/v1/search';
// Example with query parameters
// const url = 'https://logocheck-production.up.railway.app/v1/search?top_k=5&model=best_model&include_url=true';

const formData = new FormData();
// Assuming 'imageFile' is a File object from an input element
// formData.append('file', imageFile);

// For demonstration, creating a dummy blob
const dummyBlob = new Blob(['dummy image data'], { type: 'image/jpeg' });
formData.append('file', dummyBlob, 'dummy_image.jpg');


const response = await fetch(url, {
  method: 'POST',
  headers: {
    'X-API-Key': 'YOUR_SECRET_KEY'
  },
  body: formData
});

if (response.ok) {
  const results = await response.json();
  console.log(`Query ID: ${results.query_id}`);
  console.log(`Model: ${results.model}`);
  results.results.forEach(item => {
    console.log(`  ID: ${item.id}, Score: ${item.score}`);
    if (item.url) { // if include_url=true was used
      console.log(`  URL: ${item.url}`);
    }
  });
} else {
  console.error(`Error: ${response.status} - ${await response.text()}`);
}
```

---

## 🖼 Image Retrieval Strategy

The API returns a clean `id` (e.g., `500538.jpg`). Consuming services should use this ID to interact with the S3 bucket directly using their own credentials.

- **Bucket Key Path**: `images/{id.lower()}`
- **S3 Endpoint**: `https://t3.storageapi.dev`

### Python Example (Independent Retrieval)
```python
results = requests.post(SEARCH_URL, headers=headers, files=files).json()

for item in results['results']:
    filename = item['id']
    # Consuming service generates its own URL or fetches the bytes
    s3_url = f"https://your-own-proxy.com/images/{filename}"
```

---

## 🏗 System Architecture Details
For developers maintaining this service:
- **Inference Engine**: ONNX Runtime (CPU) utilizing EfficientNet-B0 embeddings + PCA (256-dim).
- **Vector Search**: FAISS IndexFlatIP (Brute Force) memory-mapped for zero-overhead startup and low RAM footprint.
- **ID Mapping**: SQLite-based lookup (`vps_id_map.db`) on disk to handle millions of mappings without memory bloat.
- **Provisioning**: The API automatically downloads model assets from S3 on startup if not present locally (background warm-up).
