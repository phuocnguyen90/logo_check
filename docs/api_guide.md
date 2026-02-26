# 🚀 Logo Similarity Inference API: Full Integration Guide

This guide describes how to consume the Logo Similarity API and how to integrate the search results into your own frontend or microservices.

---

## 📍 API Base Details
- **Base URL**: `https://logocheck-production.up.railway.app`
- **Uptime/Health**: `GET /health`

---

## 🔒 Authentication

All protected endpoints require an API key in the request header.

- **Header Name**: `X-API-Key`
- **Value**: `E5SHJibgZwNo+3pT1gRc4wL/Xv6E2BE0nF6wdj1PmUc=` (Production)

---

## 🛠 Core Endpoints

### 1. Visual/Semantic Search
Perform a visual search using an image file.

- **Endpoint**: `POST /v1/search`
- **Content-Type**: `multipart/form-data`
- **Payload**:
  - `file`: The query image (binary).
- **Query Parameters**:
  - `top_k` (Int, default=50): Number of matches to return.
  - `model` (String, optional): Choose between `best_model` (Visual) or `semantic_v1_semantic_epoch_30` (Semantic).
  - `include_url` (Bool, default=false): Set to `true` to get temporary presigned AWS S3 URLs in the response.

#### JSON Response Structure
```json
{
  "query_id": "upload.png",
  "model": "best_model",
  "results": [
    {
      "id": "nike_logo_1971.jpg",
      "score": 0.9852,
      "image_url": "Optional: https://..." 
    }
  ]
}
```

### 2. Live Indexing
Add a new logo to the database and vector index in real-time.

- **Endpoint**: `POST /v1/index`
- **Payload**:
  - `file`: The logo to register.
- **Query Parameters**:
  - `model`: Which index to update (defaults to `best_model`).

---

## 🖼 How to Render Search Results (Recommended)

To ensure high performance and avoid authentication issues in the browser, follow this **ID-to-Storage** strategy.

### The Problem
Generating presigned URLs in our backend and passing them to your frontend can cause:
1. **Latency**: Each URL generation adds overhead.
2. **CORS/Auth**: Browsers might block these URLs if not configured for your specific frontend domain.

### The Solution: Direct S3 Access
Your frontend or proxy service should use the returned `id` (which is the exact filename) to construct the final image URL using your own AWS/S3 credentials OR a public CDN proxy.

#### 1. Constructing the S3 Path
The logos are stored in a flat structure under the `images/` prefix.
- **Result ID**: `775fc06d.jpg`
- **S3 Key**: `images/775fc06d.jpg` (Note: Ensure the filename is lowercased if not found).

#### 2. Frontend React/Next.js Example
If your web application has a backend proxy for images:

```javascript
// Inside your Result component
const LogoThumb = ({ id }) => {
  // Option A: Your own proxy that handles S3 auth
  const src = `https://your-api.com/proxy/logos/${id}`;
  
  // Option B: Direct S3 URL (if you have a PUBLIC bucket or handle auth on frontend)
  const directPath = `https://t3.storageapi.dev/l3d-bucket/images/${id}`;

  return <img src={src} alt="logo match" loading="lazy" />;
};
```

#### 3. Backend Implementation (Node.js/Boto3)
If you are proxying the images from another service:

```python
# Pseudo-code for a consuming service
def get_search_results():
    api_resp = requests.post(SEARCH_URL, headers={"X-API-Key": KEY}, files=files)
    results = api_resp.json()["results"]
    
    for item in results:
        # Construct the key for your own S3 client
        s3_key = f"images/{item['id'].lower()}"
        # Fetch or Presign using YOUR service's IAM permissions
        authorized_url = my_s3_client.presign(s3_key)
        item["view_url"] = authorized_url
        
    return results
```

---

## 🎯 Model Selection Guide

| Model ID | Best For... | Description |
| :--- | :--- | :--- |
| `best_model` | **Visual Overlap** | Detects similar shapes, colors, and iconic structures. Best for "Logo Lookalikes". |
| `semantic_v1_...` | **Semantic Context** | Detects similar figurative meanings (e.g., 2 different drawings of an Eagle). |

---

## ⚠️ Important Integration Notes

1. **Filenames**: The `id` returned is the **exact filename** as it exists in the S3 bucket.
2. **Case Sensitivity**: While S3 is case-sensitive, our system primarily uses lowercase filenames. If a lookup fails, try `id.lower()`.
3. **Caching**: We recommend caching the search results for identical query images (using a file hash) for at least 1 hour.
4. **Rate Limiting**: Currently set to 60 requests per minute. If you need more, please contact the infra team.
