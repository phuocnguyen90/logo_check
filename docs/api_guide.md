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
      "filename": "775fc06d-0088-4e2f-a4f6-f17cbc35772a.JPG",
      "score": 0.9852,
      "text": "CytoView",
      "vienna_codes": ["27.05.01"],
      "year": 2016,
      "image_url": "https://...", 
      "proxied_url": "/v1/image/775fc06d-0088-4e2f-a4f6-f17cbc35772a.JPG"
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

### 3. Image Proxy (Public)
Serve images directly from S3 without requiring client-side credentials or worrying about presigned URL expiration.

- **Endpoint**: `GET /v1/image/{filename}`
- **Parameters**: `filename` (String)

---

## 🖼 How to Render Search Results (Recommended)

To ensure high performance and avoid authentication issues in the browser, you should use the provided `proxied_url`.

### The Problem
Generating presigned URLs in our backend and passing them to your frontend can cause:
1. **Latency**: Each URL generation adds overhead.
2. **CORS/Auth**: Browsers often block S3 presigned URLs due to cross-origin policies or lack of direct credentials.

### The Solution: Using the Image Proxy
The API provides a `proxied_url` (e.g., `/v1/image/apple.jpg`) for every result. This endpoint handles the S3 connection on the server side and streams the image directly to the client. **This endpoint is public** and does not require an API key, making it perfect for tags like `<img src="..." />`.

#### 1. Frontend React/Next.js Example
The easiest way to display results:

```javascript
// Inside your Result component
const LogoThumb = ({ proxied_url, filename }) => {
  // Combine the Base URL with the relative proxied_url
  const src = `https://logocheck-production.up.railway.app${proxied_url}`;

  return (
    <div className="logo-card">
      <img src={src} alt={filename} loading="lazy" />
      <p>Similarity: {filename}</p>
    </div>
  );
};
```

#### 2. Backend Implementation (Advanced)
If you prefer to handle S3 retrieval in your own backend:

```python
# Pseudo-code for a consuming service
def get_search_results():
    api_resp = requests.post(SEARCH_URL, headers={"X-API-Key": KEY}, files=files)
    results = api_resp.json()["results"]
    
    for item in results:
        # The 'filename' is the exact key name in S3
        filename = item['filename']
        # Construct the key for your own S3 client
        s3_key = f"images/{filename.lower()}"
        # Fetch or Presign using YOUR service's own credentials
        authorized_url = my_s3_client.presign(s3_key)
        item["my_custom_url"] = authorized_url
        
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

1. **Filenames**: The `filename` returned is the **exact filename** as it exists in the S3 bucket.
2. **Case Sensitivity**: While S3 is case-sensitive, our system primarily uses lowercase filenames in the bucket.
3. **Caching**: We recommend caching the search results for identical query images (using a file hash) for at least 1 hour.
4. **Rate Limiting**: Currently set to 60 requests per minute on protected endpoints.
