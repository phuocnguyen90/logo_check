# Production Serving Plan: Logo Similarity System

## 1. Overview
The serving system is designed to provide high-performance, low-latency logo search and image retrieval. It leverages FastAPI for the API layer, ONNX for model inference, FAISS for vector search, and MinIO for reliable image storage.

## 2. System Architecture

### 2.1 API Layer (FastAPI)
- **Framework**: FastAPI (Asynchronous, Type-safe).
- **Inference Pipeline**:
    - **Preprocessing**: Fast resizing and normalization using OpenCV.
    - **Embedding Generation**: Multi-threaded ONNX inference (CUDA-accelerated if available).
    - **Vector Search**: FAISS ANN search (HNSW index) for < 20ms retrieval over the 770k dataset.
    - **Re-ranking**: Optional second-stage reranking for top-K candidates.
- **Routing & Serving**:
    - Acts as a "routing middleware" for MinIO.
    - Handles pre-signed URL generation for secure, direct image access by clients.

### 2.2 Storage Layer (MinIO)
- S3-compatible storage hosting the full dataset.
- Optimized for serving high volumes of small image files.
- Accessible via the `S3Service` utility.

### 2.3 Metadata Management
- **Lookup Engine**: Lightweight SQLite or optimized In-Memory cache for mapping FAISS results back to trademark metadata (Company, Class, Vienna Codes).
- **Schema**:
    - `id`: Primary key (mapped to FAISS ID).
    - `filename`: Key for S3 retrieval.
    - `metadata`: JSON blob or relational columns for trademark details.

## 3. API Endpoints

### `POST /v1/search`
- **Purpose**: Search for similar logos using an uploaded image.
- **Input**: `multipart/form-data` containing `file` (image).
- **Processing**:
    1. Preprocess query image.
    2. Compute embedding.
    3. Retrieve top-100 candidates from FAISS.
    4. Enrich with metadata and generated URLs.
- **Output**:
```json
{
  "query_id": "uuid-...",
  "results": [
    {
      "score": 0.985,
      "id": "US-123456",
      "image_url": "http://192.168.1.98:9000/l3d/images/abc.jpg?...",
      "metadata": {
        "company": "Example Corp",
        "vienna_codes": ["26.01.01"],
        "application_date": "2023-01-01"
      }
    }
  ]
}
```

### `GET /v1/image/{image_id}`
- **Purpose**: Act as a router/proxy for images.
- **Logic**: Generates a short-lived Pre-signed URL for the requested `image_id` and performs a `307 Temporary Redirect`.
- **Benefit**: Clients don't need S3 credentials; they just need to call the API.

## 4. Implementation Steps

1.  **Metadata Database**: Convert `results.json` into a SQLite database (`metadata.db`) for fast indexed lookups by FAISS integer IDs.
2.  **API Core**: Flesh out `logo_similarity/api/app.py` with the full pipeline.
3.  **S3 Integration**: Connect `S3Service` for URL generation.
4.  **Deployment**:
    - Build a production Docker image using `uvicorn` with `gunicorn` workers.
    - Configure resource limits (RAM for FAISS index, GPU for ONNX).
5.  **Benchmarking**: Test end-to-end latency for a 50-user concurrent load.

## 5. Scalability & Resilience
- **Workers**: Horizontal scaling of FastAPI containers.
- **Index Loading**: FAISS index remains in memory (or shared memory) for peak performance.
- **Storage**: MinIO scaled as a cluster for high throughput.
- **Security**: Optional API Key validation in a FastAPI middleware.
