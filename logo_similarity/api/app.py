from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
import uvicorn
import cv2
import numpy as np
import json
import os
import sqlite3
import faiss
from pathlib import Path
from typing import List, Dict, Any, Union

# Minimal imports for production
from ..config import settings, paths
from ..utils.logging import logger
from ..utils.s3 import s3_service
from ..api.onnx_inference import ONNXInference
from ..retrieval.vector_store import VectorStore

app = FastAPI(title="Logo Similarity Production API", version="1.0.0")

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. API Key Security
API_KEY = os.getenv("API_KEY", "dev_key_change_me")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY:
        return header_key
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )

import asyncio
import uuid
import threading

class ModelBundle:
    def __init__(self, name: str):
        self.name = name
        self.inference = None
        self.vector_store = None
        self.id_map_db = None
        self.id_map_list = None # Fallback for JSON mapping
        self.index_path = None
        self.id_map_path = None

class APIContext:
    def __init__(self):
        self.bundles: Dict[str, ModelBundle] = {}
        self.lock = threading.Lock()
        self.bucket = os.getenv("RAILWAY_BUCKET_NAME") or os.getenv("MINIO_BUCKET", "l3d")
        # List of models to load on startup, e.g. "best_model,semantic_v1"
        self.enabled_models = os.getenv("SERVE_MODELS", "best_model").split(",")

    def initialize(self):
        """Warm up all enabled models."""
        for model_id in self.enabled_models:
            model_id = model_id.strip()
            self.load_bundle(model_id)

    def load_bundle(self, model_id: str):
        """Download and load a specific model bundle."""
        logger.info(f"🏗 Preparing bundle: {model_id}")
        onnx_dir = paths.MODELS_DIR / "onnx" / model_id
        
        bundle = ModelBundle(model_id)
        model_path = onnx_dir / "model.onnx"
        pca_path = onnx_dir / "pca_model.joblib"
        bundle.index_path = onnx_dir / "vps_index.bin"
        bundle.id_map_path = onnx_dir / "vps_id_map.db"
        id_map_json_path = onnx_dir / "vps_id_map.json"

        # 1. Download if missing
        required_files = {
            f"models/{model_id}/model.onnx": model_path,
            f"models/{model_id}/model.onnx.data": onnx_dir / "model.onnx.data",
            f"models/{model_id}/pca_model.joblib": pca_path,
            f"models/{model_id}/vps_index.bin": bundle.index_path,
            f"models/{model_id}/vps_id_map.db": bundle.id_map_path,
            f"models/{model_id}/vps_id_map.json": id_map_json_path,
        }

        for s3_key, local_path in required_files.items():
            if not local_path.exists():
                logger.info(f"🚚 Downloading {s3_key}...")
                s3_service.download_file(self.bucket, s3_key, local_path)

        # 2. Load Inference
        if model_path.exists():
            bundle.inference = ONNXInference(str(model_path), pca_path=str(pca_path) if pca_path.exists() else None)
        
        # 3. Load Index & ID Map
        if bundle.index_path.exists():
            bundle.vector_store = VectorStore(256)
            bundle.vector_store.load(str(bundle.index_path), use_mmap=True)
            
            # Prefer SQLite for efficiency
            if bundle.id_map_path.exists():
                bundle.id_map_db = str(bundle.id_map_path)
            # Fallback to JSON if DB is missing
            elif id_map_json_path.exists():
                logger.info(f"📂 Loading fallback JSON ID map for {model_id}...")
                try:
                    with open(id_map_json_path, 'r') as f:
                        bundle.id_map_list = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load JSON ID map: {e}")
            
            self.bundles[model_id] = bundle
            logger.info(f"✅ Bundle '{model_id}' ready.")

    def get_bundle(self, model_id: str = "best_model") -> ModelBundle:
        return self.bundles.get(model_id)

    def get_filenames(self, bundle: ModelBundle, indices: List[int]) -> List[str]:
        """Bulk lookup for search results (SQLite first, JSON fallback)."""
        if not bundle.id_map_db and not bundle.id_map_list:
            return [str(idx) for idx in indices]
        
        # Scenario A: SQLite lookup
        if bundle.id_map_db:
            results = {}
            try:
                conn = sqlite3.connect(bundle.id_map_db)
                cursor = conn.cursor()
                placeholders = ",".join(["?"] * len(indices))
                cursor.execute(f"SELECT id, filename FROM id_map WHERE id IN ({placeholders})", indices)
                results = {row[0]: row[1] for row in cursor.fetchall()}
                conn.close()
                return [results.get(idx, str(idx)) for idx in indices]
            except Exception as e:
                logger.error(f"Batch ID lookup failed: {e}")

        # Scenario B: JSON List fallback
        if bundle.id_map_list:
            res = []
            for idx in indices:
                try:
                    res.append(bundle.id_map_list[idx])
                except (IndexError, TypeError):
                    res.append(str(idx))
            return res
            
        return [str(idx) for idx in indices]

    def add_to_bundle_index(self, bundle: ModelBundle, embedding: np.ndarray, filename: str):
        """Thread-safe incremental update and write-back to cloud for a specific bundle."""
        with self.lock:
            try:
                # 1. Update In-memory FAISS
                embedding = embedding.astype('float32')
                bundle.vector_store.index.add(embedding)
                new_id = bundle.vector_store.index.ntotal - 1
                
                # 2. Update Local SQLite
                conn = sqlite3.connect(bundle.id_map_db)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO id_map (id, filename) VALUES (?, ?)", (new_id, filename))
                conn.commit()
                conn.close()
                
                # 3. Save Index to disk
                faiss.write_index(bundle.vector_store.index, str(bundle.index_path))
                
                # 4. Sync binaries back to S3
                prefix = f"models/{bundle.name}"
                s3_service.upload_file(str(bundle.index_path), self.bucket, f"{prefix}/vps_index.bin")
                s3_service.upload_file(str(bundle.id_map_path), self.bucket, f"{prefix}/vps_id_map.db")
                
                logger.info(f"🚀 Incremental Index Complete [{bundle.name}]: {filename} -> ID {new_id}")
                return True
            except Exception as e:
                logger.error(f"Incremental update failed for {bundle.name}: {e}")
                return False

# Global context
ctx = APIContext()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(asyncio.to_thread(ctx.initialize))

@app.post("/v1/index")
async def index_image(file: UploadFile = File(...), model: str = "best_model", api_key: str = Depends(get_api_key)):
    """Receives a new image, uploads to S3, and adds to FAISS index of a specific model."""
    bundle = ctx.get_bundle(model)
    if not bundle or not bundle.inference or not bundle.vector_store:
        raise HTTPException(status_code=503, detail=f"Model bundle '{model}' not ready or missing.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # 1. Generate unique filename and upload to S3
        ext = Path(file.filename).suffix or ".jpg"
        new_filename = f"user_{uuid.uuid4().hex}{ext}"
        s3_key = f"images/{new_filename.lower()}"
        
        # Save temp for upload
        temp_path = Path(f"/tmp/{new_filename}")
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        success_s3 = s3_service.upload_file(str(temp_path), ctx.bucket, s3_key, content_type="image/jpeg")
        temp_path.unlink() # Cleanup

        if not success_s3:
            raise HTTPException(status_code=500, detail="Failed to upload image to storage")

        # 2. Extract embedding and add to vector store
        embedding = bundle.inference.run(img_bgr)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed")

        # Update specific bundle
        success_index = ctx.add_to_bundle_index(bundle, embedding, new_filename)
        
        if not success_index:
            raise HTTPException(status_code=500, detail="Failed to update index")

        return {
            "status": "indexed",
            "model": model,
            "filename": new_filename,
            "s3_key": s3_key,
            "index_total": bundle.vector_store.index.ntotal
        }

    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/v1/search")
async def search_image(
    file: UploadFile = File(...), 
    top_k: int = 50, 
    model: str = "best_model", 
    include_url: bool = False,
    api_key: str = Depends(get_api_key)
):
    bundle = ctx.get_bundle(model)
    if not bundle or not bundle.inference or not bundle.vector_store:
        raise HTTPException(status_code=503, detail=f"Model bundle '{model}' not ready or missing.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        embedding = bundle.inference.run(img_bgr)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Inference processing failed")

        lengths, indices = bundle.vector_store.search(embedding, k=top_k)

        # Bulk ID Map lookup
        valid_indices = [int(idx) for idx in indices if idx != -1]
        filenames = ctx.get_filenames(bundle, valid_indices)

        results = []
        for d, filename in zip(lengths, filenames):
            res = {
                "score": float(d),
                "filename": filename,
                "proxied_url": f"/v1/image/{filename}"
            }
            
            # Optional: S3 presigned URL
            if include_url:
                image_key = f"images/{filename.lower()}"
                res["image_url"] = s3_service.get_presigned_url(ctx.bucket, image_key)
            
            results.append(res)
            
        return {"query_id": file.filename, "model": model, "results": results}

    except Exception as e:
        logger.exception(f"Search failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/v1/image/{filename}")
async def get_image(filename: str):
    """Proxy image delivery from S3 to bypass browser/credential issues."""
    image_key = f"images/{filename.lower()}"
    try:
        response = s3_service.get_object(ctx.bucket, image_key)
        if not response:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return StreamingResponse(
            response['Body'], 
            media_type=response.get('ContentType', 'image/jpeg')
        )
    except Exception as e:
        logger.error(f"Failed to serve image {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health():
    return {
        "status": "ready",
        "models_loaded": list(ctx.bundles.keys()),
        "details": {
            name: {"ready": b.inference is not None, "count": b.vector_store.index.ntotal if b.vector_store else 0}
            for name, b in ctx.bundles.items()
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
