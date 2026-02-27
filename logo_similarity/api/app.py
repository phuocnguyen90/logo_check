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
import asyncio
import time
import gc
import threading
import uuid
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

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
import hashlib
import psutil
LOGO_API_KEY_ENV = os.getenv("LOGO_API_KEY", "dev_key_change_me")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_key: str = Security(api_key_header)):
    # 1. Check Master ENV key (for internal/dev use)
    if header_key == LOGO_API_KEY_ENV:
        return header_key
        
    # 2. Check DB keys (verified via Context)
    if header_key and ctx.verify_api_key(header_key):
        return header_key
        
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )



import asyncio
import time
import gc
import threading
import uuid

class ModelBundle:
    def __init__(self, name: str):
        self.name = name
        self.inference = None
        self.vector_store = None
        self.id_map_db = None
        self.id_map_list = None
        self.index_path = None
        self.id_map_path = None
        self.last_used = time.time()

    def touch(self):
        self.last_used = time.time()

class APIContext:
    def __init__(self):
        self.bundles: Dict[str, ModelBundle] = {}
        self.lock = threading.Lock()
        self.bucket = os.getenv("RAILWAY_BUCKET_NAME") or os.getenv("MINIO_BUCKET", "l3d")
        self.enabled_models = [m.strip() for m in os.getenv("SERVE_MODELS", "best_model").split(",")]
        self.idle_timeout = int(os.getenv("MODEL_IDLE_TIMEOUT", 300)) # Default 5 mins
        
        # Master Metadata & Auth State
        self._auth_cache: Dict[str, bool] = {}
        self.last_activity = time.time()
        self.last_master_access = time.time() # For auth/master DB activity

    def initialize(self):
        """Ensure local files exist by checking/downloading from S3."""
        logger.info(f"💾 Initializing API Context (Enabled: {self.enabled_models})")
        self.touch()
        # 1. Ensure Master Metadata DB is present
        master_db_path = paths.MASTER_METADATA_DB
        if not master_db_path.exists():
            logger.info("🚚 Downloading Master Metadata DB from S3...")
            try:
                s3_service.download_file(self.bucket, f"data/{master_db_path.name}", master_db_path)
            except Exception as e:
                logger.error(f"Failed to download master DB: {e}")

        # 2. Sync enabled models
        for model_id in self.enabled_models:
            self._ensure_files_locally(model_id)

    def touch(self):
        """Updates the global activity timer."""
        self.last_activity = time.time()

    def _ensure_files_locally(self, model_id: str):
        """Purely file-level sync. Does NOT load into memory."""
        onnx_dir = paths.MODELS_DIR / "onnx" / model_id
        onnx_dir.mkdir(parents=True, exist_ok=True)
        
        required_files = {
            f"models/{model_id}/model.onnx": onnx_dir / "model.onnx",
            f"models/{model_id}/model.onnx.data": onnx_dir / "model.onnx.data",
            f"models/{model_id}/pca_model.joblib": onnx_dir / "pca_model.joblib",
            f"models/{model_id}/vps_index.bin": onnx_dir / "vps_index.bin",
            f"models/{model_id}/vps_id_map.db": onnx_dir / "vps_id_map.db",
            f"models/{model_id}/vps_id_map.json": onnx_dir / "vps_id_map.json",
        }

        for s3_key, local_path in required_files.items():
            if not local_path.exists():
                logger.info(f"🚚 Downloading {s3_key} to persistent volume...")
                try:
                    s3_service.download_file(self.bucket, s3_key, local_path)
                except Exception as e:
                    logger.warning(f"Could not download {s3_key}: {e}. (May not exist)")
    def verify_api_key(self, key: str) -> bool:
        """Verified key against cache or DB, then touches activity timer."""
        self.touch()
        
        # 1. Check RAM cache
        if key in self._auth_cache:
            return self._auth_cache[key]
            
        # 2. Check DB
        db_path = paths.MASTER_METADATA_DB
        self._ensure_master_db(db_path)
        
        if not db_path.exists():
            return False
            
        khash = hashlib.sha256(key.encode()).hexdigest()
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM api_keys WHERE key_hash = ? AND is_active = 1", (khash,))
            valid = cursor.fetchone() is not None
            conn.close()
            
            # Cache for future use (will be cleared after 5m idle)
            self._auth_cache[key] = valid
            return valid
        except Exception as e:
            logger.error(f"Auth DB error: {e}")
            return False

    def _ensure_master_db(self, path: Path):
        """Download master DB if missing."""
        if not path.exists():
            logger.info("🚚 Downloading Master Metadata DB from S3...")
            try:
                s3_service.download_file(self.bucket, f"data/{path.name}", path)
            except Exception as e:
                logger.error(f"Failed to download master DB: {e}")

    def get_bundle(self, model_id: str) -> Optional[ModelBundle]:
        """Get or load model bundle on demand."""
        self.touch()
        if model_id not in self.enabled_models:
            return None

        with self.lock:
            if model_id in self.bundles:
                bundle = self.bundles[model_id]
                bundle.touch()
                return bundle

            # Load into memory on demand
            logger.info(f"🧠 Loading model '{model_id}' into memory (on-demand)")
            self._ensure_files_locally(model_id)
            
            onnx_dir = paths.MODELS_DIR / "onnx" / model_id
            bundle = ModelBundle(model_id)
            model_path = onnx_dir / "model.onnx"
            pca_path = onnx_dir / "pca_model.joblib"
            bundle.index_path = onnx_dir / "vps_index.bin"
            bundle.id_map_path = onnx_dir / "vps_id_map.db"
            id_map_json_path = onnx_dir / "vps_id_map.json"

            if model_path.exists():
                bundle.inference = ONNXInference(str(model_path), pca_path=str(pca_path) if pca_path.exists() else None)
            
            if bundle.index_path.exists():
                bundle.vector_store = VectorStore(256)
                bundle.vector_store.load(str(bundle.index_path), use_mmap=True)
                
                if bundle.id_map_path.exists():
                    bundle.id_map_db = str(bundle.id_map_path)
                elif id_map_json_path.exists():
                    try:
                        with open(id_map_json_path, 'r') as f:
                            bundle.id_map_list = json.load(f)
                    except: pass
            
            self.bundles[model_id] = bundle
            return bundle

    async def cleanup_loop(self):
        """Background task to offload inactive models and caches."""
        logger.info(f"⏲️ Starting inactivity cleanup loop (Timeout: {self.idle_timeout}s)")
        while True:
            await asyncio.sleep(60) # Check every minute
            now = time.time()
            
            # 1. Check Global Activity
            is_idle = (now - self.last_activity) > self.idle_timeout
            
            if is_idle:
                with self.lock:
                    if self._auth_cache or self.bundles:
                        logger.info("❄️ Service idle: Offloading all models and clearing auth cache.")
                        self._auth_cache.clear()
                        
                        for mid in list(self.bundles.keys()):
                            bundle = self.bundles.pop(mid)
                            del bundle.inference
                            del bundle.vector_store
                            del bundle
                            
                        # Aggressive GC
                        gc.collect()
                        logger.info("🧹 RAM footprint minimized.")
            else:
                # 2. Selective offloading (if some models are idle but others aren't)
                to_unload = []
                with self.lock:
                    for mid, bundle in self.bundles.items():
                        if now - bundle.last_used > self.idle_timeout:
                            to_unload.append(mid)
                    
                    for mid in to_unload:
                        logger.info(f"❄️ Offloading inactive model: {mid}")
                        bundle = self.bundles.pop(mid)
                        del bundle
                
                if to_unload:
                    gc.collect()


    def get_metadata_batch(self, bundle: ModelBundle, indices: List[int]) -> List[Dict[str, Any]]:
        """Bulk lookup for search results (SQLite first, JSON fallback)."""
        if not bundle.id_map_db and not bundle.id_map_list:
            return [{"filename": str(idx)} for idx in indices]
        
        # Scenario A: SQLite lookup (Rich Metadata)
        if bundle.id_map_db:
            results = {}
            try:
                conn = sqlite3.connect(bundle.id_map_db)
                cursor = conn.cursor()
                placeholders = ",".join(["?"] * len(indices))
                # Fetch all rich metadata columns
                cursor.execute(f"SELECT id, filename, tm_text, vienna_codes, year FROM id_map WHERE id IN ({placeholders})", indices)
                for row in cursor.fetchall():
                    results[row[0]] = {
                        "filename": row[1],
                        "text": row[2],
                        "vienna_codes": row[3].split(",") if row[3] else [],
                        "year": row[4]
                    }
                conn.close()
                return [results.get(idx, {"filename": str(idx)}) for idx in indices]
            except Exception as e:
                logger.error(f"Batch metadata lookup failed: {e}")

        # Scenario B: JSON List fallback (Filename only)
        if bundle.id_map_list:
            res = []
            for idx in indices:
                try:
                    res.append({"filename": bundle.id_map_list[idx]})
                except (IndexError, TypeError):
                    res.append({"filename": str(idx)})
            return res
            
        return [{"filename": str(idx)} for idx in indices]

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
    # Sync files from S3 to volume (in thread to avoid blocking)
    asyncio.create_task(asyncio.to_thread(ctx.initialize))
    # Start the inactivity offloader
    asyncio.create_task(ctx.cleanup_loop())

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

        # Bulk ID Map lookup (Now returns rich metadata)
        valid_indices = [int(idx) for idx in indices if idx != -1]
        metadata_results = ctx.get_metadata_batch(bundle, valid_indices)

        results = []
        for d, meta in zip(lengths, metadata_results):
            filename = meta["filename"]
            res = {
                "score": float(d),
                "filename": filename,
                "text": meta.get("text", ""),
                "vienna_codes": meta.get("vienna_codes", []),
                "year": meta.get("year"),
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
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    
    return {
        "status": "ready",
        "memory_usage_mb": round(ram_mb, 2),
        "models_in_memory": list(ctx.bundles.keys()),
        "enabled_models": ctx.enabled_models,
        "details": {
            name: {"loaded": name in ctx.bundles, "count": ctx.bundles[name].vector_store.index.ntotal if name in ctx.bundles and ctx.bundles[name].vector_store else 0}
            for name in ctx.enabled_models
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
