from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union

# Minimal imports for production
from ..config import settings, paths
from ..utils.logging import logger
from ..utils.s3 import s3_service
from ..api.onnx_inference import ONNXInference
from ..retrieval.vector_store import VectorStore

app = FastAPI(title="Logo Similarity Production API", version="1.0.0")

import asyncio
import uuid
import threading

class APIContext:
    def __init__(self):
        self.inference = None
        self.vector_store = None
        self.id_map_db = None
        self.db_path = Path("data/metadata.db")
        self.lock = threading.Lock() # Protect index/DB during incremental updates
        # Priority: RAILWAY_BUCKET_NAME -> MINIO_BUCKET -> 'l3d'
        self.bucket = os.getenv("RAILWAY_BUCKET_NAME") or os.getenv("MINIO_BUCKET", "l3d")

    def initialize(self):
        """Warm up the API: Download models if missing and load them."""
        # 1. Locate VPS Bundle
        ckpt_stem = "best_model"
        onnx_dir = paths.MODELS_DIR / "onnx" / ckpt_stem
        
        self.model_path = onnx_dir / "model.onnx"
        self.pca_path = onnx_dir / "pca_model.joblib"
        self.index_path = onnx_dir / "vps_index.bin"
        self.id_map_path = onnx_dir / "vps_id_map.db"

        # Required files to download if missing
        required_files = {
            f"models/{ckpt_stem}/model.onnx": self.model_path,
            f"models/{ckpt_stem}/model.onnx.data": onnx_dir / "model.onnx.data",
            f"models/{ckpt_stem}/pca_model.joblib": self.pca_path,
            f"models/{ckpt_stem}/vps_index.bin": self.index_path,
            f"models/{ckpt_stem}/vps_id_map.db": self.id_map_path,
        }

        # 2. Check and Download from S3/MinIO
        for s3_key, local_path in required_files.items():
            if not local_path.exists():
                logger.info(f"🚚 Downloading {s3_key} from {self.bucket}...")
                success = s3_service.download_file(self.bucket, s3_key, local_path)
                if not success:
                    logger.error(f"❌ Failed to download {s3_key}")

        # 3. Load ONNX Inference Engine
        if self.model_path.exists():
            try:
                self.inference = ONNXInference(str(self.model_path), pca_path=str(self.pca_path) if self.pca_path.exists() else None)
                logger.info(f"✅ Loaded ONNX Model")
            except Exception as e:
                logger.error(f"❌ Failed to load ONNX: {e}")

        # 4. Load FAISS
        if self.index_path.exists():
            try:
                self.vector_store = VectorStore(256) 
                self.vector_store.load(str(self.index_path), use_mmap=True)
                logger.info(f"✅ Loaded MMAP Index: {self.index_path}")
                
                if self.id_map_path.exists():
                    self.id_map_db = str(self.id_map_path)
                    logger.info(f"✅ Using SQLite ID Map: {self.id_map_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load Index: {e}")

    def get_filename(self, idx: int) -> str:
        if not self.id_map_db: return str(idx)
        try:
            conn = sqlite3.connect(self.id_map_db)
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM id_map WHERE id = ?", (int(idx),))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else str(idx)
        except Exception as e:
            logger.error(f"ID lookup failed: {e}")
            return str(idx)

    def add_to_index(self, embedding: np.ndarray, filename: str):
        """Thread-safe incremental update and write-back to cloud."""
        with self.lock:
            try:
                # 1. Update In-memory FAISS
                embedding = embedding.astype('float32')
                self.vector_store.index.add(embedding)
                new_id = self.vector_store.index.ntotal - 1
                
                # 2. Update Local SQLite
                conn = sqlite3.connect(self.id_map_db)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO id_map (id, filename) VALUES (?, ?)", (new_id, filename))
                conn.commit()
                conn.close()
                
                # 3. Save Index to disk
                faiss.write_index(self.vector_store.index, str(self.index_path))
                
                # 4. Sync binaries back to S3 (Background synchronization)
                # Note: We do this synchronously here to ensure consistency 
                # but we could background it if upload is slow.
                prefix = "models/best_model"
                s3_service.upload_file(str(self.index_path), self.bucket, f"{prefix}/vps_index.bin")
                s3_service.upload_file(str(self.id_map_path), self.bucket, f"{prefix}/vps_id_map.db")
                
                logger.info(f"🚀 Incremental Index Complete: {filename} mapped to ID {new_id}")
                return True
            except Exception as e:
                logger.error(f"Incremental update failed: {e}")
                return False

# Global context
ctx = APIContext()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(asyncio.to_thread(ctx.initialize))

@app.post("/v1/index")
async def index_image(file: UploadFile = File(...)):
    """Receives a new image, uploads to S3, and adds to FAISS index."""
    if not ctx.inference or not ctx.vector_store:
        raise HTTPException(status_code=503, detail="System initializing.")

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
        embedding = ctx.inference.run(img_bgr)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed")

        success_index = ctx.add_to_index(embedding, new_filename)
        
        if not success_index:
            raise HTTPException(status_code=500, detail="Failed to update index")

        return {
            "status": "indexed",
            "filename": new_filename,
            "s3_key": s3_key,
            "index_total": ctx.vector_store.index.ntotal
        }

    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/v1/search")
async def search_image(file: UploadFile = File(...), top_k: int = 50):
    if not ctx.inference or not ctx.vector_store:
        raise HTTPException(status_code=503, detail="System initializing or files missing.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        embedding = ctx.inference.run(img_bgr)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Inference processing failed")

        distances, indices = ctx.vector_store.search(embedding, k=top_k)

        results = []
        for d, idx in zip(distances, indices):
            if idx == -1: continue
            
            filename = ctx.get_filename(idx)
            
            # S3 lookup (case-insensitive)
            image_key = f"images/{filename.lower()}"
            image_url = s3_service.get_presigned_url(ctx.bucket, image_key)
            
            results.append({
                "score": float(d),
                "filename": filename,
                "image_url": image_url
            })
            
        return {"query_id": file.filename, "results": results}

    except Exception as e:
        logger.exception(f"Search failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health():
    logger.debug("Health check pinged")
    return {
        "status": "ready",
        "model": ctx.inference is not None,
        "index_count": ctx.vector_store.index.ntotal if ctx.vector_store and ctx.vector_store.index else 0
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
