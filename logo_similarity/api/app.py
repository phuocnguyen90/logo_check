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

class APIContext:
    def __init__(self):
        self.inference = None
        self.vector_store = None
        self.id_map = []
        self.db_path = Path("data/metadata.db")
        self.bucket = os.getenv("MINIO_BUCKET", "l3d")

    def initialize(self):
        # 1. Locate VPS Bundle
        ckpt_stem = "best_model"
        onnx_dir = paths.MODELS_DIR / "onnx" / ckpt_stem
        
        model_path = onnx_dir / "model.onnx"
        pca_path = onnx_dir / "pca_model.joblib"
        index_path = onnx_dir / "vps_index.bin"
        id_map_path = onnx_dir / "vps_id_map.json"

        # 2. Load ONNX Inference Engine
        if model_path.exists():
            self.inference = ONNXInference(str(model_path), pca_path=str(pca_path) if pca_path.exists() else None)
            logger.info(f"✅ Loaded ONNX Model: {model_path}")
        else:
            logger.error(f"❌ Model not found: {model_path}")

        # 3. Load FAISS (Memory Mapped for Serverless speed)
        if index_path.exists():
            # Initial dim doesn't strictly matter if we use .load() as it overwrites index
            self.vector_store = VectorStore(256) 
            self.vector_store.load(str(index_path), use_mmap=True)
            logger.info(f"✅ Loaded MMAP Index: {index_path} (dim={self.vector_store.dimension})")
            
            if id_map_path.exists():
                with open(id_map_path, "r") as f:
                    self.id_map = json.load(f)
        else:
            logger.error(f"❌ Index not found: {index_path}")

# Global context
ctx = APIContext()

@app.on_event("startup")
async def startup_event():
    ctx.initialize()

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
            
            if ctx.id_map and idx < len(ctx.id_map):
                filename = ctx.id_map[idx]
            else:
                filename = str(idx)
            
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
    return {
        "status": "ready",
        "model": ctx.inference is not None,
        "index_count": ctx.vector_store.index.ntotal if ctx.vector_store and ctx.vector_store.index else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
