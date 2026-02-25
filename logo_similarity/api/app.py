from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
import cv2
import numpy as np
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import torch

from ..config import settings, paths
from ..utils.logging import logger
from ..utils.s3 import s3_service
from ..preprocessing.pipeline import PreprocessingPipeline
from ..api.onnx_inference import ONNXInference
from ..retrieval.vector_store import VectorStore
from ..embeddings.pca_reducer import PCAReducer
from ..reranking.reranker import ReRanker
from ..reranking.composite_scorer import CompositeScoringPipeline

app = FastAPI(title="Logo Similarity Production API", version="1.0.0")

# Global state for components
class APIContext:
    def __init__(self):
        self.pipeline = PreprocessingPipeline()
        self.inference = None
        self.vector_store = None
        self.pca = None
        self.scorer = CompositeScoringPipeline()
        self.db_path = Path("data/metadata.db")
        self.bucket = os.getenv("MINIO_BUCKET", "l3d")

    def initialize(self):
        # 1. ONNX Model
        model_path = paths.ONNX_DIR / "model_fp16.onnx"
        if model_path.exists():
            self.inference = ONNXInference(str(model_path))
            logger.info(f"Loaded ONNX model from {model_path}")
        else:
            logger.warning(f"ONNX model NOT found at {model_path}. Search will be disabled.")

        # 2. FAISS Index
        self.vector_store = VectorStore(settings.REDUCED_DIM)
        index_path = paths.EMBEDDING_INDEX_DIR / "faiss_index.bin"
        if index_path.exists():
            self.vector_store.load(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path}")
        else:
            logger.warning(f"FAISS index NOT found at {index_path}. Search will be disabled.")

        # 3. PCA Model (Optional but recommended in config)
        pca_path = paths.MODELS_DIR / "pca_model.joblib"
        if pca_path.exists():
            self.pca = PCAReducer.load(str(pca_path))
            logger.info(f"Loaded PCA model from {pca_path}")

        # 4. Metadata DB
        if not self.db_path.exists():
            logger.warning(f"Metadata database NOT found at {self.db_path}. Results will lack info.")

    def get_metadata(self, ids: List[int]) -> List[Dict]:
        """Fetch metadata for a list of integer IDs from SQLite."""
        if not self.db_path.exists():
            return [{} for _ in ids]
        
        results = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Construct query for multiple IDs
            query = f"SELECT id, filename, company, vienna_codes, year, raw_json FROM metadata WHERE id IN ({','.join(['?']*len(ids))})"
            cursor.execute(query, ids)
            rows = cursor.fetchall()
            
            # Map by ID for easy lookup
            row_map = {r[0]: {
                "id": r[0],
                "filename": r[1],
                "company": r[2],
                "vienna_codes": json.loads(r[3]),
                "year": r[4],
                "details": json.loads(r[5])
            } for r in rows}
            
            # Maintain input order
            results = [row_map.get(idx, {"id": idx, "error": "not found"}) for idx in ids]
            conn.close()
        except Exception as e:
            logger.error(f"Metadata lookup failed: {e}")
            results = [{"id": idx, "error": str(e)} for idx in ids]
            
        return results

# Initialize context
ctx = APIContext()

@app.on_event("startup")
async def startup_event():
    ctx.initialize()

@app.post("/v1/search")
async def search_image(file: UploadFile = File(...), top_k: int = 50):
    """
    Search for similar logos.
    Returns matches with metadata and secure image URLs.
    """
    if not ctx.inference or not ctx.vector_store:
        raise HTTPException(status_code=503, detail="Inference engine or Index not initialized. Please build the index first.")

    try:
        # 1. Load and Preprocess
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Basic cleanup/normalization as per pipeline
        # (Using minimal manual step for speed)
        from ..preprocessing.image_normalizer import ImageNormalizer
        normalizer = ImageNormalizer(target_size=settings.IMG_SIZE)
        normalized = normalizer.normalize(img_bgr)
        
        # 2. Vectorization (ONNX)
        inp_tensor = ctx.inference.preprocess_tensor(normalized)
        batch = np.expand_dims(inp_tensor, axis=0)
        embedding = ctx.inference.run(batch)
        
        # 3. PCA Reduction
        if ctx.pca:
            embedding = ctx.pca.transform(embedding)
        
        # 4. FAISS Search
        distances, indices = ctx.vector_store.search(embedding, k=top_k)
        
        # 5. Enrich with Metadata and S3 URLs
        if len(indices) == 0:
            return {"query_id": file.filename, "results": []}

        # Filter out padding indices (-1)
        valid_indices = [int(idx) for idx in indices if idx != -1]
        valid_scores = [float(dist) for i, dist in enumerate(distances) if indices[i] != -1]
        
        metadata_list = ctx.get_metadata(valid_indices)
        
        results = []
        for i, meta in enumerate(metadata_list):
            filename = meta.get("filename")
            score = valid_scores[i]
            
            # Generate pre-signed URL (1 hour expiry)
            image_url = None
            if filename:
                image_url = s3_service.get_presigned_url(ctx.bucket, f"images/{filename}")
                # Note: 'images/' prefix matches our migration script structure
            
            results.append({
                "score": score,
                "metadata": meta,
                "image_url": image_url
            })
            
        return {
            "query_id": file.filename,
            "total_matches": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.exception(f"Search request failed: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal server error", "detail": str(e)})

@app.get("/v1/image/{filename}")
async def get_image_router(filename: str):
    """
    Router for images. Redirects to a secure MinIO URL.
    This allows the frontend to request an image without direct S3 access.
    """
    url = s3_service.get_presigned_url(ctx.bucket, f"images/{filename}")
    if not url:
        raise HTTPException(status_code=404, detail="Image not found or access denied")
    
    return RedirectResponse(url=url, status_code=307)

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "gpu_available": torch.cuda.is_available(),
        "index_loaded": ctx.vector_store.index.ntotal if ctx.vector_store and ctx.vector_store.index else 0,
        "model_loaded": ctx.inference is not None
    }

if __name__ == "__main__":
    uvicorn.run("logo_similarity.api.app:app", host="0.0.0.0", port=8000, reload=True)
