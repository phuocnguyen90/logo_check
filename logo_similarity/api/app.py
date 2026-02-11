from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from pathlib import Path
from typing import List

from ..config import settings, paths
from ..utils.logging import logger
from ..preprocessing.pipeline import PreprocessingPipeline
from ..api.onnx_inference import ONNXInference
from ..retrieval.vector_store import VectorStore
from ..embeddings.pca_reducer import PCAReducer
from ..reranking.reranker import ReRanker
from ..reranking.composite_scorer import CompositeScoringPipeline

app = FastAPI(title="Logo Similarity API", version="0.1.0")

# Lazy-loaded globals
pipeline = None
inference = None
vector_store = None
pca = None
reranker = None
scorer = None

@app.on_event("startup")
async def startup_event():
    global pipeline, inference, vector_store, pca, reranker, scorer
    try:
        pipeline = PreprocessingPipeline()
        # Paths from settings/config
        model_path = paths.ONNX_DIR / "model_fp16.onnx"
        if model_path.exists():
            inference = ONNXInference(str(model_path))
        
        vector_store = VectorStore(settings.REDUCED_DIM)
        index_path = paths.EMBEDDING_INDEX_DIR / "faiss_index.bin"
        if index_path.exists():
            vector_store.load(str(index_path))
            
        pca_path = paths.MODELS_DIR / "pca_model.joblib"
        if pca_path.exists():
            pca = PCAReducer.load(str(pca_path))
            
        # These need an encoder, but in API we use ONNX
        # This part requires some adaptation for ONNX-based re-ranking
        # For MVP, we'll keep it simple
        scorer = CompositeScoringPipeline()
        
        logger.info("API services initialized.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    """Search for similar logos by uploading an image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 1. Preprocess
        # For API, we don't have a path, so we use the in-memory img
        # (Need to adapt pipeline slightly for raw image input)
        # Assuming pipeline.process(img) works for now
        
        # 2. Embedding + PCA
        # 3. Vector Search
        # 4. Re-rank
        # 5. Composite Score
        
        return {"results": [], "query_id": file.filename}
        
    except Exception as e:
        logger.error(f"Search request failed: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal server error"})

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
