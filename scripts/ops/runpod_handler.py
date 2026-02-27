
import runpod
import cv2
import numpy as np
import base64
from logo_similarity.api.app import APIContext
from logo_similarity.utils.s3 import s3_service

# Initialize the context globally for warm starts
ctx = APIContext()
ctx.initialize()

def handler(event):
    """
    RunPod Serverless Handler
    Input: { "input": { "image_base64": "...", "top_k": 50 } }
    """
    job_input = event.get("input", {})
    img_b64 = job_input.get("image_base64")
    top_k = job_input.get("top_k", 50)

    if not img_b64:
        return {"error": "No image provided"}

    try:
        # 1. Decode Image
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Run Inference
        embedding = ctx.inference.run(img_bgr)
        
        # 3. FAISS Search
        distances, indices = ctx.vector_store.search(embedding, k=top_k)
        
        # 4. Format Results
        results = []
        for d, idx in zip(distances, indices):
            if idx == -1: continue
            
            filename = ctx.id_map[idx] if hasattr(ctx, "id_map") else str(idx)
            image_url = s3_service.get_presigned_url(ctx.bucket, f"images/{filename}")
            
            results.append({
                "score": float(d),
                "filename": filename,
                "image_url": image_url
            })
            
        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
