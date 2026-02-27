
import onnxruntime as ort
import numpy as np
import joblib
import json
import faiss
from pathlib import Path
import cv2

# Minimal inference class for limited memory
class LightweightLogoSearch:
    def __init__(self, model_dir: str):
        self.dir = Path(model_dir)
        
        # 1. Load ONNX Model (Quantized INT8)
        # Use CPU provider for VPS
        self.session = ort.InferenceSession(
            str(self.dir / "model_quant.onnx"), 
            providers=['CPUExecutionProvider']
        )
        
        # 2. Load PCA
        self.pca = joblib.load(self.dir / "pca_model.joblib")
        
        # 3. Load FAISS Index
        self.index = faiss.read_index(str(self.dir / "vps_index.bin"))
        
        # 4. Load ID Map
        with open(self.dir / "vps_id_map.json", "r") as f:
            self.id_map = json.load(f)

    def preprocess(self, image_path: str, size: int = 224):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        
        # Manual normalization
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1) # HWC -> CHW
        return img[np.newaxis, ...] # Add batch dim

    def search(self, image_path: str, k: int = 5):
        # 1. Get Embedding via ONNX
        input_data = self.preprocess(image_path)
        outputs = self.session.run(None, {'input': input_data})
        embedding = outputs[0] # [1, 1280]
        
        # 2. Apply PCA
        reduced = self.pca.transform(embedding) # [1, 256]
        
        # 3. FAISS Search
        faiss.normalize_L2(reduced) # Usually IP index needs normalization
        distances, indices = self.index.search(reduced.astype('float32'), k)
        
        results = []
        for d, idx in zip(distances[0], indices[0]):
            results.append({
                "id": self.id_map[idx],
                "score": float(d)
            })
        return results

if __name__ == "__main__":
    # Example usage
    # searcher = LightweightLogoSearch("models/onnx/semantic_v1_semantic_epoch_30")
    # print(searcher.search("test.jpg"))
    print("🚀 Lightweight serve module ready.")
