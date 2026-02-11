import json
import cv2
import torch
import numpy as np
from pathlib import Path
from logo_similarity.config import paths, settings
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.embeddings.pca_reducer import PCAReducer
from logo_similarity.retrieval.vector_store import VectorStore
from logo_similarity.utils.logging import logger
from PIL import Image
from torchvision import transforms
from logo_similarity.preprocessing.pipeline import PreprocessingPipeline

def test_similarity():
    """
    Verified test search using trained toy model and built index.
    Checks if similar pairs from validation set are retrieved.
    """
    logger.info("Initializing similarity test...")
    
    # 1. Load Index and ID Map
    dim = settings.REDUCED_DIM
    store = VectorStore(dimension=dim, index_type="hnsw")
    index_path = paths.EMBEDDING_INDEX_DIR / "faiss_index.bin"
    if not index_path.exists():
        logger.error(f"Index not found at {index_path}")
        return
        
    store.load(index_path)
    
    pca_path = paths.MODELS_DIR / "pca_model.joblib"
    reducer = PCAReducer.load(pca_path)
    
    id_map_path = paths.EMBEDDING_INDEX_DIR / "id_map.json"
    with open(id_map_path, "r") as f:
        id_map = json.load(f)
        
    # 2. Load Model
    model = EfficientNetEmbedder().cuda().eval()
    checkpoint_path = paths.CHECKPOINTS_DIR / "latest.pth"
    if checkpoint_path.exists():
        logger.info(f"Loading trained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        
        # Strip MoCo prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q.'):
                new_state_dict[k.replace('encoder_q.', '')] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        logger.warning("No trained model found, using base pre-trained weights.")
    
    # 3. Load Test Pairs
    pairs_path = paths.TOY_VALIDATION_DIR / "similar_pairs.json"
    with open(pairs_path, "r") as f:
        pairs = json.load(f)
        
    # Build a temporary RAW (1280-dim) index for the toy set
    # to see if the model is learning ANYTHING before PCA compression
    print("Building temporary 1280-dim index for direct test...")
    raw_store = VectorStore(dimension=1280, index_type="hnsw")
    
    # We need to compute raw embeddings for all toy images
    # Let's just use the already built chunks if they are available
    chunk_dir = paths.EMBEDDING_INDEX_DIR / "chunks"
    chunk_files = sorted(chunk_dir.glob("*.npz"))
    
    toy_id_list = []
    start_idx = 0
    for cf in chunk_files:
        with np.load(cf) as data:
            emb = data['embeddings'] # These are now normalized 1280-dim
            ids = data['ids']
            # Add to raw store
            indices = list(range(start_idx, start_idx + len(ids)))
            raw_store.add(emb, indices)
            toy_id_list.extend(ids.tolist())
            start_idx += len(ids)
            
    transform = transforms.Compose([
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n{'='*50}")
    print(f"SIMILARITY SEARCH (RAW 1280-DIM, NO PCA) - Top-100")
    print(f"{'='*50}")
    
    top10_count = 0
    top100_count = 0
    test_n = 10
    
    pipeline = PreprocessingPipeline(config={'skip_text_removal': True})

    for i in range(test_n):
        p = pairs[i]
        potential_path = str(paths.RAW_DATASET_DIR / "images" / p['image1'])
        img_np = pipeline.load_image(potential_path)
        if img_np is None: continue
            
        img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).cuda()
        
        with torch.no_grad():
            raw_emb = model(img_tensor).cpu().numpy().reshape(1, -1)
        
        # Search Top 100 in RAW index
        distances, indices = raw_store.search(raw_emb, k=100)
        
        matched_filenames = [toy_id_list[int(idx)] for idx in indices if idx < len(toy_id_list)]
        
        found_top10 = p['image2'] in matched_filenames[:10]
        found_top100 = p['image2'] in matched_filenames
        
        print(f"Test Pair {i+1}: {p['image1']} -> {p['image2']}")
        if found_top10:
            top10_count += 1
            print(f"  RESULT: SUCCESS (Top 10!)")
        elif found_top100:
            top100_count += 1
            rank = matched_filenames.index(p['image2']) + 1
            print(f"  RESULT: SUCCESS (Rank {rank})")
        else:
            print(f"  RESULT: FAILURE (Not in top 100)")
        
        print(f"  Top 5 Dists: {distances[:5]}")
        print(f"  Top 3 Matches: {matched_filenames[:3]}...")

    print(f"\nRAW EMBEDDING SUMMARY:")
    print(f"- Found in Top 10:  {top10_count}/{test_n}")
    print(f"- Found in Top 100: {top10_count + top100_count}/{test_n}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_similarity()
