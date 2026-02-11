import json
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
    
    # 4. Run Search Tests
    transform = transforms.Compose([
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*50)
    print("SIMILARITY SEARCH TEST RESULTS (Top-10)")
    print("="*50)
    
    # Test first 10 pairs
    success_count = 0
    test_n = 10
    
    for i in range(test_n):
        p = pairs[i]
        img1_path = paths.RAW_DATASET_DIR / "images" / p['image1']
        img2_path = paths.RAW_DATASET_DIR / "images" / p['image2']
        
        if not img1_path.exists():
            continue
            
        # Get embedding for img1
        img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            emb1 = model(img1).cpu().numpy()
        
        reduced_emb1 = reducer.transform(emb1)
        
        # Search
        distances, indices = store.search(reduced_emb1, k=10)
        
        matched_filenames = [id_map[idx] for idx in indices if idx < len(id_map)]
        
        print(f"Test Pair {i+1} (Vienna: {p['vienna_code']})")
        print(f"  Query:  {p['image1']}")
        print(f"  Target: {p['image2']}")
        
        if p['image2'] in matched_filenames:
            rank = matched_filenames.index(p['image2']) + 1
            print(f"  RESULT: SUCCESS (Rank {rank})")
            success_count += 1
        else:
            print(f"  RESULT: FAILURE (Not in top 10)")
        
        print(f"  Matches: {matched_filenames[:3]}...")
        print("-" * 50)

    print(f"\nFinal Score: {success_count}/{test_n} targets found in Top 10.")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_similarity()
