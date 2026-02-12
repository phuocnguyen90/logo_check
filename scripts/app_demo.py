import streamlit as st
import numpy as np
import json
import torch
import cv2
from pathlib import Path
from PIL import Image
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.retrieval.vector_store import VectorStore
from logo_similarity.reranking.reranker import ReRanker
from logo_similarity.reranking.composite_scorer import CompositeScoringPipeline
from logo_similarity.preprocessing.pipeline import PreprocessingPipeline

# --- Helper: Torch Safe Globals ---
try:
    import torch.serialization
    import numpy as np
    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np._core.multiarray.scalar if hasattr(np, '_core') else np.core.multiarray.scalar,
        np.float64, np.int64
    ])
except Exception as e:
    pass

# --- Page Config ---
st.set_page_config(
    page_title="Logo Similarity Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Logo Similarity Retrieval Pipeline")

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_pipeline():
    status = st.empty()
    status.write("Loading pipeline components...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Embedder
    # 1. Embedder
    try:
        # Priority: Semantic Test Model -> Best Model -> Latest Model -> Pretrained
        ckpt_path = None
        possible_ckpts = [
            paths.CHECKPOINTS_DIR / "semantic_v1_epoch1.pth",  # Test Model
            paths.CHECKPOINTS_DIR / "best_model.pth",
            paths.CHECKPOINTS_DIR / "latest.pth"
        ]
        
        for p in possible_ckpts:
            if p.exists():
                ckpt_path = p
                break
        
        embedder = EfficientNetEmbedder().to(device)
        
        if ckpt_path and ckpt_path.exists():
            status.text(f"Loading checkpoint: {ckpt_path.name}...")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                # Clean MoCo prefixes
                if k.startswith('encoder_q.'):
                    new_state_dict[k.replace('encoder_q.', '')] = v
                elif k.startswith('module.encoder_q.'): # Multi-GPU case
                    new_state_dict[k.replace('module.encoder_q.', '')] = v
                else:
                    new_state_dict[k] = v
                    
            embedder.load_state_dict(new_state_dict, strict=False)
            embedder.eval()
            st.success(f"✅ Loaded checkpoint: **{ckpt_path.name}**")
        else:
            st.warning("⚠️ Checkpoint not found. Using pretrained ImageNet weights.")
    except Exception as e:
        st.error(f"Failed to load embedder: {e}")
        return None

    # 2. Vector Store
    try:
        index_path = paths.EMBEDDING_INDEX_DIR / "faiss_index.bin"
        id_map_path = paths.EMBEDDING_INDEX_DIR / "id_map.json"
        
        index_dim = 1280 if not settings.USE_PCA else settings.REDUCED_DIM
        store = VectorStore(dimension=index_dim, index_type="hnsw")
        if index_path.exists():
            store.load(index_path)
        
        if id_map_path.exists():
            with open(id_map_path, "r") as f:
                full_id_list = json.load(f)
        else:
            full_id_list = []
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None

    # 3. Metadata
    try:
        metadata_path = paths.DATASET_METADATA
        if not metadata_path.exists():
            metadata_path = paths.TOY_DATASET_METADATA
            
        with open(metadata_path, "r") as f:
            metadata_list = json.load(f)
            
        metadata_map = {}
        for item in metadata_list:
            key = item.get('image') or item.get('file')
            if key:
                metadata_map[key] = item
    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        return None

    # 4. Pipeline Modules
    reranker = ReRanker(embedder)
    composite_pipeline = CompositeScoringPipeline()
    preprocessing = PreprocessingPipeline(config={'skip_text_removal': True})
    
    # 5. PCA Reduction
    reducer = None
    if settings.USE_PCA:
        from logo_similarity.embeddings.pca_reducer import PCAReducer
        pca_path = paths.MODELS_DIR / "pca_model.joblib"
        if pca_path.exists():
            reducer = PCAReducer.load(pca_path)

    status.empty()
    return embedder, store, full_id_list, metadata_map, reranker, composite_pipeline, preprocessing, reducer, device

# Load
pipeline = load_pipeline()

if not pipeline:
    st.stop()

embedder, store, full_id_list, metadata_map, reranker, composite_pipeline, preprocessing, reducer, device = pipeline

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Input Mode", ["Toy Validation Pair", "Raw Image ID", "Upload Image"])

top_k_global = st.sidebar.slider("Stage 1 (Global) Candidates", 100, 1000, 500)
top_k_rerank = st.sidebar.slider("Stage 2 (Spatial) Candidates", 10, 100, 50)

# --- Input Handling ---
query_img_path = None
query_img_name = "Upload"
target_img_name = None

if mode == "Toy Validation Pair":
    toy_val_path = paths.TOY_VALIDATION_DIR / "similar_pairs.json"
    if toy_val_path.exists():
        with open(toy_val_path, "r") as f:
            pairs = json.load(f)
        
        pair_options = [f"Pair {i}: {p['image1']} -> {p['image2']}" for i, p in enumerate(pairs)]
        selected_pair_idx = st.sidebar.selectbox("Select Pair", range(len(pairs)), format_func=lambda x: pair_options[x])
        
        pair = pairs[selected_pair_idx]
        query_img_name = pair['image1']
        target_img_name = pair['image2']
        
        # Check path
        query_path = paths.RAW_DATASET_DIR / "images" / query_img_name
        if not query_path.exists():
            query_path = paths.RAW_DATASET_DIR / query_img_name
        
        if query_path.exists():
            query_img_path = str(query_path)
        else:
            st.error(f"Image not found locally: {query_path}")

elif mode == "Raw Image ID":
    query_img_name = st.sidebar.text_input("Enter Image Filename (e.g., uuid.jpg)")
    if query_img_name:
        query_path = paths.RAW_DATASET_DIR / "images" / query_img_name
        if not query_path.exists():
            query_path = paths.RAW_DATASET_DIR / query_img_name
        
        if query_path.exists():
            query_img_path = str(query_path)
        else:
            st.error("File not found on disk.")

elif mode == "Upload Image":
    uploaded = st.sidebar.file_uploader("Upload Query Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        # Save temp
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        query_img_path = str(temp_dir / uploaded.name)
        with open(query_img_path, "wb") as f:
            f.write(uploaded.getbuffer())
        query_img_name = uploaded.name

# --- Execution ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Query Image")
    if query_img_path:
        st.image(query_img_path, width='stretch')
        if target_img_name:
            st.caption(f"Target: {target_img_name}")
    else:
        st.info("Select or upload an image to start.")

with col2:
    if query_img_path and st.button("Run Retrieval", type="primary"):
        with st.spinner("Processing Pipeline..."):
            # A. Preprocessing
            preprocessed = preprocessing.process_on_the_fly(query_img_path)
            if not preprocessed:
                st.error("Preprocessing failed.")
                st.stop()
                
            # B. Stage 1 Global Search
            q_emb = embedder.get_embedding(preprocessed.normalized, device=device)
            
            if reducer:
                q_search = reducer.transform(q_emb.reshape(1, -1))
            else:
                q_search = q_emb.reshape(1, -1)
                
            distances, indices = store.search(q_search, k=top_k_global)
            
            candidates = []
            found_target = False
            
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if idx == -1: continue
                img_name = full_id_list[idx]
                
                # Resolve path
                cand_path = paths.RAW_DATASET_DIR / "images" / img_name
                if not cand_path.exists():
                    cand_path = paths.RAW_DATASET_DIR / img_name
                
                candidates.append({
                    "image": img_name,
                    "path": str(cand_path),
                    "global_score": float(dist),
                    "metadata": metadata_map.get(img_name, {})
                })
                
                if img_name == target_img_name:
                    found_target = True

            st.write(f"Stage 1 found {len(candidates)} candidates.")
            if target_img_name:
                 if found_target:
                     rank = next((i for i, c in enumerate(candidates) if c['image'] == target_img_name), -1)
                     st.success(f"Target found in Top {top_k_global} at Rank {rank+1}")
                 else:
                     st.warning(f"Target NOT found in Top {top_k_global}")

            # C. Stage 2 Re-ranking
            refined_candidates = reranker.rerank_candidates(
                preprocessed.normalized,
                candidates,
                top_k=top_k_rerank
            )
            
            # D. Stage 5 Composition
            final_candidates = composite_pipeline.score_results(
                metadata_map.get(query_img_name, {}),
                preprocessed.original,
                refined_candidates
            )
            
            # --- Results Display ---
            st.subheader("Top Results")
            
            results_cols = st.columns(3)
            for i, cand in enumerate(final_candidates[:9]):
                col = results_cols[i % 3]
                with col:
                    # Load image
                    if os.path.exists(cand['path']):
                        st.image(cand['path'], width='stretch')
                    else:
                        st.warning("Image missing")
                        
                    color = "green" if cand['image'] == target_img_name else "black"
                    st.markdown(f"**#{i+1}: {cand['image']}**")
                    if cand['image'] == target_img_name:
                        st.metric("Final Score", f"{cand['final_score']:.3f}", delta="TARGET")
                    else:
                        st.write(f"Score: **{cand['final_score']:.3f}**")
                        
                    with st.expander("Details"):
                        st.write(f"Global: {cand.get('global_score',0):.3f}")
                        st.write(f"Spatial: {cand.get('spatial_score',0):.3f}")
                        st.write(f"Text: {cand.get('text_similarity',0):.3f}")
                        st.write(f"Color: {cand.get('color_score',0):.3f}")
