"""
Logo Similarity Retrieval & Evaluation Demo

Supports per-model FAISS indexes for evaluating different training checkpoints.
Each checkpoint gets its own index at indexes/embeddings/{checkpoint_stem}/.
"""
import streamlit as st
import numpy as np
import json
import torch
import cv2
from pathlib import Path
from PIL import Image
import os
import sys
import subprocess
import time
import sqlite3
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.retrieval.vector_store import VectorStore
from logo_similarity.reranking.reranker import ReRanker
from logo_similarity.reranking.composite_scorer import CompositeScoringPipeline
from logo_similarity.preprocessing.pipeline import PreprocessingPipeline

# --- Torch Safe Globals ---
try:
    import torch.serialization
    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np._core.multiarray.scalar if hasattr(np, '_core') else np.core.multiarray.scalar,
        np.float64, np.int64
    ])
except Exception:
    pass

# --- Page Config ---
st.set_page_config(
    page_title="Logo Similarity Evaluation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Logo Similarity Retrieval Pipeline")


# =================================================================
# Helper Functions
# =================================================================

def get_index_dir(checkpoint_name: str) -> Path:
    """Get model-specific index directory."""
    stem = Path(checkpoint_name).stem
    return paths.INDEXES_DIR / "embeddings" / stem


def has_index(checkpoint_name: str) -> bool:
    """Check if a FAISS index exists for the given model."""
    d = get_index_dir(checkpoint_name)
    return (d / "faiss_index.bin").exists() and (d / "id_map.json").exists()


def resolve_image_path(filename: str) -> Optional[Path]:
    """Robustly resolve image path (case-insensitive, subfolder check)."""
    if not filename:
        return None
    
    path = Path(filename)
    # If it's already an absolute path that exists, return it
    if path.is_absolute() and path.exists():
        return path

    # Search candidates
    search_dirs = [paths.RAW_DATASET_DIR / "images", paths.RAW_DATASET_DIR]
    
    # If the filename itself looks like a relative path with 'images/'
    if "images" in path.parts:
        pure_name = path.name
    else:
        pure_name = str(path)

    for d in search_dirs:
        p = d / pure_name
        if p.exists():
            return p
        
        # Try swapping case of extension
        if p.suffix.lower() == '.jpg':
            alt_suffix = '.JPG' if p.suffix == '.jpg' else '.jpg'
            p_alt = p.with_suffix(alt_suffix)
            if p_alt.exists():
                return p_alt
                
    return None


def normalize_name(name) -> str:
    """
    Canonical name adapter for filename comparison across training / serving / evaluation.

    The same logo is referred to differently in:
      - id_map   ->  'images/uuid.JPG'  or  'uuid.jpg'
      - validation pairs  ->  'uuid.jpg'
      - MinIO keys  ->  'images/uuid.jpg'
      - local filesystem  ->  absolute path with .JPG extension

    This function extracts just the basename and lowercases it so all comparisons
    are consistent. Always safe to call with None.
    """
    if not name:
        return ""
    return Path(name).name.lower()


# =================================================================
# Cached Resource Loaders
# =================================================================

@st.cache_resource(max_entries=1)
def load_embedder(checkpoint_name: str):
    """
    Load EfficientNet embedder with MoCo checkpoint weights.
    max_entries=1 ensures that when the user switches checkpoints the previous
    model is evicted from the cache (and its GPU/CPU memory released) before
    the new one is loaded.  Without this, every checkpoint ever selected would
    remain resident for the lifetime of the Streamlit process.
    """
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = EfficientNetEmbedder().to(device)
    ckpt_path = paths.CHECKPOINTS_DIR / checkpoint_name

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q.'):
                new_state_dict[k.replace('encoder_q.', '')] = v
            elif k.startswith('module.encoder_q.'):
                new_state_dict[k.replace('module.encoder_q.', '')] = v

        if new_state_dict:
            embedder.load_state_dict(new_state_dict, strict=False)
        else:
            st.warning(f"No 'encoder_q' keys in {checkpoint_name}. Using ImageNet defaults.")

        embedder.eval()
    else:
        st.error(f"Checkpoint not found: {ckpt_path}")

    # Release any temporary tensors created during weight loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embedder, device


@st.cache_resource(max_entries=1)
def load_index(checkpoint_name: str):
    """Load model-specific FAISS index."""
    d = get_index_dir(checkpoint_name)
    idx_path = d / "faiss_index.bin"
    map_path = d / "id_map.json"

    if not idx_path.exists() or not map_path.exists():
        return None, []

    # VectorStore.load auto-detects dim from the .bin file
    store = VectorStore(dimension=1280, index_type="hnsw")
    store.load(idx_path)

    with open(map_path, "r") as f:
        id_list = json.load(f)

    return store, id_list


class DatabaseMetadataLookup:
    """Lazy lookup for trademark metadata using SQLite."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.cache = {} # Optional small cache for frequently accessed items
        
    def get(self, filename: str, default=None) -> Dict[str, Any]:
        if not filename: return default
        
        # Canonical name check
        name = normalize_name(filename)
        if name in self.cache: return self.cache[name]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Try original filename and lower variant
            cursor.execute("""
                SELECT t.filename, t.tm_text, t.year,
                       GROUP_CONCAT(vc.code, ',') as codes
                FROM trademarks t
                LEFT JOIN trademark_vienna tv ON tv.trademark_id = t.id
                LEFT JOIN vienna_codes vc ON vc.id = tv.vienna_code_id
                WHERE t.filename = ? OR LOWER(t.filename) = ?
                GROUP BY t.id
            """, (filename, name))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                res = {
                    "file": row[0],
                    "text": row[1],
                    "year": row[2],
                    "vienna_codes": row[3].split(",") if row[3] else []
                }
                # Minimal cache to avoid redundant queries in tight loops
                if len(self.cache) < 1000:
                    self.cache[name] = res
                return res
        except Exception as e:
            logger.error(f"DB Metadata lookup error: {e}")
            
        return default

@st.cache_resource
def load_metadata():
    """Load dataset metadata proxy."""
    db_path = paths.DATA_DIR / "metadata_v2.db"
    if not db_path.exists():
        st.warning(f"Metadata DB not found at {db_path}. Using fallback.")
        return {} # Or implement legacy JSON fallback here if needed
        
    return DatabaseMetadataLookup(db_path)



@st.cache_resource
def load_validation_pairs():
    """Load validation pairs."""
    p = paths.TOY_VALIDATION_DIR / "similar_pairs.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return []


# =================================================================
# Sidebar — Model & Index
# =================================================================
st.sidebar.header("⚙️ Configuration")

avail_models = sorted([f.name for f in paths.CHECKPOINTS_DIR.glob("*.pth")])
if not avail_models:
    st.error("No model checkpoints found in " + str(paths.CHECKPOINTS_DIR))
    st.stop()

# Pick a sensible default
default_idx = 0
for i, name in enumerate(avail_models):
    if "semantic" in name and "latest" in name:
        default_idx = i
        break
    elif "best" in name:
        default_idx = i

selected_model = st.sidebar.selectbox("Model Checkpoint", avail_models, index=default_idx)

# --- Index Status ---
index_ready = has_index(selected_model)
idx_dir = get_index_dir(selected_model)

if index_ready:
    st.sidebar.success("✅ Index ready")
    meta_file = idx_dir / "build_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            bm = json.load(f)
        st.sidebar.caption(
            f"📁 {bm.get('total_images', '?')} vectors · "
            f"Built {bm.get('built_at', '?')[:16]}"
        )
else:
    st.sidebar.warning("⚠️ No index for this model")
    st.sidebar.caption("Build one before running retrieval.")

    use_toy = st.sidebar.checkbox("Toy dataset (faster)", value=True)
    build_bs = st.sidebar.select_slider("Build Batch Size (GPU)", options=[64, 128, 256, 512, 1024, 2048], value=128)
    build_workers = st.sidebar.slider("Build Workers (CPU)", 1, 32, 4)

    if st.sidebar.button("🔨 Build Index", type="primary"):
        # Use subprocess for indexing to enable multi-processing and resumability
        # This is much faster for the full 770k image dataset.
        from datetime import datetime
        
        log_box = st.empty()
        project_root = str(Path(__file__).resolve().parent.parent)
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "03b_fast_index.py"),
            "--checkpoint", str(paths.CHECKPOINTS_DIR / selected_model),
            "--batch-size", str(build_bs),
            "--workers", str(build_workers)
        ]
        if use_toy:
            cmd.append("--toy")

        with st.spinner(f"Building index (Multiprocessing ON)..."):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=project_root, env=env
            )
            
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip())
                # Show last 20 lines of logs
                log_box.code('\n'.join(lines[-20:]), language="text")
            proc.wait()

        if proc.returncode == 0:
            st.success("✅ Index built! Reloading…")
            st.cache_resource.clear()
            # Eagerly free GPU and numpy heap memory after evicting all cached
            # resources so the new index/model start with a clean slate.
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"❌ Build failed (code {proc.returncode})")
            st.caption("Check terminal/logs for details. Common issues: OOM or path errors.")

    st.stop()   # Block rest of app until index exists


# =================================================================
# Load Resources
# =================================================================
st.sidebar.markdown("---")
st.sidebar.header("🔍 Retrieval Settings")

# Query Preprocessing Settings
clean_query = st.sidebar.checkbox("Clean query logo (remove text)", value=True, help="Use full preprocessing pipeline to detect and mask text for internet logos.")

embedder, device = load_embedder(selected_model)
store, full_id_list = load_index(selected_model)
metadata_map = load_metadata()

if store is None:
    st.error("Failed to load FAISS index.")
    st.stop()

# ---------------------------------------------------------------------------
# Cached helpers for objects that are NOT in @st.cache_resource above but
# still hold significant memory (the PreprocessingPipeline keeps an LRU cache
# of decoded numpy image arrays).
#
# Keying on `clean_query` means the pipeline's internal image cache is
# invalidated whenever the toggle flips, preventing stale inpainted images
# from polluting a subsequent "clean=OFF" session (or vice-versa).
# max_entries=1 ensures old pipelines are GC'd rather than accumulated.
# ---------------------------------------------------------------------------
@st.cache_resource(max_entries=1)
def _make_preprocessing(skip_text_removal: bool) -> PreprocessingPipeline:
    return PreprocessingPipeline(config={'skip_text_removal': skip_text_removal})

@st.cache_resource(max_entries=1)
def _make_reranker(_embedder):
    # Underscore prefix on arg tells Streamlit not to hash the embedder object
    return ReRanker(_embedder)

@st.cache_resource(max_entries=1)
def _make_composite():
    return CompositeScoringPipeline()

preprocessing = _make_preprocessing(not clean_query)
reranker = _make_reranker(embedder)
composite_pipeline = _make_composite()

reducer = None
if settings.USE_PCA:
    from logo_similarity.embeddings.pca_reducer import PCAReducer
    pca_path = paths.MODELS_DIR / "pca_model.joblib"
    if pca_path.exists():
        reducer = PCAReducer.load(pca_path)
        st.sidebar.caption(f"PCA: {reducer.n_components}d loaded")


# =================================================================
# Sidebar — Evaluation Mode
# =================================================================
st.sidebar.markdown("---")
st.sidebar.header("🎯 Evaluation")
mode = st.sidebar.radio(
    "Mode",
    ["Toy Validation Pair", "Raw Image ID", "Upload Image", "📊 Batch Evaluate"]
)
top_k_global = st.sidebar.slider("Stage 1 (Global) Candidates", 100, 1000, 500)
top_k_rerank = st.sidebar.slider("Stage 2 (Spatial) Candidates", 10, 100, 50)


# =================================================================
# Helper: Embed a single image
# =================================================================
def embed_query(img_path: str):
    """Preprocess and embed a query image. Returns (preprocessed, q_search) or (None, None)."""
    preprocessed = preprocessing.process_on_the_fly(img_path)
    if preprocessed is None:
        return None, None

    q_emb = embedder.get_embedding(preprocessed.normalized, device=device)
    q_vec = q_emb.reshape(1, -1).astype("float32")

    # Auto-select raw vs PCA-reduced based on what the loaded index actually uses
    if store is not None:
        if store.dimension == q_vec.shape[1]:
            q_search = q_vec  # 1280d index (original GPU setup)
        elif reducer is not None and store.dimension == reducer.n_components:
            q_search = reducer.transform(q_vec).astype("float32")  # PCA-reduced
        else:
            pca_dim = reducer.n_components if reducer else 'None'
            st.error(f"Dimension mismatch — Index: {store.dimension}d | Raw: {q_vec.shape[1]}d | PCA: {pca_dim}d. Rebuild index.")
            return preprocessed, None
    else:
        q_search = q_vec

    return preprocessed, q_search

# =================================================================
# BATCH EVALUATE MODE
# =================================================================
if mode == "📊 Batch Evaluate":
    st.subheader(f"📊 Batch Evaluation — `{selected_model}`")

    pairs = load_validation_pairs()
    if not pairs:
        st.warning("No validation pairs found.")
        st.stop()

    st.info(f"**{len(pairs)}** validation pairs available.")

    if st.button("▶️ Run Batch Evaluation", type="primary"):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        progress = st.progress(0, text="Starting…")
        results = []

        for i, pair in enumerate(pairs):
            q_name, t_name = pair['image1'], pair['image2']

            # Resolve query path robustly
            q_path = resolve_image_path(q_name)

            if not q_path:
                results.append(dict(query=q_name, target=t_name, found=False, rank=-1, score=0, error="missing"))
                continue

            preprocessed, q_search = embed_query(str(q_path))
            if preprocessed is None:
                results.append(dict(query=q_name, target=t_name, found=False, rank=-1, score=0, error="preproc"))
                continue

            distances, indices = store.search(q_search, k=top_k_global)

            found, rank, score = False, -1, 0.0
            for j, (d, idx) in enumerate(zip(distances, indices)):
                if idx == -1:
                    continue
                if normalize_name(full_id_list[idx]) == normalize_name(t_name):
                    found, rank, score = True, j + 1, float(d)
                    break

            results.append(dict(query=q_name, target=t_name, found=found, rank=rank, score=score, error=None))
            progress.progress((i + 1) / len(pairs), text=f"Evaluated {i+1}/{len(pairs)}")

        progress.empty()

        # --- Metrics ---
        valid = [r for r in results if r['error'] is None]
        found_list = [r for r in valid if r['found']]

        st.markdown("---")
        st.subheader("📈 Results")

        k_values = [k for k in [1, 5, 10, 50, 100, 500] if k <= top_k_global]
        recalls = {f"R@{k}": sum(1 for r in valid if r['found'] and r['rank'] <= k) / max(len(valid), 1) for k in k_values}

        mrr = sum(1.0 / r['rank'] for r in found_list) / max(len(valid), 1)

        cols = st.columns(len(recalls) + 2)
        cols[0].metric("Pairs", f"{len(valid)}/{len(results)}")
        cols[1].metric("MRR", f"{mrr:.4f}")
        for i, (label, val) in enumerate(recalls.items()):
            cols[i + 2].metric(label, f"{val:.1%}")

        if found_list:
            ranks = [r['rank'] for r in found_list]
            st.caption(
                f"Found: {len(found_list)}/{len(valid)} · "
                f"Mean Rank: {np.mean(ranks):.1f} · "
                f"Median: {np.median(ranks):.0f} · "
                f"Best/Worst: {min(ranks)}/{max(ranks)}"
            )

        with st.expander("📋 Per-Pair Details", expanded=False):
            import pandas as pd
            df = pd.DataFrame(results)
            df['rank'] = df['rank'].apply(lambda x: x if x > 0 else 'N/A')
            st.dataframe(df, width='stretch', hide_index=True)

    st.stop()


# =================================================================
# SINGLE QUERY MODE
# =================================================================
query_img_path = None
query_img_name = "Upload"
target_img_name = None

if mode == "Toy Validation Pair":
    pairs = load_validation_pairs()
    if pairs:
        pair_opts = [f"Pair {i}: {p['image1']} -> {p['image2']}" for i, p in enumerate(pairs)]
        sel = st.sidebar.selectbox("Select Pair", range(len(pairs)), format_func=lambda x: pair_opts[x])
        pair = pairs[sel]
        query_img_name = pair['image1']
        target_img_name = pair['image2']

        qp = resolve_image_path(query_img_name)
        if qp:
            query_img_path = str(qp)
        else:
            st.error(f"Image not found: {query_img_name}")

elif mode == "Raw Image ID":
    query_img_name = st.sidebar.text_input("Enter Image Filename (e.g., uuid.jpg)")
    if query_img_name:
        qp = resolve_image_path(query_img_name)
        if qp:
            query_img_path = str(qp)
        else:
            st.error(f"File '{query_img_name}' not found on disk.")

elif mode == "Upload Image":
    uploaded = st.sidebar.file_uploader("Upload Query Image", type=["jpg", "png", "jpeg"])
    if uploaded:
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
            preprocessed, q_search = embed_query(query_img_path)
            if preprocessed is None:
                st.error("Preprocessing failed.")
                st.stop()

            # Stage 1 — Global Search
            distances, indices = store.search(q_search, k=top_k_global)

            candidates = []
            found_target = False
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if idx == -1:
                    continue
                img_name = full_id_list[idx]
                cand_path = resolve_image_path(img_name)

                candidates.append({
                    "image": img_name,
                    "path": str(cand_path) if cand_path else None,
                    "global_score": float(dist),
                    "metadata": metadata_map.get(normalize_name(img_name), {})
                })
                if normalize_name(img_name) == normalize_name(target_img_name):
                    found_target = True

            st.write(f"Stage 1 found {len(candidates)} candidates.")
            if target_img_name:
                if found_target:
                    rank = next((j for j, c in enumerate(candidates) if normalize_name(c['image']) == normalize_name(target_img_name)), -1)
                    st.success(f"Target found at Rank {rank + 1}")
                else:
                    st.warning(f"Target NOT found in Top {top_k_global}")

            # Stage 2 — Re-ranking
            refined = reranker.rerank_candidates(
                preprocessed.normalized, candidates, top_k=top_k_rerank
            )

            # Stage 3 — Composite Scoring
            final = composite_pipeline.score_results(
                metadata_map.get(normalize_name(query_img_name), {}),
                preprocessed.original,
                refined
            )

            # --- Results Display ---
            st.subheader("Top Results")
            rcols = st.columns(3)
            for i, cand in enumerate(final[:9]):
                col = rcols[i % 3]
                with col:
                    if os.path.exists(cand['path']):
                        st.image(cand['path'], width='stretch')
                    else:
                        st.warning("Image missing")

                    st.markdown(f"**#{i+1}: {cand['image']}**")
                    if normalize_name(cand['image']) == normalize_name(target_img_name):
                        st.metric("Final Score", f"{cand['final_score']:.3f}", delta="TARGET")
                    else:
                        st.write(f"Score: **{cand['final_score']:.3f}**")

                    with st.expander("Details"):
                        st.write(f"Global: {cand.get('global_score',0):.3f}")
                        st.write(f"Spatial: {cand.get('spatial_score',0):.3f}")
                        st.write(f"Text: {cand.get('text_similarity',0):.3f}")
                        st.write(f"Color: {cand.get('color_score',0):.3f}")
