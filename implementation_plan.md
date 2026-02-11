# Logo Similarity Detection — Revised Implementation Plan

## Architecture: 2-Stage Pipeline + Preprocessing

```
Input Logo → Preprocess (OCR + mask text) → Stage 1: Retrieve candidates (ANN) → Stage 2: Re-rank → Results
```

Key change from v1: **Deep embeddings from day one. No hand-crafted features.**

---
Dataset downloaded from kaggle: "home\phuoc\.cache\kagglehub\datasets\konradb\ziilogos\versions\1\L3D dataset"
use torch12 mamba environment (torch12 and cuda12 preinstalled)
---

# Logo Similarity Detection — Revised Implementation Plan

## Architecture: 2-Stage Pipeline + Preprocessing

```
Input Logo → Preprocess (OCR + mask text) → Stage 1: Retrieve candidates (ANN) → Stage 2: Re-rank → Results
```

Key change from v1: **Deep embeddings from day one. No hand-crafted features.**

---

## Stage 0: Preprocessing

**Goal**: Normalize input and remove text to isolate figurative elements.

| Step | Method | Why |
|------|--------|-----|
| Text detection | Tesseract OCR or EAST detector | Text degrades visual retrieval significantly |
| Text masking | Inpaint detected text regions | Compare figurative elements only |
| Normalization | Resize to 224×224, white background, center crop | Consistent input for embeddings |
| Binarization | Optional B&W version for shape-focused comparison | Examiners compare in B&W first |

**Tasks**:
1. Implement text detection + masking pipeline (Sharp + Tesseract)
2. Generate two versions per mark: original (color) and masked/binarized (structure)
3. Store both versions for downstream stages

---

## Stage 1: Global Embedding Retrieval (Coarse)

**Goal**: Retrieve top-1000 similar marks from full database in <100ms.

### Phase 1A — Baseline (no training, week 1-2)

Use a **pre-trained EfficientNet-B0** (ImageNet weights, ONNX export) as feature extractor.

| Item | Detail |
|------|--------|
| Model | EfficientNet-B0 (ONNX, ~20MB) |
| Embedding | Global Average Pooling of final conv layer → 1280-dim |
| Dimensionality reduction | PCA to 512-dim (fit on trademark dataset) |
| Index | pgvector with IVFFlat or HNSW index |
| Query | Cosine similarity, retrieve top-1000 |
| Inference | <50ms per image (ONNX Runtime, Node.js) |

This is the **day-one baseline**. No training needed. Pre-trained ImageNet features
capture structural/shape information surprisingly well for logo retrieval — they just
aren't optimal. But they're a working system you can demo and evaluate immediately.

### Phase 1B — Fine-tuned (weeks 4-6)

Fine-tune with **contrastive learning** on trademark data.

| Item | Detail |
|------|--------|
| Training data | EUIPO TMView marks + Vienna codes as weak supervision |
| Objective | InfoNCE loss (SimCLR-style): each anchor compared against all batch items simultaneously |
| Why not Triplet Loss | Triplet loss requires hard negative mining and collapses with easy negatives. InfoNCE uses all batch items as implicit negatives — scales better, trains more stably. |
| Batch size | Large (512+) — InfoNCE improves with more in-batch negatives |
| Augmentation | Rotation, scale, color jitter, partial occlusion |
| Validation | Hold-out set of known-similar pairs from opposition decisions |
| Expected gain | 15-30% improvement in Recall@100 over pretrained baseline |

**Why Vienna codes work here but not as a hard filter**: Using Vienna codes as
*training signal* (marks with same code are "similar-ish") is fine — noisy labels
still work for contrastive learning. Using them as a *hard retrieval filter* is
what kills recall.

---

## Stage 2: Re-ranking with Local Features (Precision)

**Goal**: Re-rank top-1000 candidates using spatial/structural verification.

### Approach: Deep Local Feature Matching

Extract feature maps (not just global pooling) from the same EfficientNet backbone:

```
Query image  → EfficientNet → Final conv layer → 7×7×1280 spatial features (49 patches)
                                                          ↓
Candidate    → EfficientNet → Final conv layer → 7×7×1280 spatial features (49 patches)
                                                          ↓
                                            Pairwise cosine similarity
                                            (49×49 matrix)
                                                          ↓
                                            For each query patch, take MAX
                                            similarity across all candidate patches
                                                          ↓
                                            Average of max scores = alignment score
```

**Why max-pooled pairwise, not grid 1-to-1**: A strict positional comparison
breaks if the logo is shifted even slightly. The 49×49 pairwise matrix lets
each query patch find its best match *anywhere* in the candidate, making
this translation/scale invariant without any alignment step.

| Method | Complexity | Translation invariant | Accuracy |
|--------|-----------|----------------------|----------|
| ~~Grid 1-to-1 cosine~~ | ~~Low~~ | ~~No — breaks with shifts~~ | ~~Poor~~ |
| **Max-pooled pairwise (recommended)** | **Low — 49×49 matrix multiply** | **Yes** | **Good** |
| Cross-attention transformer layer | Medium | Yes | Better |
| RANSAC on deep keypoints | High | Yes | Best but slow |

**Start with max-pooled pairwise** — it's a single matrix multiplication (49×49)
per candidate, trivially parallelizable, and runs on 1000 candidates in <2 seconds.

---

## Composite Scoring

```
final_score = 0.50 × global_embedding_similarity    (Stage 1)
            + 0.35 × spatial_alignment_score         (Stage 2)
            + 0.10 × syllable/text_similarity         (if composite mark with text)
            + 0.05 × color_palette_similarity          (secondary)
```

For **composite marks** (logo + text), run phonetic similarity (your existing
ALINE module) on the extracted text component alongside the visual pipeline.
This catches cases like similar logos AND similar brand names.

---

## Data Requirements

| Dataset | Source | Use | Priority |
|---------|--------|-----|----------|
| Trademark images | EUIPO TMView bulk export | Embedding index + training | P0 |
| Vienna codes | Same TMView export | Weak supervision for fine-tuning | P1 |
| Opposition decisions | EUIPO eSearch Case Law | Validation/calibration pairs | P1 |
| WIPO Global Brand Database | WIPO API | Additional coverage | P2 |

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Inference | ONNX Runtime (Node.js) | No Python in production |
| Vector index | pgvector (HNSW) | Already using Neon PostgreSQL |
| Image processing | Sharp | Already in stack |
| OCR | Tesseract.js | Node-native, no external service |
| Training (offline) | PyTorch → ONNX export | One-time, not in production path |
| Batch processing | Bull queue (Redis) | Already using Redis |

---

## Implementation Phases

| Phase | Scope | Effort | Deliverable |
|-------|-------|--------|-------------|
| **P1** | Preprocessing (OCR + mask) + Pre-trained EfficientNet embeddings + pgvector ANN | 2 weeks | Working logo search (no training) |
| **P2** | Stage 2 re-ranking (spatial feature matching) | 1.5 weeks | Improved precision on top results |
| **P3** | Fine-tune EfficientNet with triplet loss on TMView data | 3 weeks | Major recall/precision improvement |
| **P4** | Composite mark integration (visual + phonetic) | 1 week | Full mark similarity (logo + text) |
| **P5** | Calibrate against opposition decisions, tune weights | 1 week | Validated accuracy metrics |

**P1 delivers a working system in 2 weeks** — same timeline as the old plan but
with a far stronger baseline (deep embeddings vs Hu Moments).

---

## Success Metrics

| Metric | Baseline (P1) | After Fine-tune (P3) | How to measure |
|--------|---------------|----------------------|----------------|
| Recall@100 | >60% | >85% | Known-similar pairs in top 100 |
| Precision@10 | >30% | >55% | Genuinely similar in top 10 |
| Query latency | <200ms | <200ms | End-to-end single query |
| Index size | ~2KB/mark | ~2KB/mark | 512-dim float32 = 2048 bytes |

---

## What We Dropped from v1 (and Why)

| Dropped | Reason |
|---------|--------|
| Hu Moments / Fourier Descriptors | Deep features strictly dominate; would be throwaway work |
| Vienna code hard filter | Kills recall due to classification subjectivity |
| Shape Context (custom TS impl) | High engineering risk; spatial deep features achieve same goal |
| Hand-crafted 139-dim vector | Replaced by 512-dim learned embedding (better in every metric) |
