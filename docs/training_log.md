# Training Log Report

## Experiment Overview
- **Model**: EfficientNet-B0 (1280 dim)
- **Method**: MoCo v3 (Self-Supervised Learning)
- **Dataset**: ~538k Trademark Images
- **Infrastructure**: Vast.ai (RTX 3060/4090 tier equivalent)

## Phase 1: Initial Weak Supervision (Epochs 0-10)
**Strategy**: Positive pairs were sampled based on shared Vienna Codes.
- **Observation**: Loss plateaued around `7.0`.
- **Analysis**: Weak supervision introduced too much noise; visual similarity was not being effectively learned because "positive" pairs were often visually distinct despite sharing a category code.

## Phase 2: Switch to Instance Discrimination (SSL) (Epochs 10-20)
**Strategy**: Switched to standard MoCo practice. Positive pairs are two augmented views of the *same* image.
- **Goal**: Learn invariance to augmentation (crop, color jitter, grayscale).
- **Results**:
  - Epoch 10: Loss 7.0350
  - Epoch 11: Loss 7.0305
  - ...
  - Epoch 19: Loss 7.2163
- **Observation**: Loss remained relatively high and showed signs of oscillation/slight divergence towards Epoch 19.
- **Hypothesis**: Learning rate (`3e-4`) might be too high for this stage of convergence, or the model needs scheduling to settle.

## Phase 3: LR Scheduling & Extension (Epochs 20-50)
**Strategy**: 
- **Action**: Extended training to **50 Epochs** (initially 30) for deeper convergence.
- **Optimization**: `CosineAnnealingLR` scheduler (from `3e-4` to `1e-6`).
- **Resumption**: Logic updated to ensure `T_max` correctly reflects new limit when resuming from checkpoints.
- **Status**: Currently Running (~Epoch 24).

## Detailed Logs (Phase 2 Recap)
```text
Epoch 10 complete. Avg Loss: 7.0350
Epoch 11 complete. Avg Loss: 7.0305
Epoch 12 complete. Avg Loss: 7.0213
Epoch 13 complete. Avg Loss: 7.0109
Epoch 14 complete. Avg Loss: 7.0013
Epoch 15 complete. Avg Loss: 7.0030
Epoch 16 complete. Avg Loss: 7.0732
Epoch 17 complete. Avg Loss: 7.0308
Epoch 18 complete. Avg Loss: 7.1003
Epoch 19 complete. Avg Loss: 7.2163
```

## Phase 3 Observations (Ongoing)
- **Epoch 24 Loss**: ~4.79 (Significant drop from Epoch 19's ~7.2).
- **Behavior**: The introduction of the scheduler successfully broke the loss plateau. Loss is consistently decreasing without divergence.
- **Evaluation (Epoch 24 @ Toy Validation)**:
  - **Retrieval Performance (Vienna Code Matches)**: Top-10 Accuracy: 0/10 | Top-100 Accuracy: 0/10.
  - **Analysis**: While the model is learning strong **self-invariance** (lowering MoCo loss), it has not yet converged enough to capture complex semantic/figurative relationships defined by Vienna codes (which often have high intra-class variance).
  - **Qualitative**: Top matches show some basic shape/density correlation (e.g., matching text-heavy logos with other text-heavy logos), indicating a "visual shape" baseline is forming.

## Next Steps
1. Continue training to Epoch 50 to allow the learning rate to reach its minimum.
2. Re-evaluate every 5-10 epochs to monitor semantic retrieval improvement.
3. Run `test_full_pipeline.py` with the "Best Model" for qualitative visual inspection.
