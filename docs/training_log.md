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

## Phase 3: LR Scheduling & Extension (Epochs 20-30)
**Strategy**: 
- **Action**: Extended training to 30 Epochs.
- **Optimization**: Introduced `CosineAnnealingLR` scheduler to gradually reduce learning rate from `3e-4` to `1e-6`.
- **Status**: Currently Running.

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

## Next Steps
1. Monitor Phase 3 (Epoch 20-30) to see if Cosine Annealing reduces the loss below 6.0.
2. Evaluate "Best Model" vs "Latest Model" on the `test_full_pipeline.py` using the Toy Validation pairs.
