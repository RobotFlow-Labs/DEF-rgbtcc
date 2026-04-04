# Benchmarks — DEF-rgbtcc
# Dual-Modulation Framework for RGB-T Crowd Counting
# Paper: ArXiv 2509.17079
# Architecture: Shared VGG-19 + SMA Transformer (2L, 8H, d=512) + ACMF + Density Reg
# Datasets: RGBT-CC, DroneRGBT

## Paper Baseline Results (to reproduce)

### Published Results — RGBT-CC Dataset
| Method | GAME0 ↓ | GAME1 ↓ | GAME2 ↓ | GAME3 ↓ | MSE ↓ | Re ↓ |
|--------|---------|---------|---------|---------|-------|------|
| Ours (DM Framework) | TBD | TBD | TBD | TBD | TBD | TBD |

### Published Results — DroneRGBT Dataset
| Method | GAME0 ↓ | GAME1 ↓ | GAME2 ↓ | GAME3 ↓ | MSE ↓ | Re ↓ |
|--------|---------|---------|---------|---------|-------|------|
| Ours (DM Framework) | TBD | TBD | TBD | TBD | TBD | TBD |

### Our Reproduction
| Dataset | GAME0 ↓ | GAME1 ↓ | GAME2 ↓ | GAME3 ↓ | MSE ↓ | Re ↓ |
|---------|---------|---------|---------|---------|-------|------|
| RGBT-CC | TBD | TBD | TBD | TBD | TBD | TBD |
| DroneRGBT | TBD | TBD | TBD | TBD | TBD | TBD |

## Model Complexity

| Metric | Value |
|--------|-------|
| Total Parameters | TBD (from thop) |
| Trainable Parameters | TBD |
| FLOPs (224×224 input) | TBD (from thop) |
| FLOPs (640×480 input) | TBD |
| Model Size (fp32) | TBD MB |
| Model Size (fp16) | TBD MB |

## Ablation — Component Impact

### SMA vs Standard Attention (RGBT-CC)
| Attention Type | GAME0 ↓ | GAME3 ↓ | MSE ↓ | FPS | Notes |
|---------------|---------|---------|-------|-----|-------|
| Standard Multi-Head (no spatial decay) | TBD | TBD | TBD | TBD | Baseline transformer |
| SMA (with spatial decay) | TBD | TBD | TBD | TBD | Proposed method |
| **Delta** | **TBD** | **TBD** | **TBD** | **TBD** | SMA benefit |

### Fusion Method Ablation (RGBT-CC)
| Fusion Method | GAME0 ↓ | GAME3 ↓ | MSE ↓ | FPS |
|--------------|---------|---------|-------|-----|
| Add (rgb + t) | TBD | TBD | TBD | TBD |
| Max (max(rgb, t)) | TBD | TBD | TBD | TBD |
| Concat + Conv | TBD | TBD | TBD | TBD |
| ACMF (proposed) | TBD | TBD | TBD | TBD |

### Spatial Decay Parameter Analysis
| beta_scale init | beta_bias init | GAME0 | Notes |
|----------------|----------------|-------|-------|
| 0.9 (default) | 5.0 (default) | TBD | Paper default |
| 0.5 | 5.0 | TBD | Faster decay |
| 0.9 | 2.0 | TBD | Smaller radius |
| 0.9 | 10.0 | TBD | Larger radius |
| Learned (final) | Learned (final) | TBD | Report trained values |

### Backbone Ablation
| Backbone | GAME0 ↓ | Params (M) | FLOPs | FPS |
|----------|---------|-----------|-------|-----|
| VGG-19 (shared) | TBD | TBD | TBD | TBD |
| VGG-19 (separate) | TBD | TBD | TBD | TBD |
| ResNet-50 (shared) | TBD | TBD | TBD | TBD |

## FPS / Latency

### Per-Component Breakdown (B=1, 224×224, RTX 6000 Pro)
| Component | Time (ms) | % of Forward |
|-----------|----------|-------------|
| VGG-19 backbone (RGB) | TBD | TBD |
| VGG-19 backbone (Thermal) | TBD | TBD |
| SMA Transformer (RGB, 2 layers) | TBD | TBD |
| SMA Transformer (T, 2 layers) | TBD | TBD |
| Bilinear upsample (×2 streams) | TBD | TBD |
| ACMF Fusion | TBD | TBD |
| Density Reg Head | TBD | TBD |
| **Total** | **TBD** | **100%** |

### Within SMA Transformer — Detailed Breakdown
| Operation | Time (ms) | % of SMA |
|-----------|----------|---------|
| Spatial distance generation (cdist) | TBD | TBD |
| Spatial decay computation (sigmoid, pow) | TBD | TBD |
| QKV Linear projection | TBD | TBD |
| Attention (matmul + decay + softmax + matmul) | TBD | TBD |
| Output projection + residual | TBD | TBD |
| LayerNorm | TBD | TBD |
| FFN (Linear → ReLU → Linear) | TBD | TBD |
| **SMA Total (2 layers)** | **TBD** | **100%** |

### Resolution Scaling
| Input Size | N (spatial tokens) | Latency (ms) | FPS | GPU Memory (GB) |
|-----------|-------------------|-------------|-----|----------------|
| 224×224 | 196 (14×14) | TBD | TBD | TBD |
| 256×256 | 256 (16×16) | TBD | TBD | TBD |
| 384×384 | 576 (24×24) | TBD | TBD | TBD |
| 512×512 | 1024 (32×32) | TBD | TBD | TBD |
| 640×480 | 1200 (40×30) | TBD | TBD | TBD |

Note: Latency should scale super-linearly due to O(N²) attention.

### Hardware Comparison (224×224, B=1)
| Hardware | Latency (ms) | FPS | GPU Memory (GB) |
|----------|-------------|-----|----------------|
| RTX 6000 Pro Blackwell | TBD | TBD | TBD |
| Mac Studio M-series (MLX) | TBD | TBD | TBD |
| Jetson Orin NX (TRT FP16) | TBD | TBD | TBD |

### Batch Size Scaling (224×224, RTX 6000 Pro)
| Batch Size | Latency (ms) | Throughput (img/s) | GPU Memory (GB) |
|-----------|-------------|-------------------|----------------|
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD |

## Kernel Optimization Impact

| Kernel | Baseline (ms) | Optimized (ms) | Speedup |
|--------|--------------|----------------|---------|
| Spatial distance+decay (N=196) | TBD | TBD | TBD (target 4x) |
| Spatial distance+decay (N=1200) | TBD | TBD | TBD (target 5x) |
| SMA block (N=196) | TBD | TBD | TBD (target 2x) |
| ACMF + reg | TBD | TBD | TBD (target 1.5x) |
| Batched dual VGG | TBD | TBD | TBD (target 1.3x) |
| **Full forward (224×224)** | **TBD** | **TBD** | **TBD (target 1.7x)** |
| **Full forward (640×480)** | **TBD** | **TBD** | **TBD (target 2x)** |

## Training Performance

| Config | Dataset | Epochs | Time/epoch | GPU Memory (GB) | Best GAME0 | Best Epoch |
|--------|---------|--------|-----------|----------------|-----------|-----------|
| VGG-19 + SMA + ACMF | RGBT-CC | 400 | TBD | TBD | TBD | TBD |
| VGG-19 + SMA + ACMF | DroneRGBT | 400 | TBD | TBD | TBD | TBD |

### Training Convergence
| Epoch Range | Avg GAME0 | Avg Loss | Notes |
|-------------|----------|---------|-------|
| 0-50 | TBD | TBD | Warmup |
| 50-100 | TBD | TBD | Fast convergence |
| 100-200 | TBD | TBD | Refinement |
| 200-400 | TBD | TBD | Plateau |

## Dual-Compute Validation

| Backend | Dataset | GAME0 | MSE | FPS |
|---------|---------|-------|-----|-----|
| CUDA (RTX 6000 Pro) | RGBT-CC | TBD | reference | TBD |
| MLX (Mac Studio) | RGBT-CC | TBD (within 0.5%) | TBD | TBD |
| Jetson Orin NX (TRT FP16) | RGBT-CC | TBD (within 2%) | TBD | TBD |

## Cross-Module RGB-T Comparison

| Module | Task | Fusion Type | Key Metric | Params (M) | FPS |
|--------|------|------------|-----------|-----------|-----|
| DEF-rgbtcc | Counting | ACMF (channel attn) | GAME0 MAE | TBD | TBD |
| DEF-tuni | Seg | Local+Global Attn | mIoU | TBD | TBD |
| DEF-cmssm | Seg | Mamba SSM | mIoU | TBD | TBD |
| DEF-rtfdnet | Seg | CLIP alignment | mIoU | TBD | TBD |
| DEF-hypsam | SOD | SAM refinement | Sm | ~743M | TBD |

### Task Comparison: Counting vs Segmentation vs SOD
| Aspect | Counting (rgbtcc) | Segmentation (tuni/cmssm) | SOD (hypsam) |
|--------|-------------------|--------------------------|-------------|
| Output | Density map (count) | Class label map | Binary saliency |
| Metrics | GAME, MAE, MSE | mIoU, per-class IoU | Sm, Fm, Em, MAE |
| Use case | "How many people?" | "What class is each pixel?" | "What is salient?" |
| Defense use | Crowd monitoring, perimeter | Terrain classification | Threat detection |
| Model size | Small (~50M) | Medium (~20-100M) | Large (~750M) |
| Speed | Fast (40+ FPS) | Medium (15-30 FPS) | Slow (2-5 FPS) |

## Hardware & Methodology
- RTX 6000 Pro Blackwell (training + evaluation)
- Mac Studio M-series (MLX local dev)
- Jetson Orin NX (edge deployment)
- Default input: 224×224, RGB (3ch) + Thermal (3ch)
- Test input: 256×256 (from test_game.py)
- FPS measurement: 100 warmup + 500 iterations, mean ± std
- GAME metrics: eval_game() at L=0,1,2,3
- Results stored as JSON: `results_*.json`

---
*Updated 2026-04-04 by ANIMA Research Agent*
