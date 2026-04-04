# Tasks — DEF-rgbtcc
# A Dual-Modulation Framework for RGB-T Crowd Counting
# Total PRDs: 10 | Estimated Hours: 55
# Critical Path: PRD-001 → PRD-002 → PRD-003 → PRD-004 → PRD-005 → PRD-006

---

## PRD-001: Environment Setup + Pretrained Weights (4h)
**Priority**: P0 — BLOCKER for everything
**Dependencies**: None

### Objective
Set up the full RGBT-CC environment. This is a lightweight codebase — no heavy dependencies.

### Steps
```bash
# 1. Clone repo
cd /mnt/forge-data/shared_infra/repos/
git clone https://github.com/Cht2924/RGBT-Crowd-Counting.git
cd RGBT-Crowd-Counting

# 2. Create uv environment
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. Install PyTorch cu128
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install extras
uv pip install opencv-python numpy pillow tqdm thop

# 5. Download pretrained weights
# The repo references: pretrained/vgg_vit_depth_2_head_6.pth
# This is a custom VGG+ViT pretrained checkpoint — check repo releases or author
mkdir -p pretrained/
# wget <URL> -O pretrained/vgg_vit_depth_2_head_6.pth

# 6. Verify model builds
python -c "from models.dm import Net; m = Net(); print('Model OK, params:', sum(p.numel() for p in m.parameters()))"

# 7. Quick forward pass test
python models/dm.py  # runs built-in test with thop FLOPs counting
```

### Blockers
- `vgg_vit_depth_2_head_6.pth` pretrained checkpoint URL not in README — need to find from releases or contact author
- thop optional but useful for FLOPs analysis

### Acceptance Criteria
- [ ] `from models.dm import Net` imports successfully
- [ ] Forward pass with (1,3,224,224) RGB + thermal produces output
- [ ] Output shape: (1, 1, 28, 28) — density map at 1/8 resolution
- [ ] FLOPs and params counted via thop
- [ ] Pretrained weights downloaded (or training from scratch planned)

---

## PRD-002: Dataset Download — RGBT-CC + DroneRGBT (4h)
**Priority**: P0 — BLOCKER for training + eval
**Dependencies**: None (parallel with PRD-001)

### Objective
Download both RGB-T crowd counting datasets.

### Steps
```bash
# Dataset storage
mkdir -p /mnt/forge-data/datasets/RGB-T-CC/

# RGBT-CC dataset
# Source: https://github.com/chen-judge/RGBTCrowdCounting
# Contains RGB-T pairs with point annotations
# Format: {id}_RGB.jpg, {id}_T.jpg, {id}_GT.npy
# Split: train/ val/ test/
cd /mnt/forge-data/datasets/RGB-T-CC/
# Download from official project page
# Expected structure:
# RGBT-CC/
#   train/
#     0001_RGB.jpg, 0001_T.jpg, 0001_GT.npy
#     ...
#   val/
#     ...
#   test/
#     ...

# DroneRGBT dataset
# Source: https://github.com/VisDrone/DroneRGBT
# Drone-perspective RGB-T crowd counting
cd /mnt/forge-data/datasets/RGB-T-CC/
# Download from VisDrone project page

# Verify dataset structure
ls RGBT-CC/train/ | head -10
ls RGBT-CC/test/ | head -10

# Verify GT format
python -c "
import numpy as np
gt = np.load('RGBT-CC/train/0001_GT.npy')
print(f'GT shape: {gt.shape}, columns: x, y, nearest_distance')
print(f'Sample: {gt[:3]}')
"

# Create symlinks
cd /mnt/forge-data/shared_infra/repos/RGBT-Crowd-Counting/
ln -s /mnt/forge-data/datasets/RGB-T-CC/RGBT-CC data/RGBT-CC
ln -s /mnt/forge-data/datasets/RGB-T-CC/DroneRGBT data/DroneRGBT
```

### Notes
- GT format: numpy array with columns [x, y, nearest_neighbor_distance]
- nearest_neighbor_distance used for Bayesian loss spatial assignment
- Images: `{id}_RGB.jpg` + `{id}_T.jpg` naming convention
- Thermal: 3-channel (RGB-replicated from thermal grayscale)

### Acceptance Criteria
- [ ] RGBT-CC train/val/test directories populated with RGB-T pairs + GT
- [ ] DroneRGBT downloaded with similar structure
- [ ] GT files load as numpy arrays with 3 columns (x, y, nn_dist)
- [ ] Dataset loader works: `Crowd(path, 224, 8, 'test')` returns samples

---

## PRD-003: Training — RGBT-CC Dataset (6h)
**Priority**: P0 — Need trained model for evaluation
**Dependencies**: PRD-001, PRD-002

### Objective
Train the Dual-Modulation model on RGBT-CC.

### Steps
```bash
cd /mnt/forge-data/shared_infra/repos/RGBT-Crowd-Counting/

# Train with default config
python train.py \
  --data-dir /mnt/forge-data/datasets/RGB-T-CC/RGBT-CC \
  --save-dir ./checkpoints/rgbtcc/ \
  --pretrained-model pretrained/vgg_vit_depth_2_head_6.pth \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --max-epoch 400 \
  --batch-size 1 \
  --crop-size 224 \
  --downsample-ratio 8 \
  --sigma 8.0 \
  --background-ratio 0.15 \
  --seed 3407 \
  --device 0

# Monitor: loss, MAE, MSE per epoch
# Best model saved to: checkpoints/rgbtcc/best_model.pth
# Validation: GAME0, GAME1, GAME2, GAME3, MSE, Relative Error
```

### Training Details
- 400 epochs, batch_size=1, crop_size=224
- Adam optimizer, lr=1e-5, weight_decay=1e-4
- Bayesian Loss with sigma=8, background_ratio=0.15
- Deterministic training (seed=3407)
- Expected time: ~8-12h on RTX 6000 Pro (400 epochs × batch=1)
- Best model saved when val GAME0 or GAME3 improves

### Acceptance Criteria
- [ ] Training runs 400 epochs without error
- [ ] Loss converges (check training curves)
- [ ] best_model.pth saved with best validation GAME0/GAME3
- [ ] Val GAME0 (MAE) competitive with paper results

---

## PRD-004: Evaluation — RGBT-CC + DroneRGBT (5h)
**Priority**: P0 — Reproduce paper results
**Dependencies**: PRD-003

### Objective
Evaluate trained model on both datasets using GAME metrics.

### Steps
```bash
cd /mnt/forge-data/shared_infra/repos/RGBT-Crowd-Counting/

# Test on RGBT-CC
python test_game.py \
  --data-dir /mnt/forge-data/datasets/RGB-T-CC/RGBT-CC \
  --save-dir ./checkpoints/rgbtcc/best_model.pth \
  --device 0

# Output: GAME0, GAME1, GAME2, GAME3, MSE, Relative Error, per-image MAE

# If DroneRGBT has separate training:
python train.py \
  --data-dir /mnt/forge-data/datasets/RGB-T-CC/DroneRGBT \
  --save-dir ./checkpoints/dronergbt/ \
  --max-epoch 400

python test_game.py \
  --data-dir /mnt/forge-data/datasets/RGB-T-CC/DroneRGBT \
  --save-dir ./checkpoints/dronergbt/best_model.pth
```

### GAME Metrics Explanation
| Metric | Grid Level | Description |
|--------|-----------|-------------|
| GAME0 | 1×1 | MAE — whole image count error |
| GAME1 | 2×2 | Mean absolute error in 4 quadrants |
| GAME2 | 4×4 | Mean absolute error in 16 sub-regions |
| GAME3 | 8×8 | Mean absolute error in 64 sub-regions |
| MSE | 1×1 | Root mean squared error |
| Relative | 1×1 | Relative counting error |

### Acceptance Criteria
- [ ] GAME0-3, MSE, Relative Error computed on RGBT-CC test set
- [ ] Results within ±5% of paper-reported values
- [ ] DroneRGBT evaluated (if trained separately)
- [ ] Per-image predictions logged for analysis
- [ ] Results saved as JSON for benchmarks

---

## PRD-005: FPS Benchmarking (3h)
**Priority**: P1
**Dependencies**: PRD-001

### Objective
Measure inference latency for the full pipeline.

### Steps
```bash
python -c "
import torch, time
from models.dm import Net

model = Net().cuda().eval()
rgb = torch.randn(1, 3, 224, 224).cuda()
t = torch.randn(1, 3, 224, 224).cuda()

# Warmup
for _ in range(100):
    with torch.no_grad():
        _ = model([rgb, t])
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(500):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model([rgb, t])
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

import numpy as np
print(f'Latency: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms')
print(f'FPS: {1000/np.mean(times):.1f}')
"

# Also test at higher resolutions
# 256×256, 384×384, 512×512, 640×480
```

### Expected Performance
| Input Size | Expected Latency | Expected FPS | Notes |
|-----------|-----------------|-------------|-------|
| 224×224 | ~15-25ms | ~40-65 | Training resolution |
| 384×384 | ~30-50ms | ~20-33 | Higher accuracy |
| 640×480 | ~60-100ms | ~10-17 | Full resolution |

**Note**: Spatial distance matrix is O(N²) where N = H/16 × W/16 features. At 224×224, N=14×14=196. At 640×480, N=40×30=1200. The distance matrix grows quadratically.

### Acceptance Criteria
- [ ] FPS measured at 4+ resolutions
- [ ] Per-component breakdown: VGG, SMA transformer, fusion, reg head
- [ ] Memory usage measured at each resolution
- [ ] Batch size sweep: B=1,2,4,8

---

## PRD-006: CUDA Profiling — SMA + Distance Matrix Bottleneck (5h)
**Priority**: P1
**Dependencies**: PRD-001, PRD-005

### Objective
Profile the model to identify optimization targets. The spatial distance matrix computation is expected to be a significant bottleneck at higher resolutions.

### Steps
```bash
# Profile full forward pass
nsys profile --trace=cuda,nvtx --output=rgbtcc_profile \
  python profile_rgbtcc.py --input_size 224

# Profile at higher resolution where distance matrix matters more
nsys profile --trace=cuda,nvtx --output=rgbtcc_profile_640 \
  python profile_rgbtcc.py --input_size 640

# Key analysis:
# 1. VGG-19 backbone time vs SMA transformer time
# 2. Distance matrix generation overhead (cdist is O(N²))
# 3. Decay matrix computation (sigmoid, softplus, pow operations)
# 4. Attention with spatial modulation vs standard attention
# 5. ACMF fusion overhead
```

### Expected Bottleneck Distribution (224×224)
| Component | Expected % | Notes |
|-----------|-----------|-------|
| VGG-19 backbone (×2 streams) | ~40-50% | Largest compute, shared weights |
| SMA Transformer (×2 streams) | ~30-35% | Distance matrix + attention |
| ACMF Fusion | ~5% | Lightweight SE-like |
| Upsample (2×) | ~3% | Bilinear interpolation |
| Reg head | ~5% | 3 conv layers |
| Distance matrix generation | ~5-10% | cdist, cached within encoder |

### Acceptance Criteria
- [ ] nsys profiles generated at 224×224 and 640×480
- [ ] Per-component time breakdown documented
- [ ] Distance matrix scaling analyzed (196 vs 1200 spatial positions)
- [ ] Top-5 hotspot kernels identified
- [ ] Memory peak measured

---

## PRD-007: Custom CUDA Kernels — 4 Targets (12h)
**Priority**: P1
**Dependencies**: PRD-006

### Objective
Build custom CUDA kernels for the identified bottlenecks.

### Kernel Targets
1. **Fused Spatial Distance + Decay Kernel** (4h)
   - Fuse `generate_spatial_distance` + `process_spatial_decay` into single kernel
   - Cache distance matrix across forward passes (only depends on H,W)
   - Target: 3-5x speedup for distance computation
   - Save to: `/mnt/forge-data/shared_infra/cuda_extensions/spatial_decay_attn/`

2. **Fused SMA Block** (4h)
   - Fuse: LayerNorm → QKV proj → spatial decay attention → output proj → residual → LayerNorm → FFN → residual
   - Use Flash Attention variant with spatial decay modulation
   - Target: 2x speedup for SMA transformer
   - Save to: `/mnt/forge-data/shared_infra/cuda_extensions/sma_fused_block/`

3. **Fused ACMF + Reg Head** (2h)
   - Fuse: AvgPool → MLP → sigmoid → weighted sum → Conv chain
   - Eliminate intermediate 512-channel tensors
   - Target: 1.5x speedup for fusion + regression
   - Save to: `/mnt/forge-data/shared_infra/cuda_extensions/acmf_fused/`

4. **Batched Dual-Stream VGG** (2h)
   - PyTorch-level: stack RGB+thermal into 2B batch, single VGG pass
   - Same as DEF-cmssm Kernel 4 concept — shared weights enable this
   - Target: 1.3x speedup for backbone
   - Save to: reuse pattern from DEF-cmssm

### Acceptance Criteria
- [ ] Each kernel compiles and passes correctness tests
- [ ] Each kernel shows measurable speedup
- [ ] Kernels saved to shared_infra
- [ ] Python wrappers with clean API

---

## PRD-008: MLX Port (6h)
**Priority**: P2
**Dependencies**: PRD-001, PRD-004

### Objective
Port the model to MLX for local development on Apple Silicon.

### Key Challenges
1. **VGG-19 on MLX**: Straightforward — standard conv layers
2. **SMA Transformer on MLX**: `torch.cdist` → `mx.linalg.norm` for distance matrix. Spatial decay via `mx.power`.
3. **ACMF on MLX**: SE-like attention — simple conv + sigmoid
4. **Bayesian Loss on MLX**: Softmax assignment — direct port

### Steps
```bash
# 1. Convert weights: PyTorch → MLX safetensors
python scripts/convert_to_mlx.py \
  --checkpoint checkpoints/rgbtcc/best_model.pth \
  --output checkpoints/rgbtcc_mlx/

# 2. Implement MLX model (dm_mlx.py)
# 3. Validate: MLX vs CUDA output within tolerance
# 4. Measure MLX FPS on Mac Studio
```

### Acceptance Criteria
- [ ] Model runs on MLX (Apple Silicon)
- [ ] Output match: MLX vs CUDA within atol=1e-4 (fp32)
- [ ] MLX FPS measured on Mac Studio
- [ ] GAME0 matches within ±0.5%

---

## PRD-009: Edge Deployment — Jetson Orin NX (4h)
**Priority**: P2 — DEMO VALUE for surveillance
**Dependencies**: PRD-007

### Objective
Deploy for real-time crowd counting on Jetson (surveillance demo).

### Strategy
- VGG-19 is well-supported in TensorRT
- SMA transformer: 2 layers, 8 heads, 196 tokens — fits easily
- Export full model to ONNX → TensorRT FP16
- Target: 20+ FPS at 224×224 on Jetson

### Acceptance Criteria
- [ ] TensorRT FP16 model runs on Jetson
- [ ] ≥20 FPS at 224×224 input
- [ ] GAME0 degradation <5% from FP32
- [ ] Memory fits within Jetson 16GB budget

---

## PRD-010: RGB-T Family Integration (3h)
**Priority**: P3
**Dependencies**: PRD-004

### Objective
Compare and integrate with other RGB-T modules.

### Comparisons
| Module | Task | Shared Patterns |
|--------|------|----------------|
| DEF-tuni | Seg | Thermal preprocessing, ACMF-like fusion |
| DEF-cmssm | Seg | Shared VGG backbone pattern, dual-stream |
| DEF-hypsam | SOD | RGB-T fusion, complementary tasks |

### Acceptance Criteria
- [ ] Cross-module thermal preprocessing analysis
- [ ] Shared kernel opportunities documented
- [ ] Count → density → segmentation pipeline explored (counting + seg combined)
- [ ] Report on surveillance pipeline: count (this) → segment (TUNI) → track (future)

---

*Updated 2026-04-04 by ANIMA Research Agent*
