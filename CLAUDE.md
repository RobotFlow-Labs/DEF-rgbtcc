# DEF-rgbtcc — A Dual-Modulation Framework for RGB-T Crowd Counting
# Wave 8 Defense Module
# Paper: "A Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion"
# Authors: Cht2924 et al.
# ArXiv: 2509.17079
# Repo: https://github.com/Cht2924/RGBT-Crowd-Counting
# Domain: RGB-T Crowd Counting via Spatially Modulated Attention + Adaptive Cross-Modal Fusion
# Product Stack: ORACLE (surveillance autonomy) / ATLAS (fleet AV perimeter safety) / NEMESIS (crowd-aware nav)
# Related: DEF-tuni (RGB-T seg), DEF-cmssm (RGB-T seg), DEF-rtfdnet (RGB-T seg), DEF-hypsam (RGB-T SOD)

## Status: ENRICHED — Ready for build

## Paper Summary
This paper presents a **dual-modulation framework** for RGB-T crowd counting that uses two key innovations: (1) **Spatially Modulated Attention (SMA)** — a transformer encoder with learnable spatial distance decay that modulates attention weights based on physical distance between spatial positions, and (2) **Adaptive Cross-Modal Fusion (ACMF)** — a channel-attention-based fusion module that dynamically weights RGB vs thermal features. Uses a shared VGG-19 backbone with separate SMA transformer encoders for each modality, followed by adaptive fusion and density map regression. Evaluated on RGBT-CC and DroneRGBT datasets. ArXiv: 2509.17079.

## Architecture

### Overview
```
RGB Image ──→ VGG-19 (shared) ──→ SMA Transformer (RGB) ──→ Upsample 2× ──┐
                                                                              ├──→ ACMF ──→ Reg Head ──→ |Density Map|
Thermal Image ──→ VGG-19 (shared) ──→ SMA Transformer (T) ──→ Upsample 2× ──┘
```

### Backbone: VGG-19 (shared weights)
- Standard VGG-19 configuration: [64,64,M,128,128,M,256,256,256,256,M,512,512,512,512,M,512,512,512,512]
- No BatchNorm (plain VGG)
- Input: 3-channel RGB or 3-channel thermal (both normalized with ImageNet stats)
- Output: 512-channel feature maps at 1/16 resolution
- **Shared weights** between RGB and thermal streams
- Pretrained: `vgg_vit_depth_2_head_6.pth` (custom VGG+ViT pretrained checkpoint)

### Spatially Modulated Attention (SMA) — KEY INNOVATION
The SMA is a transformer encoder that incorporates spatial distance information into attention computation via learnable distance decay parameters.

- **TransformerEncoder**: `num_layers=2`, `d_model=512`, `nhead=8`, `dim_feedforward=2048`, `dropout=0.1`
- **Spatial Distance Matrix**: Pairwise L2 distance between all spatial positions
  ```python
  # For feature map H×W:
  coords = meshgrid(arange(H), arange(W))  # (H*W, 2)
  dist_matrix = cdist(coords, coords, p=2)  # (H*W, H*W)
  ```
- **Spatial Decay Processing**: Learnable parameters modulate distance decay per attention head
  ```python
  # beta_scale: (nhead, 1, 1), init 0.9 — controls decay rate
  # beta_bias: (nhead, 1, 1), init 5.0 — controls distance threshold
  beta_scale_sigmoid = sigmoid(beta_scale)
  beta_bias_softplus = softplus(beta_bias)
  processed_dist = leaky_relu(dist_matrix - beta_bias_softplus, negative_slope=0.1)
  decay_matrix = pow(beta_scale_sigmoid, processed_dist)
  ```
- **Attention with Spatial Modulation**:
  ```python
  attn = (Q @ K.T) / sqrt(d_k)
  attn = attn * decay_matrix  # <-- spatial modulation BEFORE softmax
  attn = softmax(attn)
  output = attn @ V
  ```
- **Key insight**: Nearby crowd members influence each other's density prediction more than distant ones. The spatial decay encodes this locality prior into attention, making it particularly effective for crowd counting where spatial density patterns matter.

### Adaptive Cross-Modal Fusion (ACMF)
- Channel-attention-based adaptive weighting of RGB vs thermal features:
  ```python
  combined = rgb_feat + t_feat
  w = Sigmoid(Conv1x1(ReLU(Conv1x1(AvgPool(combined)))))  # SE-like attention
  # w: (B, 1, 1, 1) — scalar weight per sample
  fusion = rgb_feat * w + t_feat * (1 - w)
  ```
- `reduction_ratio = 8` (512 → 64 → 1)
- Complementary weighting: w for RGB, (1-w) for thermal — ensures features sum properly
- AdaptiveAvgPool2d(1) for global context

### Density Regression Head
- Simple 3-layer conv head:
  ```python
  reg = Sequential(
      Conv2d(512, 256, kernel_size=3, padding=1), ReLU,
      Conv2d(256, 128, kernel_size=3, padding=1), ReLU,
      Conv2d(128, 1, kernel_size=1)
  )
  ```
- Output: absolute density map (`torch.abs(density)`)
- Count = sum of density map pixels

### Full Forward Pass
```python
def forward(self, inputs):
    rgb, t = inputs
    rgb_feat = self.features(rgb)          # VGG-19: (B,3,H,W) → (B,512,H/16,W/16)
    t_feat = self.features(t)              # VGG-19: shared weights
    rgb_feat = self.transformer_encoder_rgb(rgb_feat)  # SMA: (B,512,H/16,W/16) → same
    t_feat = self.transformer_encoder_t(t_feat)        # SMA: separate params
    rgb_feat = F.interpolate(rgb_feat, scale_factor=2) # Upsample: → (B,512,H/8,W/8)
    t_feat = F.interpolate(t_feat, scale_factor=2)     # Upsample
    fusion = self.fusion_layer(rgb_feat, t_feat)       # ACMF: → (B,512,H/8,W/8)
    density = self.reg_layer(fusion)                   # Reg: → (B,1,H/8,W/8)
    return torch.abs(density)
```

### Training Configuration (from `train.py` + `regression_trainer.py`)
- **Optimizer**: Adam, lr=1e-5, weight_decay=1e-4
- **Epochs**: 400
- **Batch size**: 1
- **Crop size**: 224×224
- **Downsample ratio**: 8 (density map at 1/8 resolution = 28×28)
- **Loss**: Bayesian Loss (`Bay_Loss`) with posterior probability
  - `sigma = 8.0` (Gaussian kernel width)
  - `background_ratio = 0.15`
  - `use_background = True`
- **Evaluation**: GAME (Grid Average Mean Error) at levels 0-3 + MSE + Relative Error
- **Seed**: 3407 (deterministic training)
- **Pretrained**: `vgg_vit_depth_2_head_6.pth` (custom VGG+ViT checkpoint)

### Loss Function: Bayesian Loss
- `Post_Prob`: Computes posterior probability of each annotation point belonging to each density map cell
  - Uses Gaussian kernel (sigma=8) for spatial assignment
  - Background probability added as extra class (ratio=0.15)
  - Softmax over all points + background per cell
- `Bay_Loss`: L1 loss between predicted count per cell and target assignment
  - `loss = |target - sum(density * prob)|` averaged over batch

## Datasets Used
- **RGBT-CC** — RGB-T Crowd Counting benchmark
  - From: https://github.com/chen-judge/RGBTCrowdCounting
  - Split: train/val/test
  - Format: `{id}_RGB.jpg`, `{id}_T.jpg`, `{id}_GT.npy` (keypoints with nearest-neighbor distances)
- **DroneRGBT** — Drone-based RGB-T crowd counting
  - From: https://github.com/VisDrone/DroneRGBT
  - Aerial perspective with thermal
- Input: 3-channel RGB + 3-channel thermal (ImageNet-normalized)
- Ground truth: point annotations (x, y, nearest_distance) → density maps via Bayesian loss

## Key Results (from paper — to reproduce)
- RGBT-CC: GAME0 (MAE), GAME1, GAME2, GAME3, MSE, Relative Error
- DroneRGBT: same metrics
- Paper comparison figure shows competitive with state-of-the-art methods

## Dependencies
- Python 3.8+ (we use 3.10)
- PyTorch 2.0+ + CUDA
- torchvision
- opencv-python (cv2 for image loading in dataset)
- numpy, PIL
- tqdm
- thop (optional, for FLOPs counting)
- No heavy dependencies — clean, lightweight codebase

## Repo Structure
```
RGBT-Crowd-Counting/
├── README.md
├── train.py                    — Training entry point (argparse + RegTrainer)
├── test_game.py                — Test with GAME metrics
├── models/
│   └── dm.py                   — THE CORE: Net, SpatiallyModulatedAttention, ACMF,
│                                  TransformerEncoder, VGG-19, reg head (~10KB)
├── datasets/
│   └── crowd.py                — Crowd dataset loader (RGB-T pairs + GT keypoints)
├── losses/
│   ├── bay_loss.py             — Bayesian Loss (L1 on posterior-weighted counts)
│   └── post_prob.py            — Posterior Probability computation (Gaussian assignment)
├── utils/
│   ├── regression_trainer.py   — RegTrainer: train/val/test loops, checkpointing
│   ├── evaluation.py           — GAME metrics (grid average mean error L=0-3)
│   ├── helper.py               — Save_Handle, AverageMeter utilities
│   ├── trainer.py              — Base Trainer class
│   └── logger.py               — Logging setup
└── comparison_figure.png       — Results comparison figure
```

## Build Requirements
- [ ] Clone repo: `git clone https://github.com/Cht2924/RGBT-Crowd-Counting.git`
- [ ] Create uv env: `uv venv .venv --python 3.10`
- [ ] Install PyTorch cu128: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- [ ] Install extras: `uv pip install opencv-python numpy pillow tqdm thop`
- [ ] Download pretrained: `vgg_vit_depth_2_head_6.pth` (custom checkpoint)
- [ ] Download RGBT-CC dataset from chen-judge/RGBTCrowdCounting
- [ ] Download DroneRGBT dataset from VisDrone/DroneRGBT
- [ ] Train model (400 epochs) or obtain pretrained checkpoint
- [ ] Evaluate: GAME0-3, MSE, Relative Error
- [ ] Profile and benchmark
- [ ] Build custom CUDA kernels
- [ ] Port to MLX
- [ ] Dual-compute validation

## CUDA Kernel Targets
1. **Fused Spatial Distance + Decay** — `generate_spatial_distance` + `process_spatial_decay` called every forward pass. Fuse distance computation + decay into single kernel with cached distance matrix.
2. **Fused SMA Transformer Block** — LayerNorm → SpatiallyModulatedAttention → FFN → LayerNorm in single kernel.
3. **Fused ACMF** — AvgPool + MLP + sigmoid + weighted sum — eliminate intermediate allocations.
4. **Fused Density Regression** — Upsample + fusion + reg head in single pass.

## Defense Marketplace Value
RGB-T crowd counting is directly applicable to **surveillance autonomy** (ORACLE — through-wall people counting for building security), **fleet safety** (ATLAS — counting pedestrians in vehicle path), and **crowd-aware navigation** (NEMESIS — avoiding dense crowds in autonomous missions). The spatially modulated attention is particularly suited for defense: it encodes physical distance priors, which are more reliable than learned spatial patterns when generalizing to new environments. The lightweight architecture (VGG-19 + 2-layer transformer) enables real-time deployment. Drone perspective support (DroneRGBT) is directly relevant for UAV surveillance.

## Package Manager: uv (NEVER pip)
## Python: >= 3.10
## Torch: cu128 index
## Git prefix: [DEF-rgbtcc]
