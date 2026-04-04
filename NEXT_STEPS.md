# NEXT_STEPS — DEF-rgbtcc
## Last Updated: 2026-04-04
## Status: ENRICHED — Ready for build
## MVP Readiness: 0%
## Total PRDs: 10 (55 hours estimated)
## Critical Path: PRD-001 → PRD-002 → PRD-003 → PRD-004 → PRD-005 → PRD-006

---

### Immediate Next Actions
1. Clone repo: `git clone https://github.com/Cht2924/RGBT-Crowd-Counting.git`
2. Create uv env: `uv venv .venv --python 3.10`
3. Install PyTorch cu128
4. Install opencv-python, numpy, pillow, tqdm, thop
5. **Find and download pretrained weights**: `vgg_vit_depth_2_head_6.pth` (custom VGG+ViT checkpoint — check repo releases)
6. Download RGBT-CC dataset from chen-judge/RGBTCrowdCounting
7. Download DroneRGBT dataset from VisDrone/DroneRGBT
8. Train model on RGBT-CC (400 epochs, batch=1, lr=1e-5, Adam)
9. Evaluate: GAME0-3, MSE, Relative Error — reproduce paper results

### What This Module Does
A dual-modulation framework for RGB-T crowd counting that uses **Spatially Modulated Attention (SMA)** — transformer attention weighted by a learnable spatial distance decay — and **Adaptive Cross-Modal Fusion (ACMF)** — channel-attention-based dynamic RGB/thermal weighting. Shared VGG-19 backbone extracts features from both modalities, separate SMA transformers (2 layers, 8 heads, d=512) process each stream with spatial locality priors, then ACMF adaptively fuses them before a simple 3-layer conv density regression head. Output is an absolute density map — count = sum of all pixels. Lightweight architecture (~50M params), potentially 40+ FPS at 224×224. ArXiv 2509.17079.

### Key Results to Reproduce
- RGBT-CC: GAME0 (MAE), GAME1, GAME2, GAME3, MSE, Relative Error
- DroneRGBT: same metrics
- Ablation: SMA vs standard attention, ACMF vs add/max/cat fusion

### TODO (by PRD)
- [ ] **PRD-001**: Environment setup — PyTorch + pretrained weights (4h)
- [ ] **PRD-002**: Dataset download — RGBT-CC + DroneRGBT (4h)
- [ ] **PRD-003**: Training — RGBT-CC 400 epochs (6h)
- [ ] **PRD-004**: Evaluation — reproduce paper results (5h)
- [ ] **PRD-005**: FPS benchmarking — multi-resolution (3h)
- [ ] **PRD-006**: CUDA profiling — SMA + distance matrix bottleneck (5h)
- [ ] **PRD-007**: Custom CUDA kernels — 4 targets (12h)
- [ ] **PRD-008**: MLX port (6h)
- [ ] **PRD-009**: Edge deployment Jetson — TRT FP16, 20+ FPS target (4h)
- [ ] **PRD-010**: RGB-T family integration (3h)

### Blockers
- **Pretrained weights**: `vgg_vit_depth_2_head_6.pth` URL not in README. Need to check GitHub releases or contact author. Could train from scratch using standard VGG-19 ImageNet weights as fallback.
- **RGBT-CC dataset**: Available from chen-judge/RGBTCrowdCounting — need to find download link.
- **DroneRGBT dataset**: Available from VisDrone/DroneRGBT — check for Google Drive links.
- **Training time**: 400 epochs × batch=1 → long training. May need patience or multi-GPU.

### Datasets/Models Needed
- RGBT-CC dataset (~1-2GB) — from chen-judge/RGBTCrowdCounting
- DroneRGBT dataset (~1-2GB) — from VisDrone/DroneRGBT
- `vgg_vit_depth_2_head_6.pth` pretrained (~100MB) — custom checkpoint
- Total: ~3-5GB datasets + ~100MB models

### Kernel IP Targets (shared across ANIMA)
1. **Spatial Distance + Decay Kernel** → NOVEL: first CUDA kernel for distance-modulated attention with learnable decay. Reusable by any spatial transformer, crowd counting, point cloud model. (3-5x speedup, patent-worthy)
2. **Fused SMA Block** → Flash Attention variant with pre-softmax element-wise modulation. General-purpose extension to Flash Attention. (2x speedup)
3. **Fused ACMF + Reg** → Eliminates intermediate allocations in fusion + regression pipeline. (1.5x speedup)
4. **Batched Dual VGG** → Same concept as DEF-cmssm Kernel 4 — shared backbone, batch both streams. (1.3x speedup)

### Related Modules
- **DEF-tuni** — RGB-T semantic segmentation, real-time focus, complementary task
- **DEF-cmssm** — RGB-T semantic segmentation, Mamba SSM, complementary task
- **DEF-rtfdnet** — RGB-T semantic segmentation, CLIP-based fusion
- **DEF-hypsam** — RGB-T salient object detection, SAM-based refinement

### Key Architecture Insight: Spatial Distance Decay
The SMA transformer encodes a physical distance prior: `decay = sigmoid(beta_scale)^(leaky_relu(dist - softplus(beta_bias)))`. This means:
- **beta_bias** controls the "attention radius" — positions closer than beta_bias attend strongly
- **beta_scale** controls how quickly attention decays with distance
- Both are **learnable per attention head** (8 heads × 2 params = 16 total spatial priors)
- This is particularly effective for crowd counting where local density patterns matter more than global context
- The decay matrix is **input-independent** (depends only on H,W) — can be cached during inference

### Note on Task Uniqueness
This is the only **crowd counting** module in Wave 8 (and all of ANIMA). While DEF-tuni/cmssm/rtfdnet do segmentation and DEF-hypsam does SOD, crowd counting answers "how many people?" with a density map. This is directly applicable to surveillance (ORACLE), fleet safety (ATLAS), and crowd-aware navigation (NEMESIS). The lightweight architecture makes it ideal for real-time edge deployment — a strong demo candidate for Shenzhen.

---
*Updated 2026-04-04 by ANIMA Research Agent*
