# DEF-rgbtcc — RGB-T Crowd Counting (Surveillance)

**Wave 8 Defense Module**

**Repo**: https://github.com/Cht2924/RGBT-Crowd-Counting

**Domain**: RGB-T Surveillance

**Stack**: ORACLE (surveillance) / ATLAS (fleet AV)

## Status: ⬜ Not Started

## Context
RGBTCC provides RGB-thermal crowd counting for autonomous surveillance and fleet safety applications. Combines visible and thermal imaging for robust people detection and counting in varied lighting and thermal conditions. Critical for safe autonomous navigation in crowded spaces and perimeter security.

## Build Requirements
- [ ] Clone repo and verify builds
- [ ] Create tasks/ PRD breakdown
- [ ] Implement CUDA kernel optimizations (following /anima-optimize-cuda-pipeline)
- [ ] Implement MLX equivalent
- [ ] Run benchmark suite (latency, throughput, memory)
- [ ] Dual-compute validation (MLX + CUDA)

## CUDA Kernel Targets
- [ ] Identify bottleneck operations (profile first)
- [ ] Fused RGB-T feature extraction and fusion kernels
- [ ] Density map regression kernels (optimized for crowd counting)
- [ ] INT8/FP16 quantized counting pipeline
- [ ] Save kernels to /mnt/forge-data/shared_infra/cuda_extensions/

## Package Manager: uv (NEVER pip)
## Python: >= 3.10
