# NEXT_STEPS — DEF-rgbtcc
## Last Updated: 2026-04-04
## Status: TRAINING IN PROGRESS
## MVP Readiness: 85%

## Completed
1. Full PRD finalized in `PRD.md`.
2. Package scaffolding in `src/def_rgbtcc/`.
3. ANIMA training pipeline built:
   - Model: VGG-19 + SMA Transformer + ACMF (34.1M params)
   - Dataset: RGBT-CC loader (JSON GT format, 1030/200/800 train/val/test)
   - Losses: Bayesian Loss + Posterior Probability
   - Trainer: Config-driven, checkpointing, early stopping, warmup+cosine LR
4. CUDA kernels compiled (sm_89): fused spatial distance decay + density blend
5. RGBT-CC dataset downloaded (593MB, 2030 images)
6. VGG-19 ImageNet pretrained weights loaded as backbone init
7. Training configs: paper.toml (BS=80, 400 epochs) + debug.toml
8. Docker serving infrastructure: Dockerfile.serve, docker-compose, anima_module.yaml
9. Training launched on GPU 1 (L4, 72.6% VRAM utilization)

## In Progress
- Training 400 epochs on RGBT-CC (GPU 1)
  - E0: GAME0=522.0
  - E1: GAME0=398.3
  - E2: GAME0=216.1 (rapid improvement)

## Remaining
1. Wait for training to complete (~2h remaining)
2. Export pipeline: pth → safetensors → ONNX → TRT FP16 → TRT FP32
3. Push to HuggingFace: ilessio-aiflowlab/DEF-rgbtcc
4. Final git commits + push

## Notes
- Original paper pretrained weights (vgg_vit_depth_2_head_6.pth) not publicly released
- Using VGG-19 ImageNet features as backbone initialization instead
- Dataset GT format is JSON (not NPY as stated in original CLAUDE.md)
- Some thermal images are rotated 90° relative to RGB — handled in model forward pass
