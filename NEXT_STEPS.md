# NEXT_STEPS — DEF-rgbtcc
## Last Updated: 2026-04-04
## Status: PRD + TASKS FINALIZED, SCAFFOLD READY
## MVP Readiness: 35%

## Completed
1. Full PRD finalized in `PRD.md`.
2. Detailed task slices completed in `tasks/PRD-001.md` .. `tasks/PRD-010.md`.
3. CUDA handoff operational scripts prepared in `scripts/server/`.
4. Package scaffolding prepared in `src/def_rgbtcc/`.

## Immediate Execution Order
1. `PRD-001`: bootstrap Python 3.11 environment on CUDA server.
2. `PRD-002`: validate dataset/checkpoint contract.
3. `PRD-005`: capture baseline benchmarks.
4. `PRD-006`: profile and lock hotspot roadmap.

## Command Pack (CUDA server)
```bash
bash scripts/server/deploy_to_cuda_server.sh datai_srv7_development /mnt/forge-data/modules/wave-8/DEF-rgbtcc
ssh datai_srv7_development
bash /mnt/forge-data/modules/wave-8/DEF-rgbtcc/scripts/server/bootstrap_cuda_server.sh /mnt/forge-data/modules/wave-8/DEF-rgbtcc
cd /mnt/forge-data/modules/wave-8/DEF-rgbtcc
source .venv/bin/activate
python scripts/server/validate_assets.py --dataset-root /mnt/forge-data/datasets/RGB-T-CC/RGBT-CC --checkpoint /mnt/forge-data/models/rgbtcc/vgg_vit_depth_2_head_6.pth
python scripts/server/validate_cuda_runtime.py --checkpoint /mnt/forge-data/models/rgbtcc/vgg_vit_depth_2_head_6.pth
bash scripts/server/run_benchmark_suite.sh /mnt/forge-data/modules/wave-8/DEF-rgbtcc /mnt/forge-data/models/rgbtcc/vgg_vit_depth_2_head_6.pth
```

## Open Blockers
1. Final checkpoint path must be confirmed.
2. Dataset mount paths must be validated on server.
3. Kernel implementation work starts after profiling lock.
