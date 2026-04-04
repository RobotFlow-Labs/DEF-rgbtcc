# DEF-rgbtcc CUDA Server Handoff

This module is prepared for deployment to the CUDA host with Python 3.11 and `uv`.

## 1. Deploy
```bash
bash scripts/server/deploy_to_cuda_server.sh datai_srv7_development /mnt/forge-data/modules/wave-8/DEF-rgbtcc
```

## 2. Bootstrap Runtime
```bash
ssh datai_srv7_development
bash /mnt/forge-data/modules/wave-8/DEF-rgbtcc/scripts/server/bootstrap_cuda_server.sh /mnt/forge-data/modules/wave-8/DEF-rgbtcc
```

## 3. Validate Assets
```bash
cd /mnt/forge-data/modules/wave-8/DEF-rgbtcc
source .venv/bin/activate
python scripts/server/validate_assets.py \
  --dataset-root /mnt/forge-data/datasets/RGB-T-CC/RGBT-CC \
  --checkpoint /mnt/forge-data/models/rgbtcc/vgg_vit_depth_2_head_6.pth
```

## 4. Validate CUDA Runtime
```bash
python scripts/server/validate_cuda_runtime.py --checkpoint /mnt/forge-data/models/rgbtcc/vgg_vit_depth_2_head_6.pth
```

## 5. Run Benchmarks
```bash
bash scripts/server/run_benchmark_suite.sh /mnt/forge-data/modules/wave-8/DEF-rgbtcc /mnt/forge-data/models/rgbtcc/vgg_vit_depth_2_head_6.pth
```

## Required Inputs Before Full Training
1. RGBT-CC dataset mounted at `/mnt/forge-data/datasets/RGB-T-CC/RGBT-CC`.
2. Optional DroneRGBT dataset mounted at `/mnt/forge-data/datasets/RGB-T-CC/DroneRGBT`.
3. Compatible checkpoint at a known path.
