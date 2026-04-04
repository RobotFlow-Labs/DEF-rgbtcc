# Custom Kernels — DEF-rgbtcc
# Dual-Modulation Framework for RGB-T Crowd Counting
# Architecture: Shared VGG-19 + SMA Transformer (×2) + ACMF Fusion + Density Reg
# Following /anima-optimize-cuda-pipeline Phase 3

## Architecture-Specific Kernel Targets

NOTE: This model is relatively lightweight compared to DEF-hypsam or DEF-cmssm.
The main bottleneck is the dual VGG-19 backbone (~40-50% of compute). The SMA
transformer with spatial distance matrix is the most unique component and scales
quadratically with spatial resolution — this is the key optimization target.

### Kernel 1: Fused Spatial Distance + Decay (`spatial_decay_attn.cu`)
**Bottleneck**: `generate_spatial_distance` computes pairwise L2 distances between all H/16 × W/16 spatial positions using `torch.cdist`. This creates an (N×N) matrix where N = (H/16)*(W/16). At 640×480, N=1200, matrix = 1200² = 1.44M elements × 8 heads. Then `process_spatial_decay` applies sigmoid, softplus, leaky_relu, and pow operations element-wise.
**Current**: 3 separate kernel launches: cdist → leaky_relu → pow, each materializing full N×N tensors
**Target**: 3-5x speedup, cached distance matrix

```
Input: H (int), W (int), beta_scale (nhead×1×1), beta_bias (nhead×1×1)
Output: decay_matrix (nhead×N×N) where N = (H/16) * (W/16)

Method: Single fused kernel with caching

  Phase 1: Distance Matrix (cached — only recompute when H,W change)
    For each pair (i, j) where i,j ∈ [0, N):
      h_i, w_i = i // W_feat, i % W_feat
      h_j, w_j = j // W_feat, j % W_feat
      dist[i,j] = sqrt((h_i - h_j)² + (w_i - w_j)²)
    --- One thread per (i,j) pair, coalesced writes ---
    --- Cache: if H,W unchanged from last call, skip ---

  Phase 2: Decay Computation (per head, fused)
    For each head h, pair (i,j):
      beta_s = sigmoid(beta_scale[h])
      beta_b = softplus(beta_bias[h])
      d = dist[i,j] - beta_b
      d_processed = d >= 0 ? d : 0.1 * d  // leaky_relu(d, 0.1)
      decay[h,i,j] = pow(beta_s, d_processed)
    --- All operations in-register, single write ---

Key insight: Distance matrix is input-independent (only depends on H,W).
Cache it and only recompute decay when beta parameters change (every step during training,
but constant during inference). During inference, the entire decay matrix can be precomputed
once and reused.
```

**Python wrapper**: `spatial_decay_attn(H, W, beta_scale, beta_bias, cache=True)` → decay_matrix
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/spatial_decay_attn/`
**REUSABLE by**: ANY model using distance-modulated attention (crowd counting, point cloud, spatial transformers)

### Kernel 2: Fused SMA Transformer Block (`sma_fused_block.cu`)
**Bottleneck**: Each SMA block has: QKV projection → reshape → spatially modulated attention → output proj → residual → LayerNorm → FFN → residual → LayerNorm. 2 blocks × 2 streams = 4 total calls. With 8 heads and N tokens, attention is O(N²) per head.
**Target**: 2x speedup for SMA transformer

```
Input: x (B×N×512), decay_matrix (8×N×N), block_weights
Output: x_out (B×N×512)

Method: Fused transformer block

  1. residual = x
  2. q = x @ Wq  [B×N×512 → B×N×8×64]
     k = x @ Wk  [same]
     v = x @ Wv  [same]
     --- Single GEMM for QKV, split in-register ---

  3. attn = (q @ k.T) / sqrt(64)
     attn = attn * decay_matrix  // spatial modulation
     attn = softmax(attn)
     attn = dropout(attn)
     out = attn @ v
     --- Modified flash attention with element-wise decay multiplication ---
     --- decay_matrix loaded into shared memory (fits for N≤196: 196²×4B=154KB per head) ---

  4. x = out @ Wo + residual
  5. x = LayerNorm(x)

  6. residual2 = x
  7. x = ReLU(x @ W_ff1 + b_ff1)
  8. x = x @ W_ff2 + b_ff2 + residual2
  9. x = LayerNorm(x)

Savings: 10+ ops → 2 kernel launches (attention + FFN)
```

**Python wrapper**: `sma_fused_block(x, decay_matrix, block_weights)` → x_out
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/sma_fused_block/`
**REUSABLE by**: Any spatially-modulated transformer

### Kernel 3: Fused ACMF + Upsample + Reg Head (`acmf_reg_fused.cu`)
**Bottleneck**: After SMA transformers, the pipeline does: bilinear upsample 2× (×2 streams) → ACMF fusion → 3-layer conv reg head. Current implementation creates 4 intermediate 512-channel feature maps.
**Target**: 1.5x speedup, 60% memory reduction for this stage

```
Input: rgb_feat (B×512×H'×W'), t_feat (B×512×H'×W')
       Where H'=H/16, W'=W/16 (pre-upsample)
       acmf_weights, reg_weights
Output: density (B×1×H/8×W/8)

Method: Fused pipeline

  Phase 1: Upsample + ACMF (single kernel)
    For each output pixel (h, w) at 2× resolution:
      rgb_up = bilinear_sample(rgb_feat, h/2, w/2)    [512-dim]
      t_up = bilinear_sample(t_feat, h/2, w/2)        [512-dim]
      combined = rgb_up + t_up
      pooled = global_avg(combined)                     [precomputed]
      w = sigmoid(conv1x1(relu(conv1x1(pooled))))       [scalar]
      fused = rgb_up * w + t_up * (1-w)                [512-dim]
    --- No materialized upsampled tensors ---

  Phase 2: Reg Head (single kernel)
    x = ReLU(Conv3x3_256(fused))
    x = ReLU(Conv3x3_128(x))
    density = Conv1x1_1(x)
    density = abs(density)
    --- 3 convs fused into single kernel with intermediate in registers ---

Savings: Eliminates 2× B×512×(2H')×(2W') upsampled tensors + fusion intermediate
```

**Python wrapper**: `acmf_reg_fused(rgb_feat, t_feat, acmf_weights, reg_weights)` → density
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/acmf_reg_fused/`

### Kernel 4: Batched Dual-Stream VGG (`dual_stream_vgg.py`)
**Not a CUDA kernel** — PyTorch-level optimization
**Target**: 1.3x backbone speedup

```
Method: Stack RGB and thermal into single 2B batch through shared VGG-19

  stacked = cat(rgb, thermal, dim=0)  [2B×3×H×W]
  features = vgg19(stacked)           [2B×512×H/16×W/16]
  rgb_feat = features[:B]
  t_feat = features[B:]

Benefit: Single backbone call → better GPU utilization
Since VGG weights are SHARED between streams, this is mathematically identical.
At batch_size=1 (default training), this doubles effective batch → much better GPU occupancy.
```

**Python wrapper**: Modify `Net.forward()` directly
**Save to**: In-model optimization, no separate kernel needed

## MLX Metal Equivalents

### SMA on MLX
1. **`spatial_decay_mlx.py`** — MLX spatial distance + decay
   ```python
   # Distance matrix
   coords = mx.stack(mx.meshgrid(mx.arange(h), mx.arange(w)), axis=-1)
   coords = coords.reshape(-1, 2).astype(mx.float32)
   dist = mx.sqrt(mx.sum((coords[:, None] - coords[None, :]) ** 2, axis=-1))

   # Decay
   beta_s = mx.sigmoid(beta_scale)
   beta_b = mx.softplus(beta_bias)
   processed = mx.where(dist - beta_b >= 0, dist - beta_b, 0.1 * (dist - beta_b))
   decay = mx.power(beta_s, processed)
   ```

2. **`sma_block_mlx.py`** — MLX SMA transformer block
   - Use `mx.fast.scaled_dot_product_attention` — but need to inject decay multiplication
   - May need custom attention: `attn = softmax(qk * decay) @ v`

3. **`acmf_mlx.py`** — MLX adaptive fusion
   - Straightforward: `mx.conv2d` + `mx.sigmoid` + weighted sum

4. **Weight conversion**: PyTorch → MLX safetensors
   - VGG-19: standard conv weight transposition
   - Transformer: QKV weight reshaping for head dimension

## Benchmark Targets

| Kernel | Baseline (ms) | Target (ms) | Speedup |
|--------|--------------|-------------|---------|
| Spatial distance+decay (224×224, N=196) | ~2.0 | ~0.5 | 4x |
| Spatial distance+decay (640×480, N=1200) | ~15.0 | ~3.0 | 5x |
| SMA block per call (N=196) | ~3.0 | ~1.5 | 2x |
| ACMF + upsample + reg | ~3.0 | ~2.0 | 1.5x |
| Batched dual VGG (B=1) | ~12.0 | ~9.0 | 1.3x |
| **Full forward (224×224)** | **~25** | **~15** | **1.7x** |
| **Full forward (640×480)** | **~80** | **~40** | **2x** |

Note: Higher resolution benefits more from kernel optimization because the
distance matrix scales O(N²) — optimization impact grows with resolution.

## Memory Analysis

| Input Size | N (spatial) | Distance Matrix | Decay Matrix (8 heads) | VGG Features |
|-----------|------------|----------------|----------------------|-------------|
| 224×224 | 196 | 150KB | 1.2MB | ~50MB |
| 384×384 | 576 | 1.3MB | 10MB | ~150MB |
| 640×480 | 1200 | 5.5MB | 44MB | ~300MB |

Distance matrix + decay scales quadratically. At 640×480, decay matrix alone is 44MB (8 heads × 1200² × 4B).
Caching eliminates recomputation; fusing eliminates intermediates.

## IP Notes

- **spatial_decay_attn.cu** is the most novel — first CUDA kernel for spatially-modulated attention with learnable distance decay. Reusable by any model that modulates attention by spatial distance (crowd counting, point cloud processing, spatial transformers). Patent-worthy as general-purpose distance-decay attention kernel.
- **sma_fused_block.cu** extends Flash Attention with element-wise pre-softmax modulation — a general pattern for any attention variant that multiplies attention logits by a matrix before softmax.
- **Caching strategy** for distance matrix is broadly applicable: any model with input-independent attention biases benefits.
- All kernels stored at `/mnt/forge-data/shared_infra/cuda_extensions/`.

---
*Updated 2026-04-04 by ANIMA Research Agent*
