/**
 * DEF-rgbtcc: RGB-T Crowd Counting CUDA Kernels (sm_89)
 *
 * 1. fused_spatial_distance_decay — L2 distance matrix + learnable decay (SMA paper)
 * 2. fused_density_blend — Weighted RGB-T feature fusion (ACMF)
 *
 * Paper: ArXiv 2509.17079
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// --- Kernel 1: Fused Spatial Distance + Learnable Decay ---
// Computes: dist[i,j] = L2(coord_i, coord_j)
// Then for each head h:
//   processed = leaky_relu(dist - softplus(beta_bias[h]), slope=0.1)
//   decay[h,i,j] = sigmoid(beta_scale[h]) ^ processed
__global__ void fused_spatial_distance_decay_kernel(
    float* __restrict__ output,     // (nhead, H*W, H*W)
    const float* __restrict__ beta_scale,  // (nhead,)
    const float* __restrict__ beta_bias,   // (nhead,)
    int H, int W, int nhead
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total = nhead * HW * HW;
    if (tid >= total) return;

    int h = tid / (HW * HW);
    int rem = tid % (HW * HW);
    int i = rem / HW;
    int j = rem % HW;

    // Compute L2 distance
    int y1 = i / W, x1 = i % W;
    int y2 = j / W, x2 = j % W;
    float dy = (float)(y1 - y2);
    float dx = (float)(x1 - x2);
    float dist = sqrtf(dx * dx + dy * dy);

    // Learnable decay per head
    float bs = 1.0f / (1.0f + expf(-beta_scale[h]));  // sigmoid(beta_scale)
    float bb = logf(1.0f + expf(beta_bias[h]));         // softplus(beta_bias)

    // leaky_relu(dist - bb, slope=0.1)
    float shifted = dist - bb;
    float processed = (shifted >= 0.0f) ? shifted : 0.1f * shifted;

    // pow(sigmoid(beta_scale), processed)
    output[tid] = powf(bs, processed);
}

torch::Tensor fused_spatial_distance_decay(
    int H, int W,
    torch::Tensor beta_scale,  // (nhead, 1, 1)
    torch::Tensor beta_bias    // (nhead, 1, 1)
) {
    TORCH_CHECK(beta_scale.is_cuda(), "beta_scale must be CUDA");
    int nhead = beta_scale.size(0);
    int HW = H * W;

    // Flatten beta params
    auto bs_flat = beta_scale.contiguous().view({nhead});
    auto bb_flat = beta_bias.contiguous().view({nhead});

    auto output = torch::empty({nhead, HW, HW}, beta_scale.options());

    int total = nhead * HW * HW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_spatial_distance_decay_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        bs_flat.data_ptr<float>(),
        bb_flat.data_ptr<float>(),
        H, W, nhead);

    return output;
}

// --- Kernel 2: Fused ACMF Density Blend ---
// output[i] = w * rgb[i] + (1-w) * thermal[i]
// w is broadcast from (B, 1, 1, 1) weight tensor
__global__ void fused_density_blend_kernel(
    const float* __restrict__ rgb_feat,
    const float* __restrict__ thermal_feat,
    const float* __restrict__ weight,  // (B,) one weight per batch
    float* __restrict__ output,
    int B, int C, int spatial  // spatial = H*W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * spatial;
    if (idx >= total) return;

    int b = idx / (C * spatial);
    float w = weight[b];
    output[idx] = w * rgb_feat[idx] + (1.0f - w) * thermal_feat[idx];
}

torch::Tensor fused_density_blend(
    torch::Tensor rgb_feat,       // (B, C, H, W)
    torch::Tensor thermal_feat,   // (B, C, H, W)
    torch::Tensor weight          // (B, 1, 1, 1)
) {
    TORCH_CHECK(rgb_feat.is_cuda(), "must be CUDA");
    int B = rgb_feat.size(0);
    int C = rgb_feat.size(1);
    int spatial = rgb_feat.size(2) * rgb_feat.size(3);

    auto w_flat = weight.contiguous().view({B});
    auto output = torch::empty_like(rgb_feat);

    int total = B * C * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_density_blend_kernel<<<blocks, threads>>>(
        rgb_feat.data_ptr<float>(),
        thermal_feat.data_ptr<float>(),
        w_flat.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, spatial);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_spatial_distance_decay", &fused_spatial_distance_decay,
          "Fused spatial distance + learnable decay per head (CUDA)");
    m.def("fused_density_blend", &fused_density_blend,
          "Fused RGB-T density blend with per-batch weights (CUDA)");
}
