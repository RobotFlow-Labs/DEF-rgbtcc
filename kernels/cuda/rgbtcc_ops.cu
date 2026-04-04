/**
 * DEF-rgbtcc: RGB-T Crowd Counting CUDA Kernels
 * 1. fused_spatial_distance_decay — Spatial distance matrix + exponential decay
 * 2. fused_density_upsample — Bilinear upsample + fusion for density regression
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void spatial_distance_decay_kernel(
    float* __restrict__ output,
    int H, int W, float sigma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W * H * W;
    if (idx >= total) return;
    int i = idx / (H * W);
    int j = idx % (H * W);
    int y1 = i / W, x1 = i % W;
    int y2 = j / W, x2 = j % W;
    float dy = (float)(y1 - y2);
    float dx = (float)(x1 - x2);
    float dist_sq = dx * dx + dy * dy;
    output[idx] = expf(-dist_sq / (2.0f * sigma * sigma));
}

__global__ void fused_density_blend_kernel(
    const float* __restrict__ rgb_feat,
    const float* __restrict__ thermal_feat,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float w = weight[idx];
    output[idx] = w * rgb_feat[idx] + (1.0f - w) * thermal_feat[idx];
}

torch::Tensor spatial_distance_decay(int H, int W, float sigma) {
    auto output = torch::empty({H * W, H * W}, torch::TensorOptions().device(torch::kCUDA));
    int total = H * W * H * W;
    spatial_distance_decay_kernel<<<(total+255)/256, 256>>>(
        output.data_ptr<float>(), H, W, sigma);
    return output;
}

torch::Tensor fused_density_blend(
    torch::Tensor rgb_feat, torch::Tensor thermal_feat, torch::Tensor weight
) {
    TORCH_CHECK(rgb_feat.is_cuda(), "must be CUDA");
    auto output = torch::empty_like(rgb_feat);
    int N = rgb_feat.numel();
    fused_density_blend_kernel<<<(N+255)/256, 256>>>(
        rgb_feat.data_ptr<float>(), thermal_feat.data_ptr<float>(),
        weight.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_distance_decay", &spatial_distance_decay, "Spatial distance + Gaussian decay (CUDA)");
    m.def("fused_density_blend", &fused_density_blend, "Fused RGB-T density blend (CUDA)");
}
