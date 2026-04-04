"""
DEF-rgbtcc: Dual-Modulation Framework for RGB-T Crowd Counting
Paper: ArXiv 2509.17079

Architecture:
  RGB/T -> shared VGG-19 -> SMA Transformer -> Upsample 2x -> ACMF -> Reg Head -> |Density|
"""
import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from def_rgbtcc.kernels.spatial_decay_attn import (
    fused_density_blend,
    fused_spatial_distance_decay,
    is_cuda_available,
)


class AdaptiveCrossModalFusion(nn.Module):
    """Channel-attention-based adaptive fusion of RGB and thermal features."""

    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat: Tensor, t_feat: Tensor) -> Tensor:
        combined = rgb_feat + t_feat
        w = self.mlp(self.avg_pool(combined))
        return fused_density_blend(rgb_feat, t_feat, w)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, sph: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if sph is not None:
            attn = attn * sph
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        p_attn = self.dropout(F.softmax(attn, dim=-1))
        return torch.matmul(p_attn, v), p_attn


class SpatiallyModulatedAttention(nn.Module):
    """Multi-head attention with spatial distance decay modulation."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, sph: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, sph=sph, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        q = q.permute(1, 0, 2)
        return q, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048,
        dropout: float = 0.1, activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = SpatiallyModulatedAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

    def forward(
        self, src: Tensor, sph: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        q = k = src
        src2, attn = self.self_attn(q, k, src, sph=sph, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, nhead: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.nhead = nhead
        self.beta_scale = nn.Parameter(torch.full((nhead, 1, 1), 0.9))
        self.beta_bias = nn.Parameter(torch.full((nhead, 1, 1), 5.0))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src: Tensor, mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bs, c, h, w = src.shape
        sph = fused_spatial_distance_decay(h, w, self.beta_scale, self.beta_bias)
        x = src.flatten(2).permute(2, 0, 1)
        for layer in self.layers:
            x, _ = layer(x, sph=sph, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x.permute(1, 2, 0).view(bs, c, h, w)


VGG19_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
             512, 512, 512, 512, "M", 512, 512, 512, 512]


def _make_vgg_layers(cfg: list, batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Net(nn.Module):
    """Dual-Modulation RGB-T Crowd Counting Network."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.features = _make_vgg_layers(VGG19_CFG)
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, "relu"
        )
        self.transformer_encoder_rgb = TransformerEncoder(
            encoder_layer, num_encoder_layers, nhead
        )
        self.transformer_encoder_t = TransformerEncoder(
            encoder_layer, num_encoder_layers, nhead
        )
        self.fusion_layer = AdaptiveCrossModalFusion(in_channels=d_model)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, inputs: list[Tensor] | tuple[Tensor, Tensor]) -> Tensor:
        rgb, t = inputs
        # Handle mismatched RGB/thermal sizes (some datasets have rotated thermal)
        if rgb.shape != t.shape:
            t = F.interpolate(t, size=rgb.shape[2:], mode="bilinear", align_corners=False)
        rgb_feat = self.features(rgb)
        t_feat = self.features(t)
        rgb_feat = self.transformer_encoder_rgb(rgb_feat)
        t_feat = self.transformer_encoder_t(t_feat)
        rgb_feat = F.interpolate(rgb_feat, scale_factor=2)
        t_feat = F.interpolate(t_feat, scale_factor=2)
        fusion = self.fusion_layer(rgb_feat, t_feat)
        density = self.reg_layer(fusion)
        return torch.abs(density)
