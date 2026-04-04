"""DEF-rgbtcc inference serving module.

Provides AnimaNode-compatible interface for RGB-T crowd counting.
"""
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from def_rgbtcc.models.dm import Net


class RGBTCCInference:
    """Lightweight inference wrapper for RGB-T crowd counting."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Net()

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, rgb: Image.Image, thermal: Image.Image) -> dict:
        """Run crowd counting on RGB-thermal image pair.

        Returns:
            dict with 'count' (float) and 'density_map' (H x W numpy array)
        """
        rgb_tensor = self.transform(rgb.convert("RGB")).unsqueeze(0).to(self.device)
        t_tensor = self.transform(thermal.convert("RGB")).unsqueeze(0).to(self.device)

        density = self.model([rgb_tensor, t_tensor])
        density_np = density[0, 0].cpu().numpy()
        count = float(density_np.sum())

        return {
            "count": count,
            "density_map": density_np,
        }

    @torch.no_grad()
    def predict_bytes(self, rgb_bytes: bytes, thermal_bytes: bytes) -> dict:
        """Run prediction from raw image bytes."""
        rgb = Image.open(io.BytesIO(rgb_bytes))
        thermal = Image.open(io.BytesIO(thermal_bytes))
        return self.predict(rgb, thermal)
