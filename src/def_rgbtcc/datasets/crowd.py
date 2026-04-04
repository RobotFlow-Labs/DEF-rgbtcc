"""RGB-T Crowd Counting dataset loader.

Expects directory structure:
  {root}/{split}/
    {id}_RGB.jpg
    {id}_T.jpg
    {id}_GT.json  ({"points": [[x,y], ...], "count": N})
"""
import json
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial.distance import cdist
from torchvision import transforms


def random_crop(im_h: int, im_w: int, crop_h: int, crop_w: int):
    i = random.randint(0, im_h - crop_h)
    j = random.randint(0, im_w - crop_w)
    return i, j, crop_h, crop_w


def cal_inner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    return np.maximum(inner_right - inner_left, 0.0) * np.maximum(
        inner_down - inner_up, 0.0
    )


def compute_nearest_distance(points: np.ndarray) -> np.ndarray:
    """Compute nearest neighbor distance for each point."""
    if len(points) <= 1:
        return np.full(len(points), 15.0)
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists, axis=1)


class Crowd(data.Dataset):
    def __init__(self, root_path: str, crop_size: int, downsample_ratio: int, method: str):
        self.root_path = root_path
        self.rgb_list = sorted(glob(os.path.join(self.root_path, "*_RGB.jpg")))
        if method not in ("train", "val", "test"):
            raise ValueError(f"method must be train/val/test, got {method}")
        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.rgb_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.t_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rgb_list)

    def _load_gt(self, rgb_path: str) -> np.ndarray:
        """Load GT from JSON. Returns Nx3 array [x, y, nearest_dist]."""
        gt_path = rgb_path.replace("_RGB.jpg", "_GT.json")
        with open(gt_path) as f:
            data = json.load(f)
        points_xy = np.array(data["points"], dtype=np.float64)
        if len(points_xy) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        nearest_dist = compute_nearest_distance(points_xy)
        return np.column_stack([points_xy, nearest_dist])

    def __getitem__(self, item):
        rgb_path = self.rgb_list[item]
        t_path = rgb_path.replace("_RGB.jpg", "_T.jpg")

        rgb = Image.open(rgb_path).convert("RGB")
        t = Image.open(t_path).convert("RGB")

        keypoints = self._load_gt(rgb_path)

        if self.method == "train":
            return self._train_transform(rgb, t, keypoints)
        else:
            shape = cv2.imread(rgb_path)[..., ::-1].copy().shape
            k = np.zeros((shape[0], shape[1]))
            for i in range(len(keypoints)):
                if int(keypoints[i][1]) < shape[0] and int(keypoints[i][0]) < shape[1]:
                    k[int(keypoints[i][1]), int(keypoints[i][0])] = 1
            rgb = self.rgb_trans(rgb)
            t = self.t_trans(t)
            name = os.path.basename(rgb_path).split(".")[0]
            return rgb, t, k, len(keypoints), name

    def _train_transform(self, rgb, t, keypoints):
        wd, ht = rgb.size
        st_size = min(wd, ht)
        if st_size < self.c_size:
            # Resize up if image is smaller than crop
            scale = self.c_size / st_size + 0.01
            rgb = rgb.resize((int(wd * scale), int(ht * scale)), Image.BILINEAR)
            t = t.resize((int(wd * scale), int(ht * scale)), Image.BILINEAR)
            keypoints[:, :2] *= scale
            wd, ht = rgb.size
            st_size = min(wd, ht)

        if len(keypoints) == 0:
            # No annotations — return zero-padded
            i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
            rgb = TF.crop(rgb, i, j, h, w)
            t = TF.crop(t, i, j, h, w)
            return (
                self.rgb_trans(rgb),
                self.t_trans(t),
                torch.zeros(0, 2, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
                st_size,
            )

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        rgb = TF.crop(rgb, i, j, h, w)
        t = TF.crop(t, i, j, h, w)

        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_inner_area(j, i, j + w, i + h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = ratio >= 0.3

        target = ratio[mask]
        kp_masked = keypoints[mask]
        kp_masked = kp_masked[:, :2] - [j, i]
        if len(kp_masked) > 0:
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                t = TF.hflip(t)
                kp_masked[:, 0] = w - kp_masked[:, 0]
        else:
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                t = TF.hflip(t)

        return (
            self.rgb_trans(rgb),
            self.t_trans(t),
            torch.from_numpy(kp_masked.copy()).float(),
            torch.from_numpy(target.copy()).float(),
            st_size,
        )


def train_collate(batch):
    transposed = list(zip(*batch))
    rgb = torch.stack(transposed[0], 0)
    t = torch.stack(transposed[1], 0)
    points = transposed[2]
    targets = transposed[3]
    st_sizes = torch.FloatTensor(transposed[4])
    return rgb, t, points, targets, st_sizes
