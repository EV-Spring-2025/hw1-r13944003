import os
import json
import random
from typing import Dict

import imageio
import torch
import torch.nn.functional as F
from einops import rearrange


def read_data(folder: str, resize_scale: float = 1.0, image_number: int = 200) -> Dict[str, torch.Tensor]:
    camera = read_camera(folder)
    
    
    rgb_files, poses, intrisics = camera["rgb_files"], camera["poses"], camera["intrinsics"]
    idx = random.sample(range(len(rgbs)), image_number)
    rgb_files, poses, intrisics = rgb_files[idx], poses[idx], intrisics[idx]

    all_rgbs, all_depths, all_alphas, all_cameras = zip(*[
        read_image(rgb_file, pose, intrinsic, camera["max_depth"], resize_scale)
        for rgb_file, pose, intrinsic in zip(rgb_files, poses, intrisics)
    ])

    rgbs = torch.stack(all_rgbs)
    depths = torch.stack(all_depths)
    alphas = torch.stack(all_alphas)
    cameras = torch.stack(all_cameras)

    rgbs = alphas[..., None] * rgbs + (1 - alphas)[..., None]

    return {
        "rgb": rgbs,
        "camera": cameras,
        "depth": depths,
        "alpha": alphas,
    }


def read_camera(folder: str) -> tuple:
    with open(os.path.join(folder, "info.json")) as f:
        scene_info = json.load(f)

    coord_convert = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))
    max_depth = scene_info["max_depth"]

    rgb_files, poses, intrinsics = [], [], []
    for img in scene_info["images"]:
        rgb_files.append(os.path.join(folder, img["rgb"]))
        pose = torch.tensor(img["pose"], dtype=torch.float32) @ coord_convert
        poses.append(pose)
        intrinsics.append(torch.tensor(img["intrinsic"], dtype=torch.float32))

    return {
        "rgb_files": rgb_files,
        "poses": poses,
        "intrinsics": intrinsics,
        "max_depth": max_depth,
    }


def resize_image_tensor(img, scale):
    img = rearrange(img, "h w c -> 1 c h w") if img.ndim == 3 else rearrange(img, "h w -> 1 1 h w")
    img = F.interpolate(img, scale_factor=scale, mode="bilinear", align_corners=False)
    return img[0].permute(1, 2, 0) if img.shape[1] > 1 else img[0, 0]


def read_image(
    rgb_path: str,
    pose: torch.Tensor,
    intrinsic_3x3: torch.Tensor,
    max_depth: float = 1.0,
    resize_scale: float = 1.0,
) -> tuple:

    base_path = rgb_path[:-7]
    rgb = torch.tensor(imageio.imread(rgb_path), dtype=torch.float32) / 255.0
    depth = torch.tensor(imageio.imread(base_path + "depth.png"), dtype=torch.float32) / 255.0 * max_depth
    alpha = torch.tensor(imageio.imread(base_path + "alpha.png"), dtype=torch.float32) / 255.0

    if resize_scale != 1.0:
        rgb = resize_image_tensor(rgb, resize_scale)
        depth = resize_image_tensor(depth, resize_scale)
        alpha = resize_image_tensor(alpha, resize_scale)

    h, w = rgb.shape[:2]
    intrinsic = torch.eye(4, dtype=torch.float32)
    intrinsic[:3, :3] = intrinsic_3x3
    intrinsic[:2, :3] *= resize_scale

    camera_params = torch.cat(
        [
            torch.tensor([h, w], dtype=torch.float32),
            intrinsic.flatten(),
            pose.flatten()
        ]
    )

    rgb = alpha[..., None] * rgb + (1 - alpha)[..., None]

    return rgb, depth, alpha, camera_params
