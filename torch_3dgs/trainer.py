import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange

from .camera import to_viewpoint_camera
from .metric import calc_psnr, calc_ssim, calc_l2_loss
from .render import GaussRenderer


class Trainer:
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        l1_weight: float = 1.,
        l2_weight: float = 1.,
        dssim_weight: float = 1.,
        depth_weight: float = 1.,
        lr: float = 1e-3,
        num_steps: int = 10000,
        eval_interval: int = 500,
        render_kwargs: Dict[str, Any] = None,
        logger: Optional[Any] = None,
        results_folder: str = "outputs",
    ) -> None:

        self.data = data
        self.model = model.to(device)
        self.device = device

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.dssim_weight = dssim_weight
        self.depth_weight = depth_weight
        self.lr = lr

        self.num_steps = num_steps
        self.eval_interval = eval_interval

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [200, 400, 800, 1200, 1600, 2000, 2400], gamma=0.5)
        self.gauss_render = GaussRenderer(**render_kwargs)

        self.logger = logger
        os.makedirs(results_folder, exist_ok=True)
        self.results_folder = Path(results_folder)

    def train_step(self, step) -> Dict[str, torch.Tensor]:
        self.optimizer.zero_grad()
        idx = np.random.choice(len(self.data["camera"]))
        camera = to_viewpoint_camera(self.data["camera"][idx])
        rgb = self.data["rgb"][idx]
        depth = self.data["depth"][idx]
        mask = self.data["alpha"][idx].bool()
    
        output = self.gauss_render(pc=self.model, camera=camera)
    
        # TODO: Compute L1 Loss
        # Hint: L1 loss measures absolute pixel-wise differences between the rendered image and ground truth.
        # l1_loss = ...
        l1_loss = F.l1_loss(rgb, output['render'])
        l2_loss = calc_l2_loss(rgb, output['render'])
        
        # TODO: Compute DSSIM Loss
        # Hint: DSSIM loss is derived from SSIM, a perceptual loss that compares structure, contrast, and luminance.
        # dssim_loss = ...
        SSIM = calc_ssim(rgb.permute(2, 0, 1), output['render'].permute(2, 0, 1))
        dssim_loss = 0.5 * (1-SSIM)
    
        # TODO: Compute Depth Loss
        # Hint: Compute depth error only where valid (using the mask).
        # depth_loss = ...
        depth_loss = F.l1_loss(depth[mask], output['depth'][mask].squeeze(-1))
    
        # TODO: Compute Total Loss
        # Hint: Combine all losses using respective weighting coefficients.
        # total_loss = ...
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss + self.dssim_weight * dssim_loss + self.depth_weight * depth_loss
    
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
        psnr = calc_psnr(output["render"], rgb)
    
        return {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "dssim_loss": dssim_loss,
            "depth_loss": depth_loss,
            "psnr": psnr,
        }

    def eval_step(self, step: int) -> None:
        frames = []
        for idx, camera_raw in enumerate(self.data["camera"]):
            camera = to_viewpoint_camera(camera_raw).to(self.device)
            rgb_gt = self.data["rgb"][idx].detach().cpu().numpy()
            depth_gt = self.data["depth"][idx].detach().cpu().numpy()

            output = self.gauss_render(pc=self.model, camera=camera)
            rgb_pred = output["render"].detach().cpu().numpy()
            depth_pred = output["depth"].detach().cpu().numpy()[..., 0]

            depth_img = np.concatenate([depth_gt, depth_pred], axis=1)
            depth_img = (1 - depth_img / depth_img.max())
            depth_img = plt.get_cmap("jet")(depth_img)[..., :3]

            rgb_img = np.concatenate([rgb_gt, rgb_pred], axis=1)
            final_image = np.concatenate([rgb_img, depth_img], axis=0)
            frames.append((final_image * 255).clip(0, 255).astype(np.uint8))


        output_path = os.path.join(self.results_folder, f"video_{step}.mp4")
        self.save_video(frames, output_path, fps=5)

        if self.logger is not None:
            self.logger.log(
                {
                    "rendered_video": self.logger.Video(output_path, format="mp4")
                }
            )

    def train(self) -> None:
        self.eval_step(0)
        pbar = trange(1, self.num_steps + 1)

        for step in pbar:
            outputs = self.train_step(step)
            results = {name: round(value.item(), 3) for name, value in outputs.items()}
            pbar.set_postfix(results)

                
            if step % self.eval_interval == 0:
                self.eval_step(step)
                self.save(step)
            
            if self.logger is not None:
                self.logger.log(results)

    def save(self, step: int) -> None:
        checkpoint = {
            "step": step,
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.results_folder / f"model_{step}.pt")

    def load(self, step: int) -> None:
        checkpoint = torch.load(self.results_folder / f"model_{step}.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["opt"])
        self.step = checkpoint["step"]

    def save_video(self, image_list: List[np.ndarray], output_path: str, fps: int = 30) -> None:
        if not image_list:
            raise ValueError("image_list is empty!")

        writer = imageio.get_writer(output_path, fps=fps)
        for image in image_list:
            writer.append_data(image)
        writer.close()
        print(f"Video saved to {output_path}")
