import os
import sys
import torch
import wandb
from omegaconf import OmegaConf

from torch_3dgs.data import read_data
from torch_3dgs.trainer import Trainer
from torch_3dgs.model import GaussianModel
from torch_3dgs.point import get_point_clouds
from torch_3dgs.utils import dict_to_device


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    os.makedirs(config.output_folder, exist_ok=True)
    device = torch.device(config.device)

    data = read_data(config.data_folder, config.resize_scale, config.image_number)
    data = dict_to_device(data, device)

    points = get_point_clouds(
        data["camera"],
        data["depth"],
        data["alpha"],
        data["rgb"],
    )
    points.visualize(50)

    raw_points = points.random_sample(config.num_points)
    model = GaussianModel(sh_degree=4, debug=False)
    model.create_from_pcd(pcd=raw_points)

    wandb.init(
        project="EV-HW1",
        config=OmegaConf.to_container(config, resolve=True),
    )

    trainer = Trainer(
        data=data,
        model=model, 
        device=device,
        num_steps=config.num_steps,
        eval_interval=config.eval_interval,
        l1_weight=config.l1_weight,
        l2_weight=config.l2_weight,
        dssim_weight=config.dssim_weight,
        depth_weight=config.depth_weight,
        lr=config.lr,
        results_folder=config.output_folder,
        render_kwargs={
            "tile_size": config.render.tile_size,
        },
        logger=wandb,
    )
    trainer.train()
