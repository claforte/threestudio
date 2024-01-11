import bisect
import math
import random
import sys
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from uncond import RandomCameraDataset

sys.path.append("/home/kplanes2")

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class RandomCameraGridDataModuleConfig:
    # random camera
    heights: Any = 64
    widths: Any = 64
    batch_sizes: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    elevations: List[float] = field(default_factory=lambda: [-10, 0, 10, 20, 30, 40])
    num_azimuths: Any = 16
    camera_distance: Any = 1.5
    fovy: Any = 40

    # eval
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    n_val_views: int = 1
    n_test_views: int = 120


class RandomCameraGridIterableDataset(IterableDataset, Updateable):
    def create_views(self, milestone_index):
        assert (
            len(self.cfg.heights) == len(self.cfg.widths) == len(self.cfg.batch_sizes)
        )

        assert len(self.cfg.heights) == len(self.cfg.resolution_milestones) + 1
        self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focal = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.cfg.heights, self.cfg.widths)
        ]

        self.batch_size = self.cfg.batch_sizes[milestone_index]
        height = torch.tensor([self.cfg.heights[milestone_index]], dtype=torch.int32)
        width = torch.tensor([self.cfg.widths[milestone_index]], dtype=torch.int32)
        fovy = torch.tensor([self.cfg.fovy], dtype=torch.float32)
        camera_distance = torch.tensor([self.cfg.camera_distance], dtype=torch.float32)

        self.rays_o = []
        self.rays_d = []
        self.mvp_mtxs = []
        self.camera_positions = []
        self.c2ws = []
        self.light_positions = []
        self.elevations = []
        self.azimuths = []
        self.camera_distances = []
        self.heights = []
        self.widths = []

        for elevation in self.cfg.elevations:
            elevation = torch.tensor([elevation], dtype=torch.float32)
            for index_azimuth in range(self.cfg.num_azimuths):
                phi = torch.tensor(
                    [2 * np.pi * index_azimuth / self.cfg.num_azimuths],
                    dtype=torch.float32,
                )
                theta = torch.tensor([np.deg2rad(90 - elevation)], dtype=torch.float32)
                camera_position = torch.stack(
                    [
                        self.cfg.camera_distance * torch.cos(theta) * torch.cos(phi),
                        self.cfg.camera_distance * torch.cos(theta) * torch.sin(phi),
                        self.cfg.camera_distance * torch.sin(theta),
                    ],
                    dim=-1,
                )
                center = torch.tensor([0, 0, 0], dtype=torch.float32).unsqueeze(0)
                up = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0)
                lookat = F.normalize(center - camera_position, dim=-1)
                right = F.normalize(torch.cross(lookat, up), dim=-1)
                up = F.normalize(torch.cross(right, lookat), dim=-1)
                c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                    [
                        torch.stack([right, up, -lookat], dim=-1),
                        camera_position[:, :, None],
                    ],
                    dim=-1,
                )
                c2w: Float[Tensor, "B 4 4"] = torch.cat(
                    [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
                )
                c2w[:, 3, 3] = 1.0

                focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)
                directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                    milestone_index
                ][None, :, :, :].repeat(1, 1, 1, 1)
                directions[:, :, :, :2] = (
                    directions[:, :, :, :2] / focal_length[:, None, None, None]
                )

                # Importance note: the returned rays_d MUST be normalized!
                rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

                proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                    fovy, width / height, 0.1, 1000.0
                )
                mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

                azimuth = torch.tensor(
                    360 * (index_azimuth / self.cfg.num_azimuths), dtype=torch.float32
                ).unsqueeze(-1)

                self.rays_o.append(rays_o.squeeze(0))
                self.rays_d.append(rays_d.squeeze(0))
                self.mvp_mtxs.append(mvp_mtx.squeeze(0))
                self.camera_positions.append(camera_position.squeeze(0))
                self.c2ws.append(c2w.squeeze(0))
                self.light_positions.append(camera_position.squeeze(0))
                self.elevations.append(elevation.squeeze(0))
                self.azimuths.append(azimuth.squeeze(0))
                self.camera_distances.append(camera_distance.squeeze(0))
                self.heights.append(height.squeeze(0))
                self.widths.append(width.squeeze(0))

        self.rays_o = torch.stack(self.rays_o, dim=0)
        self.rays_d = torch.stack(self.rays_d, dim=0)
        self.mvp_mtxs = torch.stack(self.mvp_mtxs, dim=0)
        self.camera_positions = torch.stack(self.camera_positions, dim=0)
        self.c2ws = torch.stack(self.c2ws, dim=0)
        self.light_positions = torch.stack(self.light_positions, dim=0)
        self.elevations = torch.stack(self.elevations, dim=0)
        self.azimuths = torch.stack(self.azimuths, dim=0)
        self.camera_distances = torch.stack(self.camera_distances, dim=0)
        self.heights = torch.stack(self.heights, dim=0)
        self.widths = torch.stack(self.widths, dim=0)

        self.current_milestone_index = milestone_index

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraGridDataModuleConfig = cfg
        self.create_views(0)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        milestone_index = (
            bisect.bisect_right(self.resolution_milestones, global_step) - 1
        )
        if self.current_milestone_index != milestone_index:
            self.create_views(milestone_index)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        item_indices = torch.randint(len(self.rays_o), (self.batch_size,))

        return {
            "index": item_indices,
            "rays_o": self.rays_o[item_indices],
            "rays_d": self.rays_d[item_indices],
            "mvp_mtx": self.mvp_mtxs[item_indices],
            "camera_position": self.camera_positions[item_indices],
            "c2w": self.c2ws[item_indices],
            "light_position": self.light_positions[item_indices],
            "elevation": self.elevations[item_indices],
            "azimuth": self.azimuths[item_indices],
            "camera_distances": self.camera_distances[item_indices],
            "height": self.heights[item_indices],
            "width": self.widths[item_indices],
        }


@register("random-camera-datamodule")
class RandomCameraGridDataModule(pl.LightningDataModule):
    cfg: RandomCameraGridDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraGridDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraGridIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )


if __name__ == "__main__":
    cfg = {
        "heights": [64, 128, 256],
        "widths": [64, 128, 256],
        "batch_sizes": [6, 3, 2],
        "resolution_milestones": [200, 300],
        "elevations": [-10, 0, 10, 20, 30, 40],
        "num_azimuths": 16,
        "camera_distance": 3.8,
        "fovy": 20.0,
    }

    data_module = RandomCameraGridDataModule(cfg)
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    a = next(iter(loader))
    print(a)
    loader.dataset.update_step(0, 0)
    a = next(iter(loader))
    print(a)
    loader.dataset.update_step(0, 201)
    a = next(iter(loader))
    print(a)
    loader.dataset.update_step(0, 301)
    a = next(iter(loader))
    print(a)
