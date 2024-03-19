import bisect
import glob
import json
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import RandomCameraDataset, gen_drunk_loop, gen_elev_loop
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
class SVDCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 576
    eval_width: int = 576
    eval_batch_size: int = 1
    n_train_views: int = 21
    n_val_views: int = 21
    n_test_views: int = 21
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 5.0
    eval_camera_distance: float = 2.0
    eval_fovy_deg: float = 33.9
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy

    cond_img_path: str = None
    cond_elevation_deg: float = 0.0
    cond_azimuth_deg: float = 0.0
    cond_camera_distance: float = 2.0
    use_random_orbit: bool = False


class SVDCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: SVDCameraDataModuleConfig = cfg

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

        self.n_views = self.cfg.n_train_views

        # orbit_file = sorted(
        #     glob.glob(os.path.join(self.cfg.cond_img_path, "frame_*.json"))
        # )[20]
        # transforms = json.load(open(orbit_file, "r"))
        # self.cond_elevation_deg = 90 - transforms["polar"] * 180 / math.pi
        # self.cond_azimuth_deg = transforms["azimuth"] * 180 / math.pi
        # self.cond_camera_distance = transforms["camera_dist"]

        self.cond_elevation_deg = self.cfg.cond_elevation_deg
        self.cond_camera_distance = 2.0

        azimuth_deg: Float[Tensor, "B"] = torch.linspace(0, 360.0, self.n_views + 1)[1:]
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cond_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cond_camera_distance  # self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.n_views, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        self.fovy = fovy_deg * math.pi / 180
        # light_positions: Float[Tensor, "B 3"] = camera_positions
        # light position is always at front camera
        light_positions: Float[Tensor, "B 3"] = camera_positions[-1].repeat(
            len(camera_positions), 1
        )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.height / torch.tan(0.5 * self.fovy)
        )
        # directions_unit_focal = get_ray_directions(
        #     H=self.height, W=self.width, focal=1.0
        # )
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def update_step_legacy(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.height / torch.tan(0.5 * self.fovy)
        )
        # directions_unit_focal = get_ray_directions(
        #     H=self.height, W=self.width, focal=1.0
        # )
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, self.c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.height / torch.tan(0.5 * self.fovy)
        )
        # directions_unit_focal = get_ray_directions(
        #     H=self.height, W=self.width, focal=1.0
        # )
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        if self.cfg.use_random_orbit:
            azimuth, elevation = gen_drunk_loop(
                length=self.n_views,
                elev_deg=self.cond_elevation_deg,
            )
            # azimuth, elevation = gen_elev_loop(
            #     length=21,
            #     elev=self.cond_elevation_deg*np.pi/180,
            #     # elev_deg=self.cond_elevation_deg,
            # )
            azimuth = azimuth[-self.n_views :]
            elevation = elevation[-self.n_views :]
            azimuth = torch.from_numpy(azimuth).to(torch.float32)
            elevation = torch.from_numpy(elevation).to(torch.float32)

            elevation_deg = elevation * 180 / math.pi
            azimuth_deg = azimuth * 180 / math.pi
            # print(self.cond_elevation_deg, elevation_deg)

            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, x back, y right, z up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    self.camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    self.camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    self.camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )

            # default scene center at origin
            center = torch.zeros_like(camera_positions)
            # default camera up direction as +z
            up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(
                self.n_views, 1
            )

            # light position is always at front camera
            light_positions = camera_positions[-1].repeat(len(camera_positions), 1)

            lookat: Float[Tensor, "B 3"] = F.normalize(
                center - camera_positions, dim=-1
            )
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [
                    torch.stack([right, up, -lookat], dim=-1),
                    camera_positions[:, :, None],
                ],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0

            self.c2w = c2w
            self.camera_positions = camera_positions
            self.light_positions = light_positions
            self.elevation, self.azimuth = elevation, azimuth
            self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg

        rays_o, rays_d = get_rays(directions, self.c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        idx = torch.randperm(self.n_views)[: self.batch_size]
        idx, _ = torch.sort(idx)
        # idx = torch.randperm(self.n_views-1)[:self.batch_size-1]
        # idx = torch.cat([torch.zeros_like(idx[:1])+self.n_views-1, idx], 0)
        return {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_positions,
            "c2w": self.c2w,
            "light_positions": self.light_positions,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distances,
            "frame_idx": idx,
            "height": self.height,
            "width": self.width,
        }


@register("svd-camera-datamodule")
class SVDCameraDataModule(pl.LightningDataModule):
    cfg: SVDCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SVDCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SVDCameraIterableDataset(self.cfg)
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

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
