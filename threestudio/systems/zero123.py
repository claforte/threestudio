import glob
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from math import ceil

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.img_gradient import compute_image_gradients
from threestudio.utils.misc import C, get_CPU_mem, get_GPU_mem
from threestudio.utils.ops import binary_cross_entropy, dot, normalize
from threestudio.utils.typing import *


@threestudio.register("zero123-system")
class Zero123(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        rays_divisor_power: int = 0
        ref_batch_size: int = 1
        obj_name: str = None
        dataroot: str = None
        disable_grid_prune_step: int = 2000

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        if len(self.cfg.guidance_type):
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.device = torch.device("cuda:0")
            self.comp_rgb_cache = None

        if self.cfg.loss.lambda_lpips != 0:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True
            )
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor

        # visualize all training images
        try:
            all_images = (
                self.trainer.datamodule.train_dataloader().dataset.get_all_images()
            )
            self.save_image_grid(
                "all_training_images.png",
                [
                    {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                    for image in all_images
                ],
                name="on_fit_start",
                step=self.true_global_step,
            )
        except Exception as e:
            pass

        self.pearson = PearsonCorrCoef().to(self.device)

    def compute_loss(
        self, out, guidance, batch, rays_divisor, offset_x_tensor, offset_y_tensor
    ):
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "zero123"
            and getattr(self.cfg.freq, "guidance_eval", 0) > 0
            and self.true_global_step % getattr(self.cfg.freq, "guidance_eval", 0) == 0
        )

        if guidance == "ref":
            gt_mask = torch.cat(
                [
                    batch["mask"][:, xx::rays_divisor, yy::rays_divisor, :]
                    for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                ]
            )
            gt_rgb = torch.cat(
                [
                    batch["rgb"][:, xx::rays_divisor, yy::rays_divisor, :]
                    for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                ]
            )

            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                1 - gt_mask.float()
            )
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))

            if self.C(self.cfg.loss.lambda_lpips) > 0:
                # lpips expects images in range [-1,1] and (N,3,H,W)
                set_loss(
                    "lpips",
                    self.lpips(
                        gt_rgb.permute(0, 3, 1, 2),
                        out["comp_rgb"].clamp(0, 1).permute(0, 3, 1, 2),
                    ),
                )

            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = torch.cat(
                    [
                        batch["ref_depth"][:, xx::rays_divisor, yy::rays_divisor, :]
                        for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                    ]
                )[gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = torch.cat(
                    [
                        batch["ref_depth"][:, xx::rays_divisor, yy::rays_divisor, :]
                        for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                    ]
                )[gt_mask.squeeze(-1)]
                valid_pred_depth = out["depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    torch.cat(
                        [
                            batch["ref_normal"][
                                :, xx::rays_divisor, yy::rays_divisor, :
                            ]
                            for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                        ]
                    )[gt_mask.squeeze(-1)]
                    * 2
                    - 1
                )  # [B, 3]

                valid_pred_normal = (
                    out["comp_normal"][gt_mask.squeeze(-1)] * 2 - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )

            # Smoothness loss
            bilateral_smoothness_enabled = any(
                [
                    k.startswith("lambda_bilateral_smoothness_")
                    for k in self.cfg.loss.keys()
                ]
            )
            if bilateral_smoothness_enabled:
                grad = sum([t.abs() for t in compute_image_gradients(gt_rgb)])
                grad_scaler = (-grad * 3).exp().detach() * gt_mask.float()
                for k in self.cfg.loss.keys():
                    if k.startswith("lambda_bilateral_smoothness_"):
                        if self.C(self.cfg.loss[k]) > 0:
                            smoothness_type = k.replace(
                                "lambda_bilateral_smoothness_", ""
                            )
                            if smoothness_type in out:
                                val = out[smoothness_type]
                                val_grad = compute_image_gradients(val)
                                val_norm = [
                                    t.norm(dim=-1, keepdim=True) for t in val_grad
                                ]
                                val_loss = (
                                    sum([(1 + x).sqrt() - 1 for x in val_norm])
                                    * grad_scaler
                                )[gt_mask.squeeze(-1)]
                                set_loss(
                                    f"bilateral_smoothness_{smoothness_type}",
                                    val_loss.mean(),
                                )

            if "comp_illumination" in out:
                if self.C(self.cfg.loss.lambda_light_demodulation) > 0:
                    luminance_illumination = out["comp_illumination"].mean(
                        dim=-1, keepdim=True
                    )
                    value_rgb = gt_rgb.max(dim=-1, keepdim=True).values
                    light_demodulation = (
                        (luminance_illumination - value_rgb)
                        .abs()[gt_mask.squeeze(-1)]
                        .mean()
                    )
                    set_loss("light_demodulation", light_demodulation)

            self.log("train/mem_cpu", get_CPU_mem(), prog_bar=True)
            self.log("train/mem_gpu", get_GPU_mem()[0], prog_bar=True)

        elif guidance == "zero123":
            # zero123

            # cam_pos: Float[Tensor, "B 3"] = batch["ref_cam_pos"]
            # mesh_positions: Float[Tensor, "V 3"] = out["mesh"].v_pos
            # view_directions: Float[Tensor, "B V 3"] = normalize(
            #     cam_pos.unsqueeze(1) - mesh_positions.unsqueeze(0)
            # )

            # mesh_normals: Float[Tensor, "V 3"] = normalize(out["mesh"].v_nrm)
            # n_dot_c: Float[Tensor, "B V"] = (
            #     mesh_normals.unsqueeze(0) * view_directions
            # ).sum(-1)
            # non_visible: Float[Tensor, "V"] = (n_dot_c < 0.1).all(dim=0)

            guidance_out = self.guidance(
                out["comp_rgb"],
                out["comp_vis"],
                # **batch,
                batch["elevation"],
                batch["azimuth"],
                batch["frame_idx"],
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
            # claforte: TODO: rename the loss_terms keys
            set_loss("sds", guidance_out["loss_sds"])

            self.log("train/mem_cpu", guidance_out["cpu_mem"], prog_bar=True)
            self.log("train/mem_gpu", guidance_out["gpu_mem"], prog_bar=True)

        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        pred_mask = out["opacity"] > 0.5
        if "comp_albedo" in out:
            if self.C(self.cfg.loss.lambda_albedo_range) > 0:
                # Albedo should be larger than 0.15 and small 0.95
                loss = (
                    (0.15 - out["comp_albedo"]).clip(0).square()
                    + (out["comp_albedo"] - 0.95).clip(0).square()
                )[pred_mask.detach().squeeze(-1)].mean()
                set_loss("albedo_range", loss)

        if "comp_roughness" in out:
            if self.C(self.cfg.loss.lambda_lambertian) > 0:
                lambertian_loss = (
                    (1 - out["comp_roughness"])
                    .square()[pred_mask.detach().squeeze(-1)]
                    .mean()
                )
                if "comp_metallic" in out:
                    lambertian_loss = (
                        lambertian_loss
                        + (out["comp_metallic"])
                        .square()[pred_mask.detach().squeeze(-1)]
                        .mean()
                    )
                set_loss("lambertian", lambertian_loss)

        if "comp_residual" in out:
            if self.C(self.cfg.loss.lambda_residual) > 0:
                set_loss("residual", out["comp_residual"].square().mean())

        if self.cfg.geometry.pos_encoding_config.otype == "KPlanes":
            if self.C(self.cfg.loss.lambda_total_variation) > 0:
                set_loss("total_variation", self.geometry.encoding.encoding.loss_tv())
            if self.C(self.cfg.loss.lambda_l1_regularization) > 0:
                set_loss("l1_regularization", self.geometry.encoding.encoding.loss_l1())

        smoothness_3d_enabled = any(
            [k.startswith("lambda_3d_smoothness_") for k in self.cfg.loss.keys()]
        )
        if smoothness_3d_enabled:
            gt_rays_d = torch.cat(
                [
                    batch["rays_d"][:, xx::rays_divisor, yy::rays_divisor, :]
                    for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                ]
            )
            gt_rays_o = torch.cat(
                [
                    batch["rays_o"][:, xx::rays_divisor, yy::rays_divisor, :]
                    for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                ]
            )
            surface_pos = out["depth"] * gt_rays_d + gt_rays_o
            binary_mask = (out["opacity"] > 0.5).squeeze(-1)
            selected_pos = surface_pos[binary_mask]
            sample_delta = self.cfg.loss.loss_3d_smoothness_delta
            random_offset = torch.randn_like(selected_pos) * sample_delta

            main_out = self.renderer.geometry(selected_pos)
            offset_out = self.renderer.geometry(selected_pos + random_offset)
            main_out.update(self.renderer.material.export(**main_out))
            offset_out.update(self.renderer.material.export(**offset_out))
            for k in self.cfg.loss.keys():
                if k.startswith("lambda_3d_smoothness_"):
                    if self.C(self.cfg.loss[k]) > 0:
                        smoothness_type = k.replace("lambda_3d_smoothness_", "")
                        if (
                            smoothness_type in main_out
                            and smoothness_type in offset_out
                        ):
                            val_loss = (
                                main_out[smoothness_type] - offset_out[smoothness_type]
                            ).square()
                            set_loss(
                                f"3d_smoothness_{smoothness_type}", val_loss.mean()
                            )

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )
        else:
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )

        return {"loss": loss}

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        ref_cam_pos = batch["camera_positions"]
        if self.ref_cam_pos is None or self.ref_cam_pos.shape[0] < ref_cam_pos.shape[0]:
            self.ref_cam_pos = ref_cam_pos

        if guidance == "ref":
            # bg_color = torch.rand_like(batch['rays_o'])
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
            rays_divisor = 2 ** ceil(self.C(self.cfg.rays_divisor_power))
            offset_x_tensor = torch.randint(0, rays_divisor, (self.cfg.ref_batch_size,))
            offset_y_tensor = torch.randint(0, rays_divisor, (self.cfg.ref_batch_size,))
        elif guidance == "zero123":
            batch = batch["random_camera"]
            batch["ref_cam_pos"] = self.ref_cam_pos
            batch["render_vis"] = True
            # ambient_ratio = (
            #     self.C(self.cfg.ambient_ratio_min)
            #     + (1 - self.C(self.cfg.ambient_ratio_min)) * random.random()
            # )
            ambient_ratio = self.C(self.cfg.ambient_ratio_min)
            # rays_divisor = 1
            # offset_x_tensor = torch.zeros(1, dtype=torch.int64)
            # offset_y_tensor = torch.zeros(1, dtype=torch.int64)
            rays_divisor = 2 ** ceil(self.C(self.cfg.rays_divisor_power))
            offset_x_tensor = torch.randint(0, rays_divisor, (self.cfg.ref_batch_size,))
            offset_y_tensor = torch.randint(0, rays_divisor, (self.cfg.ref_batch_size,))

        batch["rays_divisor"] = rays_divisor
        batch["offset_x"] = offset_x_tensor
        batch["offset_y"] = offset_y_tensor

        batch["bg_color"] = None
        batch["ambient_ratio"] = ambient_ratio

        # import pdb; pdb.set_trace()

        if guidance == "ref":
            out = self(batch)

        elif guidance == "zero123":
            # self.renderer.to(self.guidance.device)
            # for k, v in batch.items():
            #     if torch.is_tensor(v):
            #         batch[k] = v.to(self.guidance.device)

            # B, H, W = batch["rays_o"].shape[:3]
            # elevation_all = batch["elevation"].clone()
            # azimuth_all = batch["azimuth"].clone()
            # for k, v in batch.items():
            #     if k != "frame_idx" and torch.is_tensor(v) and v.shape[0] == B:
            #         batch[k] = batch[k][batch["frame_idx"]]

            out = self(batch)

            # comp_rgb_cache = torch.ones(B, H, W, 3).float().to(self.device)
            # comp_rgb_cache[batch["frame_idx"]] = out["comp_rgb"]
            # out["comp_rgb"] = comp_rgb_cache
            # batch['elevation'] = elevation_all
            # batch['azimuth'] = azimuth_all

            # for k, v in out.items():
            #     if torch.is_tensor(v):
            #         out[k] = v.to(self.device)
            # for k, v in batch.items():
            #     if torch.is_tensor(v):
            #         batch[k] = v.to(self.device)
            # self.renderer.to(self.device)

        return self.compute_loss(
            out, guidance, batch, rays_divisor, offset_x_tensor, offset_y_tensor
        )

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        if batch_idx == 0:
            self.ref_cam_pos = None

        if batch_idx > self.cfg.disable_grid_prune_step:
            self.renderer.cfg.grid_prune = False

        # ZERO123
        if (
            self.cfg.guidance_type == "svd-guidance"
            and self.C(self.cfg.loss.lambda_sds) > 0
        ):
            out = self.training_substep(batch, batch_idx, guidance="zero123")
            total_loss += out["loss"]

        # REF
        out = self.training_substep(batch, batch_idx, guidance="ref")
        total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            # (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": batch["rgb"][0],
            #             "kwargs": {"data_format": "HWC"},
            #         }
            #     ]
            #     if "rgb" in batch
            #     else []
            # )
            # +
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name=None,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="validation_epoch_end",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        )

    # def test_step(self, batch, batch_idx):
    #     out = self(batch)
    #     self.save_image_grid(
    #         f"it{self.true_global_step}-test/{batch['index'][0]}.png",
    #         [
    #             {
    #                 "type": "rgb",
    #                 "img": out["comp_rgb"][0],
    #                 "kwargs": {"data_format": "HWC"},
    #             },
    #         ]
    #         + (
    #             [
    #                 {
    #                     "type": "rgb",
    #                     "img": out["comp_normal"][0],
    #                     "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
    #                 }
    #             ]
    #             if "comp_normal" in out
    #             else []
    #         )
    #         + (
    #             [
    #                 {
    #                     "type": "grayscale",
    #                     "img": out["depth"][0],
    #                     "kwargs": {},
    #                 }
    #             ]
    #             if "depth" in out
    #             else []
    #         )
    #         + [
    #             {
    #                 "type": "grayscale",
    #                 "img": out["opacity"][0, :, :, 0],
    #                 "kwargs": {"cmap": None, "data_range": (0, 1)},
    #             },
    #         ],
    #         name="test_step",
    #         step=self.true_global_step,
    #     )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        if batch["index"][0] < 21:
            self.save_image_grid(
                f"sober/{batch['index'][0]:0>4}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="test_step",
            )
        elif batch["index"][0] < 42:
            self.save_image_grid(
                f"drunk/{batch['index'][0]-21:0>4}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="test_step",
            )
        else:
            self.save_image_grid(
                f"random/{batch['index'][0]-42:0>4}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="test_step",
            )

    # def on_test_epoch_end(self):
    #     self.save_img_sequence(
    #         f"it{self.true_global_step}-test",
    #         f"it{self.true_global_step}-test",
    #         "(\d+)\.png",
    #         save_format="mp4",
    #         fps=30,
    #         name="test",
    #         step=self.true_global_step,
    #     )
    #     shutil.rmtree(
    #         os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test")
    #     )

    def on_test_epoch_end(self):
        self.save_img_grid_sequence(
            f"test",
            save_format="mp4",
            fps=10,
        )
        # self.save_img_sequence(
        #     f"sober",
        #     f"sober",
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=10,
        #     name="test",
        # )
        # self.save_img_sequence(
        #     f"drunk",
        #     f"drunk",
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=10,
        #     name="test",
        # )

    def save_img_grid_sequence(
        self,
        filename,
        save_format="mp4",
        fps=30,
    ) -> str:
        assert save_format in ["gif", "mp4"]
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        save_path = self.get_save_path(filename)
        nerf_sober_pattern = os.path.join(self.get_save_dir(), "sober", "*.png")
        nerf_drunk_pattern = os.path.join(self.get_save_dir(), "drunk", "*.png")
        nerf_random_pattern = os.path.join(self.get_save_dir(), "random", "*.png")
        gt_sober_pattern = os.path.join(
            "/weka/proj-sv3d/DATASETS/OmniObject3D_sober21",
            self.cfg.obj_name,
            "rgba",
            "*.png",
        )
        gt_drunk_pattern = os.path.join(
            "/weka/proj-sv3d/DATASETS/OmniObject3D_drunk21",
            self.cfg.obj_name,
            "rgba",
            "*.png",
        )
        svd_pattern = os.path.join(self.cfg.dataroot, "*.png")
        svd_label = "SV3D NVS"
        if (
            len(glob.glob(gt_sober_pattern)) == 0
            or len(glob.glob(gt_drunk_pattern)) == 0
        ):
            gt_sober_pattern = svd_pattern
            gt_drunk_pattern = svd_pattern

        imgs_gt_sober = [
            cv2.imread(f, -1) for f in sorted(glob.glob(gt_sober_pattern))[:21]
        ]
        imgs_gt_drunk = [
            cv2.imread(f, -1) for f in sorted(glob.glob(gt_drunk_pattern))[:21]
        ]
        imgs_nerf_sober = [cv2.imread(f) for f in sorted(glob.glob(nerf_sober_pattern))]
        imgs_nerf_drunk = [cv2.imread(f) for f in sorted(glob.glob(nerf_drunk_pattern))]
        imgs_nerf_random = [
            cv2.imread(f) for f in sorted(glob.glob(nerf_random_pattern))
        ]
        imgs_svd = [cv2.imread(f) for f in sorted(glob.glob(svd_pattern))]

        img_grids = []
        for im1, im2, im3, im4, im5, im6 in zip(
            imgs_gt_sober,
            imgs_gt_drunk,
            imgs_svd,
            imgs_nerf_sober,
            imgs_nerf_drunk,
            imgs_nerf_random,
        ):
            if im1.shape[-1] == 4:
                im1[im1[..., 3] == 0, :3] = 255
                im2[im2[..., 3] == 0, :3] = 255
                im1 = im1[..., :3]
                im2 = im2[..., :3]
            im1 = cv2.resize(im1, dsize=im4.shape[:2])
            im2 = cv2.resize(im2, dsize=im4.shape[:2])
            im3 = cv2.resize(im3, dsize=im4.shape[:2])
            img_grid = np.vstack(
                (
                    np.hstack((im1, im2, im3)),
                    np.hstack((im4, im5, im6)),
                )
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="GT_sober",
                org=(30, 50),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="GT_drunk",
                org=(606, 50),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text=svd_label,
                org=(1182, 50),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="3D (sober renders)",
                org=(30, 606),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="3D (drunk renders)",
                org=(606, 606),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="3D (random renders)",
                org=(1182, 606),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grids.append(img_grid)

        if save_format == "gif":
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in img_grids]
            imageio.mimsave(save_path, imgs, fps=fps, palettesize=256)
        elif save_format == "mp4":
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in img_grids]
            imageio.mimsave(save_path, imgs, fps=fps)
        return save_path
