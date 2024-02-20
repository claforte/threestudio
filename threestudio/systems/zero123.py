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
from threestudio.utils.misc import C, get_CPU_mem, get_GPU_mem
from threestudio.utils.ops import binary_cross_entropy, dot
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
        train_on_drunk: bool = False
        disable_grid_prune_step: int = 2000

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        if len(self.cfg.guidance_type):
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.device = torch.device("cuda:1")
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
                    1
                    - 2
                    * torch.cat(
                        [
                            batch["ref_normal"][
                                :, xx::rays_divisor, yy::rays_divisor, :
                            ]
                            for (xx, yy) in zip(offset_x_tensor, offset_y_tensor)
                        ]
                    )[gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )

            self.log("train/mem_cpu", get_CPU_mem(), prog_bar=True)
            self.log("train/mem_gpu", get_GPU_mem()[0], prog_bar=True)

        elif guidance == "zero123":
            # zero123
            guidance_out = self.guidance(
                out["comp_rgb"],
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

        if self.cfg.geometry.pos_encoding_config.otype == "KPlanes":
            if self.C(self.cfg.loss.lambda_total_variation) > 0:
                set_loss("total_variation", self.geometry.encoding.encoding.loss_tv())
            if self.C(self.cfg.loss.lambda_l1_regularization) > 0:
                set_loss("l1_regularization", self.geometry.encoding.encoding.loss_l1())

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

        if guidance == "ref":
            out = self(batch)

        elif guidance == "zero123":
            # self.renderer.to(self.guidance.device)
            # for k, v in batch.items():
            #     if torch.is_tensor(v):
            #         batch[k] = v.to(self.guidance.device)

            B, H, W = batch["rays_o"].shape[:3]
            elevation_all = batch["elevation"].clone()
            azimuth_all = batch["azimuth"].clone()
            for k, v in batch.items():
                if k != "frame_idx" and torch.is_tensor(v) and v.shape[0] == B:
                    batch[k] = batch[k][batch["frame_idx"]]

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
            "/weka/proj-sv3d/DATASETS/GSO_sober21", self.cfg.obj_name, "rgba", "*.png"
        )
        gt_drunk_pattern = os.path.join(
            "/weka/proj-sv3d/DATASETS/GSO_drunk21", self.cfg.obj_name, "rgba", "*.png"
        )
        if self.cfg.train_on_drunk:
            svd_pattern = os.path.join(
                "/weka/home-chunhanyao/GSO_drunk21", self.cfg.obj_name, "*.png"
            )
            svd_label = "SVD_drunk"
        else:
            svd_pattern = os.path.join(
                "/weka/home-chunhanyao/GSO_sober21", self.cfg.obj_name, "*.png"
            )
            svd_label = "SVD_sober"

        imgs_gt_sober = [cv2.imread(f, -1) for f in sorted(glob.glob(gt_sober_pattern))]
        imgs_gt_drunk = [cv2.imread(f, -1) for f in sorted(glob.glob(gt_drunk_pattern))]
        imgs_nerf_sober = [cv2.imread(f) for f in sorted(glob.glob(nerf_sober_pattern))]
        imgs_nerf_drunk = [cv2.imread(f) for f in sorted(glob.glob(nerf_drunk_pattern))]
        imgs_nerf_random = [
            cv2.imread(f) for f in sorted(glob.glob(nerf_random_pattern))
        ]
        imgs_svd = [cv2.imread(f) for f in sorted(glob.glob(svd_pattern))]

        img_grids = []
        for im1, im2, im3, im4, im5, im6 in zip(
            imgs_gt_drunk,
            imgs_svd,
            imgs_nerf_drunk,
            imgs_gt_sober,
            imgs_nerf_sober,
            imgs_nerf_random,
        ):
            im1[im1[..., 3] == 0, :3] = 255
            im4[im4[..., 3] == 0, :3] = 255
            img_grid = np.vstack(
                (
                    np.hstack((im1[..., :3], im2, im3)),
                    np.hstack((im4[..., :3], im5, im6)),
                )
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="GT_drunk",
                org=(30, 50),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text=svd_label,
                org=(606, 50),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="NeRF (drunk renders)",
                org=(1182, 50),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="GT_sober",
                org=(30, 606),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="NeRF (sober renders)",
                org=(606, 606),
                fontFace=2,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
            )
            img_grid = cv2.putText(
                img=np.copy(img_grid),
                text="NeRF (random renders)",
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
