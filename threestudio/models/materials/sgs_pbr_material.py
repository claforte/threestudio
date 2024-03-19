import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.color_space import srgb_to_linear
from threestudio.utils.ops import (
    dot,
    get_activation,
    normalize,
    reflect,
    safe_exp,
    safe_sqrt,
)
from threestudio.utils.typing import Any, Dict, Float, Optional, Tensor, Tuple, Union


@threestudio.register("sgs-pbr-material")
class SGsPBRMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        sgs_init_color: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
        sgs_init_bw_color: Optional[float] = None
        num_sgs: int = 24
        num_illuminations: int = 1
        activation: str = "sigmoid"

        diffuse_only: bool = False
        use_metallic: bool = False
        base_reflectivity: float = 0.04

        use_residual_color: bool = False
        residual_dir_encoding_config: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3}
        )
        residual_features: int = 8
        residual_mlp_config: dict = field(
            default_factory=lambda: {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )
        residual_pre_activation_bias: float = -4.0
        residual_activation: str = "sigmoid"

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.requires_normal = True

        base_sgs = (
            torch.from_numpy(self.setup_uniform_axis_sharpness(self.cfg.num_sgs))
            .to(torch.float32)
            .unsqueeze(0)
            .repeat(self.cfg.num_illuminations, 1, 1)
        )
        if self.cfg.sgs_init_bw_color is not None:
            color = [self.cfg.sgs_init_bw_color]
        elif self.cfg.sgs_init_color is not None:
            color = self.cfg.sgs_init_color
        else:
            raise ValueError("Either sgs_init_color or sgs_init_bw_color must be set")
        sgs_amplitude = torch.tensor(color).to(torch.float32).view(1, 1, -1)
        base_amplitude = (
            torch.randn(
                self.cfg.num_illuminations, self.cfg.num_sgs, sgs_amplitude.shape[-1]
            )
            * 0.01  # Add small variations
            + sgs_amplitude
        )
        full_sgs = torch.cat([base_amplitude, base_sgs], -1)

        self.sgs = torch.nn.Parameter(full_sgs, requires_grad=True)

        self.residual_encoding = None
        self.residual_network = None
        if self.cfg.use_residual_color:
            self.residual_encoding = get_encoding(
                3, self.cfg.residual_dir_encoding_config
            )
            self.residual_net_input_dims = (
                self.cfg.residual_features + self.residual_encoding.n_output_dims
            )
            self.residual_network = get_mlp(
                self.residual_net_input_dims, 3, self.cfg.residual_mlp_config
            )

        if not self.tone_mapping.transforms_linear_color_space:
            threestudio.warn(
                "No color space conversion is applied. Make sure to select a color space conversion. "
                "Otherwise this won't be shaded in a linear color space."
            )

    def forward_impl(
        self,
        features: Float[Tensor, "B ... Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        illumination_idx: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        albedo = get_activation(self.cfg.activation)(features[..., :3])
        albedo_lin = albedo
        start_dim = 3
        if self.tone_mapping.transforms_linear_color_space:
            albedo_lin = srgb_to_linear(albedo)

        if self.cfg.diffuse_only:
            diffuse = albedo_lin
            specular = None
            roughness = None
        else:
            roughness = get_activation(self.cfg.activation)(features[..., 3:4])
            start_dim = 4
            specular = torch.full_like(albedo_lin, self.cfg.base_reflectivity)
            if self.cfg.use_metallic:
                metallic = get_activation(self.cfg.activation)(features[..., 4:5])
                start_dim = 5
                diffuse = albedo_lin * (1.0 - metallic)
                specular = specular * (1.0 - metallic) + albedo_lin * metallic

        if illumination_idx is not None and self.cfg.num_illuminations > 1:
            sgs = self.sgs[illumination_idx]
        else:
            sgs = self.sgs[0]

        material_eval = self._brdf_eval(
            sgs,
            shading_normal,
            viewdirs,
            diffuse,
            specular,
            roughness,
        )

        material_eval["albedo"] = albedo
        if not self.cfg.diffuse_only:
            material_eval["roughness"] = roughness
            if self.cfg.use_metallic:
                material_eval["metallic"] = metallic

        material_eval["color"] = self.tone_mapping.forward_impl(material_eval["color"])

        if self.cfg.use_residual_color:
            view_dirs_embd = self.residual_encoding(
                viewdirs.view(-1, 3) * 0.5 + 0.5
            )  # TCNN expects (-1, 1) -> (0, 1)
            residual_features = features[
                ..., start_dim : start_dim + self.cfg.residual_features
            ]
            network_inp = torch.cat(
                [
                    residual_features.view(-1, residual_features.shape[-1]),
                    view_dirs_embd,
                ],
                dim=-1,
            )
            residual_color = self.residual_network(network_inp).view(
                *residual_features.shape[:-1], 3
            )
            residual_color = get_activation(self.cfg.residual_activation)(
                residual_color + self.cfg.residual_pre_activation_bias
            )
            material_eval["color"] = material_eval["color"] + residual_color

            material_eval["residual"] = residual_color

        material_eval["color"] = self.tone_mapping.color_space(material_eval["color"])
        material_eval["illumination"] = self.tone_mapping(material_eval["illumination"])

        return material_eval

    def forward(self, *args, **kwargs) -> Float[Tensor, "*B 3"]:
        return self.forward_impl(*args, **kwargs)

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        albedo = get_activation(self.cfg.activation)(features[..., :3]).clamp(0.0, 1.0)
        ret = {"albedo": albedo}
        if not self.cfg.diffuse_only:
            roughness = get_activation(self.cfg.activation)(features[..., 3:4]).clamp(
                0.0, 1.0
            )
            ret["roughness"] = roughness
            if self.cfg.use_metallic:
                metallic = get_activation(self.cfg.activation)(
                    features[..., 4:5]
                ).clamp(0.0, 1.0)
                ret["metallic"] = metallic

        return ret

    def _brdf_eval(
        self,
        sg_illuminations: Float[Tensor, "N C"],
        shading_normal: Float[Tensor, "*B 3"],
        view_direction: Float[Tensor, "*B 3"],
        diffuse: Float[Tensor, "*B 3"],
        specular: Optional[Float[Tensor, "*B 3"]] = None,
        roughness: Optional[Float[Tensor, "*B 1"]] = None,
    ) -> Dict[str, Float[Tensor, "*B 3"]]:
        diffuse_eval = self._evaluate_diffuse(sg_illuminations, diffuse, shading_normal)
        total_shaded = diffuse_eval["diffuse_shaded"]
        total_illumination = diffuse_eval["diffuse_illumination"]

        base = diffuse_eval
        if not self.cfg.diffuse_only:
            ndf = self._distribution_term(shading_normal, roughness)

            warped_ndf = self._sg_warp_distribution(ndf, view_direction)
            _, warpDir, _ = self._extract_sg_components(warped_ndf)

            ndl = dot(shading_normal, warpDir).clip(0, 1)
            ndv = dot(shading_normal, view_direction).clip(0, 1)
            h = normalize(warpDir + view_direction)
            ldh = dot(warpDir, h).clip(0, 1)

            specular_eval = self._evaluate_specular(
                sg_illuminations, specular, roughness, warped_ndf, ndl, ndv, ldh
            )
            base.update(specular_eval)
            total_shaded = total_shaded + specular_eval["specular_shaded"]
            total_illumination = total_illumination + specular_eval["specular_shaded"]

        base["color"] = total_shaded
        base["illumination"] = total_illumination

        return base

    def _evaluate_diffuse(
        self,
        sg_illuminations: Float[Tensor, "N C"],
        diffuse: Float[Tensor, "B 3"],
        shading_normal: Float[Tensor, "B 3"],
    ) -> Dict[str, Tensor]:
        diff = diffuse
        norm = shading_normal.unsqueeze(1)

        _, s_axis, s_sharpness = [
            x.unsqueeze(0) for x in self._extract_sg_components(sg_illuminations)
        ]
        mudn: Float[Tensor, "B N 1"] = dot(s_axis, norm).clip(0, 1)

        c0 = 0.36
        c1 = 1.0 / (4.0 * c0)

        eml: Float[Tensor, "B N 1"] = safe_exp(-s_sharpness)
        em2l: Float[Tensor, "B N 1"] = eml * eml
        rl: Float[Tensor, "B N 1"] = torch.reciprocal(s_sharpness.clip(1e-4))

        scale = 1.0 + 2.0 * em2l - rl
        bias = (eml - em2l) * rl - em2l

        x = safe_sqrt(1.0 - scale)
        x0 = c0 * mudn
        x1 = c1 * x

        n = x0 + x1

        y_cond = x0.abs() <= x1
        y_true = n * (n / x.clip(1e-4))
        y_false = mudn
        y = torch.where(y_cond, y_true, y_false)

        res = scale * y + bias

        diffuse_illum: Float[Tensor, "B C"] = (
            res * self._sg_integral(sg_illuminations)
        ).sum(1) / torch.pi
        diffuse_shaded: Float[Tensor, "B 3"] = diff * diffuse_illum

        return {
            "diffuse_illumination": diffuse_illum,
            "diffuse_shaded": diffuse_shaded,
        }

    def _evaluate_specular(
        self,
        sg_illuminations: Float[Tensor, "N C"],
        specular: Float[Tensor, "B 3"],
        roughness: Float[Tensor, "B 1"],
        warped_ndf: Float[Tensor, "B C"],
        ndl: Float[Tensor, "B 1"],
        ndv: Float[Tensor, "B 1"],
        ldh: Float[Tensor, "B 1"],
    ) -> Dict[str, Tensor]:
        a2 = roughness.square().clip(1e-4)
        D: Float[Tensor, "B N C"] = self._sg_inner_product(
            warped_ndf.unsqueeze(1), sg_illuminations.unsqueeze(0)
        )

        G: Float[Tensor, "B 1"] = self._ggx(a2, ndl) * self._ggx(a2, ndv)

        powTerm: Float[Tensor, "B 1"] = (1.0 - ldh).pow(5)
        F: Float[Tensor, "B 3"] = specular + (1.0 - specular) * powTerm

        specular_shaded = D.sum(1) * G * ndl * F

        return {
            "specular_shaded": specular_shaded,
        }

    def _ggx(
        self, a2: Float[Tensor, "B 1"], ndx: Float[Tensor, "B 1"]
    ) -> Float[Tensor, "B 1"]:
        return torch.reciprocal((ndx + safe_sqrt(a2 + (1 - a2) * ndx * ndx)).clip(1e-4))

    def _distribution_term(
        self, d: Float[Tensor, "B 3"], roughness: Float[Tensor, "B 1"]
    ) -> Float[Tensor, "B 5"]:
        a2 = roughness.square().clip(1e-4)

        ret = self._stack_sg_components(
            torch.reciprocal(torch.pi * a2),
            d,
            2.0 / a2,
        )
        return ret

    def _sg_warp_distribution(
        self, ndfs: Float[Tensor, "B 5"], view_direction: Float[Tensor, "B 3"]
    ) -> Float[Tensor, "B 5"]:
        ndf_amplitude, ndf_axis, ndf_sharpness = self._extract_sg_components(ndfs)

        ret = torch.cat(
            [
                ndf_amplitude,
                reflect(-view_direction, ndf_axis),
                ndf_sharpness / (4.0 * dot(ndf_axis, view_direction).clip(1e-4)),
            ],
            -1,
        )

        return ret

    def _extract_sg_components(
        self, sg: Float[Tensor, "*N C+3+1"]
    ) -> Tuple[Float[Tensor, "*N C"], Float[Tensor, "*N 3"], Float[Tensor, "*N 1"]]:
        s_amplitude = sg[..., :-4]
        s_axis = sg[..., -4:-1]
        s_sharpness = sg[..., -1:]

        return (
            s_amplitude.abs(),
            normalize(s_axis),
            s_sharpness.clip(1e-4),
        )

    def _stack_sg_components(
        self,
        amplitude: Float[Tensor, "*N C"],
        axis: Float[Tensor, "*N 3"],
        sharpness: Float[Tensor, "*N 1"],
    ) -> Float[Tensor, "*N C+3+1"]:
        return torch.cat(
            [
                amplitude,
                axis,
                sharpness,
            ],
            -1,
        )

    def _sg_integral(self, sg: Float[Tensor, "*N C+3+1"]) -> Float[Tensor, "*N C"]:
        s_amplitude, _, s_sharpness = self._extract_sg_components(sg)

        expTerm = 1.0 - safe_exp(-2.0 * s_sharpness)
        return 2 * torch.pi * s_amplitude / s_sharpness * expTerm

    def _sg_inner_product(
        self, sg1: Float[Tensor, "*N C+3+1"], sg2: Float[Tensor, "*N C+3+1"]
    ) -> Float[Tensor, "*N C+3+1"]:
        s1_amplitude, s1_axis, s1_sharpness = self._extract_sg_components(sg1)
        s2_amplitude, s2_axis, s2_sharpness = self._extract_sg_components(sg2)

        umLength = torch.norm(
            s1_sharpness * s1_axis + s2_sharpness * s2_axis, dim=-1, keepdim=True
        )
        expo = (
            safe_exp(umLength - s1_sharpness - s2_sharpness)
            * s1_amplitude
            * s2_amplitude
        )

        other = 1.0 - safe_exp(-2.0 * umLength)

        return (2.0 * torch.pi * expo * other) / umLength

    def _sg_evaluate(
        self, sg: Float[Tensor, "*N C+3+1"], direction: Float[Tensor, "*B 3"]
    ) -> Float[Tensor, "... C"]:
        s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)
        s_amplitude_flat, s_axis_flat, s_sharpness_flat = (
            s_amplitude.flatten(0, -2),
            s_axis.flatten(0, -2),
            s_sharpness.flatten(0, -2),
        )
        dir_flat: Float[Tensor, "S 3"] = direction.flatten(0, -2)

        cosAngle = dot(dir_flat.unsqueeze(1), s_axis.unsqueeze(0))
        evaled = s_amplitude_flat.unsqueeze(0) * safe_exp(
            s_sharpness_flat.unsqueeze(0) * (cosAngle - 1.0)
        )

        return evaled.view(*direction.shape[:-1], *s_amplitude.shape)

    @classmethod
    def setup_uniform_axis_sharpness(cls, num_sgs) -> np.ndarray:
        def dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.sum(x * y, axis=-1, keepdims=True)

        def magnitude(x: np.ndarray) -> np.ndarray:
            return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1e-12))

        def normalize(x: np.ndarray) -> np.ndarray:
            return x / magnitude(x)

        axis = []
        inc = np.pi * (3.0 - np.sqrt(5.0))
        off = 2.0 / num_sgs
        for k in range(num_sgs):
            y = k * off - 1.0 + (off / 2.0)
            r = np.sqrt(1.0 - y * y)
            phi = k * inc
            axis.append(normalize(np.array([np.cos(phi) * r, np.sin(phi) * r, y])))

        minDp = 1.0
        for a in axis:
            h = normalize(a + axis[0])
            minDp = min(minDp, dot(h, axis[0]))

        sharpness = (np.log(0.65) * num_sgs) / (minDp - 1.0)

        axis = np.stack(axis, 0)  # Shape: num_sgs, 3
        sharpnessNp = np.ones((num_sgs, 1)) * sharpness
        return np.concatenate([axis, sharpnessNp], -1)
