from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.ops import smoothstep
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.ssao import calculate_ssao
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

        perform_ssao: bool = False
        ssao_max_distance: float = 0.25
        ssao_intensity: float = 1.0
        ssao_samples: int = 8
        ssao_sample_rad: float = 0.02

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        render_vis: bool = False,
        # mesh_invis_cache: Float[Tensor, "V 3"] = None,
        ref_cam_pos: Float[Tensor, "B 3"] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
        gb_pos_aa = torch.lerp(torch.zeros_like(gb_pos), gb_pos, mask.float())
        gb_pos_aa = self.ctx.antialias(gb_pos_aa, rast, v_pos_clip, mesh.t_pos_idx)

        depth = gb_pos - camera_positions[:, None, None, :]

        out.update({"positions": gb_pos_aa, "depth": depth})

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_vis:
            with torch.no_grad():
                view_directions: Float[Tensor, "B V 3"] = F.normalize(
                    ref_cam_pos.unsqueeze(1) - mesh.v_pos.unsqueeze(0), dim=-1
                )
                mesh_normals: Float[Tensor, "V 3"] = F.normalize(mesh.v_nrm, dim=-1)
                dot_prod: Float[Tensor, "B V"] = (
                    mesh_normals.unsqueeze(0) * view_directions
                ).sum(-1)
                max_dp: Float[Tensor, "V"] = smoothstep(
                    dot_prod.max(dim=0).values, 0.1, 0.5
                )

                # non_visible: Float[Tensor, "V"] = (n_dot_c < 0.2).all(dim=0).float()
                # visibility: Float[Tensor, "V"] = dot_prod.max(dim=0)[0].float().clip(0.0, 1.0)
                visibility: Float[Tensor, "V"] = max_dp.clip(0.0, 1.0)
                # non_visible = non_visible[:,None].repeat(1,3)
                visibility = visibility[:, None].repeat(1, 3)

                # gb_invis, _ = self.ctx.interpolate_one(non_visible, rast, mesh.t_pos_idx)
                # gb_invis_aa = torch.lerp(torch.zeros_like(gb_pos), gb_invis, mask.float())
                # gb_invis_aa = self.ctx.antialias(
                #     gb_invis, rast, v_pos_clip, mesh.t_pos_idx
                # )
                # out.update({"comp_vis": 1 - gb_invis_aa, "mesh_invis_cache": mesh_invis_cache})
                gb_vis, _ = self.ctx.interpolate_one(visibility, rast, mesh.t_pos_idx)
                gb_vis_aa = torch.lerp(torch.zeros_like(gb_pos), gb_vis, mask.float())
                gb_vis_aa = self.ctx.antialias(gb_vis, rast, v_pos_clip, mesh.t_pos_idx)
                out.update({"comp_vis": gb_vis_aa})
        else:
            out.update({"comp_vis": torch.zeros_like(gb_normal_aa)})

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            material_eval = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out,
            )
            material_out = {}
            if isinstance(material_eval, dict):

                def fill_material(mat):
                    base = torch.zeros(
                        batch_size,
                        height,
                        width,
                        mat.shape[-1],
                        dtype=mat.dtype,
                        device=mat.device,
                    )
                    base[selector] = mat
                    return base

                material_eval_filled = {
                    k: fill_material(v) for k, v in material_eval.items()
                }
                gb_rgb_fg = material_eval_filled["color"]
                material_out.update(
                    {
                        f"comp_{k}": v
                        for k, v in material_eval_filled.items()
                        if k != "color"
                    }
                )
            else:
                gb_rgb_fg = torch.zeros(
                    batch_size,
                    height,
                    width,
                    3,
                    dtype=material_eval.dtype,
                    device=material_eval.device,
                )
                gb_rgb_fg[selector] = material_eval

            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update(
                {"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg, **material_out}
            )

        return out
