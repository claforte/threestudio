import torch
import torch.nn.functional as F

from .typing import Float, Tensor


def access_coord(
    tex: Float[Tensor, "1 C H W"], coord: Float[Tensor, "*N 2"]
) -> Float[Tensor, "*N C"]:
    shaped_coord = coord.reshape(tex.shape[0], -1, 1, 2)
    sample = F.grid_sample(tex, shaped_coord * 2 - 1, align_corners=False)
    return sample.squeeze(-1).permute(0, 2, 1).view(*coord.shape[:-1], tex.shape[1])


def dot(
    a: Float[torch.Tensor, "*B 3"], b: Float[torch.Tensor, "*B 3"]
) -> Float[torch.Tensor, "*B 1"]:
    return (a * b).sum(dim=-1, keepdim=True)


def smoothstep(
    edge0: Float[Tensor, "*B 1"], edge1: Float[Tensor, "*B 1"], x: Float[Tensor, "*B 1"]
) -> Float[Tensor, "*B 1"]:
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


MOD3 = torch.tensor([0.1031, 0.11369, 0.13787], dtype=torch.float32)


@torch.no_grad()
def hash12(p: Float[Tensor, "*B 2"]) -> Float[Tensor, "*B 1"]:
    px, py = p[..., 0], p[..., 1]
    p3_full = torch.stack([px, py, px], -1) * MOD3.view(
        *[1 for _ in range(len(px.shape))], 3
    ).to(p.device)
    p3_floor = torch.floor(p3_full)
    p3_fract = p3_full - p3_floor

    p3_yzx = torch.stack([p3_fract[..., 1], p3_fract[..., 2], p3_fract[..., 0]], -1)
    p3 = p3_fract + dot(p3_fract, p3_yzx + 19.19)
    p3_x, p3_y, p3_z = p3[..., 0], p3[..., 1], p3[..., 2]

    p1 = (p3_x + p3_y) * p3_z
    p1_floor = torch.floor(p1)
    return (p1 - p1_floor).unsqueeze(-1)


@torch.no_grad()
def hash22(p: Float[Tensor, "*B 2"]) -> Float[Tensor, "*B 2"]:
    px, py = p[..., 0], p[..., 1]
    p3_full = torch.stack([px, py, px], -1) * MOD3.view(
        *[1 for _ in range(len(px.shape))], 3
    ).to(p.device)
    p3_floor = torch.floor(p3_full)
    p3_fract = p3_full - p3_floor

    p3_yzx = torch.stack([p3_fract[..., 1], p3_fract[..., 2], p3_fract[..., 0]], -1)
    p3 = p3_fract + dot(p3_fract, p3_yzx + 19.19)

    p3_x, p3_y, p3_z = p3[..., 0], p3[..., 1], p3[..., 2]
    p2 = torch.stack([(p3_x + p3_y) * p3_z, (p3_x + p3_z) * p3_y], -1)

    p2_floor = torch.floor(p2)
    return p2 - p2_floor


@torch.no_grad()
def get_random(uv: Float[Tensor, "*B 2"]) -> Float[Tensor, "*B 2"]:
    return F.normalize(hash22(uv * 126.1231) * 2.0 - 1.0, dim=-1)


def calc_ambient_occlusion(
    tcoord: Float[Tensor, "*B 2"],
    uv: Float[Tensor, "*B 2"],
    p: Float[Tensor, "*B 3"],
    cnorm: Float[Tensor, "*B 3"],
    pos_texture: Float[Tensor, "B 3 H W"],
    scale: float,
    bias: float,
    max_distance: float,
) -> Float[Tensor, "*B 1"]:
    offset_coord = (tcoord + uv).permute(1, 0, *range(2, len(tcoord.shape)))

    tex_query = access_coord(pos_texture, offset_coord).permute(
        1, 0, *range(2, len(tcoord.shape))
    )
    diff = tex_query - p
    l = torch.norm(diff, dim=-1, keepdim=True)
    v = diff / l
    d = l * scale
    ao = (dot(cnorm, v) - bias).clamp(min=0.0) * (1.0 / (1.0 + d))
    ao = ao * smoothstep(max_distance, max_distance * 0.5, l)
    return ao


def spiral_ao(
    uv: Float[Tensor, "*B 2"],
    pos: Float[Tensor, "*B 3"],
    normal: Float[Tensor, "*B 3"],
    rad: Float[Tensor, "*B 1"],
    pos_texture: Float[Tensor, "B 3 H W"],
    samples: int,
    scale: float,
    bias: float,
    max_distance: float,
) -> Float[Tensor, "*B 1"]:
    goldenAngle = 2.4
    inv = 1.0 / samples

    rStep: Float[Tensor, "1 *B 1"] = (inv * rad).unsqueeze(0)

    samples: Float[Tensor, "A *B 1"] = torch.arange(
        1, samples + 1, dtype=torch.float32, device=uv.device
    ).view(-1, *[1 for _ in rStep.shape[1:]])

    radius_step: Float[torch.Tensor, "A *B 1"] = samples * rStep

    rotatePhase: Float[Tensor, "1 *B 1"] = hash12(uv.unsqueeze(0) * 100.0) * 6.28
    rotatePhase: Float[Tensor, "A *B 1"] = rotatePhase + goldenAngle * (samples - 1)
    spiral_uv: Float[Tensor, "A *B 2"] = torch.cat(
        [torch.sin(rotatePhase), torch.cos(rotatePhase)], -1
    )

    ao_ind: Float[Tensor, "A *B 1"] = calc_ambient_occlusion(
        uv.unsqueeze(0),
        spiral_uv * radius_step,
        pos.unsqueeze(0),
        normal.unsqueeze(0),
        pos_texture,
        scale,
        bias,
        max_distance,
    )
    ao = ao_ind.sum(dim=0) * inv
    return ao


def calculate_ssao(
    positions: Float[Tensor, "B H W 3"],
    normals: Float[Tensor, "B H W 3"],
    mask: Float[Tensor, "B H W 1"],
    c2w: Float[Tensor, "B 4 4"],
    sample_rad: float = 0.02,
    samples: int = 16,
    intensity: float = 1.5,
    scale: float = 2.5,
    bias: float = 0.05,
    max_distance: float = 0.15,
    negative_depth: bool = True,
) -> Float[torch.Tensor, "B 1 H W"]:
    uv = (
        torch.stack(
            torch.meshgrid(
                torch.linspace(
                    0,
                    1,
                    positions.shape[2],
                    dtype=torch.float32,
                    device=positions.device,
                ),
                torch.linspace(
                    0,
                    1,
                    positions.shape[1],
                    dtype=torch.float32,
                    device=positions.device,
                ),
                indexing="xy",
            ),
            -1,
        )
        .unsqueeze(0)
        .expand(positions.shape[0], -1, -1, -1)
    )
    # Positions and normals are in world space. Move to camera space.
    w2c = c2w.inverse()
    vs_pos = torch.einsum(
        "bij,bsj->bsi",
        w2c,
        F.pad(positions, (0, 1), value=1.0).view(positions.shape[0], -1, 4),
    )[..., :3].view(positions.shape)
    vs_norm = torch.einsum(
        "bij,bsj->bsi",
        w2c,
        F.pad(normals, (0, 1), value=0.0).view(normals.shape[0], -1, 4),
    )[..., :3].view(normals.shape)

    if negative_depth:
        bg_depth = -100
    else:
        bg_depth = 100

    positions = torch.zeros_like(positions)
    positions[..., -1] = bg_depth
    positions[mask[..., 0] > 0.5] = vs_pos[mask[..., 0] > 0.5]

    normals = torch.zeros_like(normals)
    normals[..., -1] = 1
    normals[mask[..., 0] > 0.5] = vs_norm[mask[..., 0] > 0.5]

    rad = sample_rad / abs(positions[..., -1:])

    ao = spiral_ao(
        uv,
        positions,
        normals,
        rad,
        positions.permute(0, 3, 1, 2),
        samples=samples,
        scale=scale,
        bias=bias,
        max_distance=max_distance,
    )

    ao = 1.0 - ao * intensity

    return ao.clip(0, 1)
