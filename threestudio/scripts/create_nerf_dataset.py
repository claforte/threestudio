import glob
import json
import math
import os

import fire
import imageio
import matplotlib.pyplot as plt
import numpy as np
import ray_volume
import torch
from PIL import Image


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def look_at(camera_position, target_position, up_vector):
    z_axis = camera_position - target_position
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up_vector, z_axis)
    y_axis = np.cross(z_axis, x_axis)

    camera_matrix = np.array(
        [
            [x_axis[0], y_axis[0], z_axis[0], camera_position[0]],
            [x_axis[1], y_axis[1], z_axis[1], camera_position[1]],
            [x_axis[2], y_axis[2], z_axis[2], camera_position[2]],
            [0, 0, 0, 1],
        ]
    )

    return camera_matrix


def generate_camera_matrices(camera_dists, polars_rad, azims_rad):
    target_position = np.array([0, 0, 0])
    up_vector = np.array([0, 0, 1])
    camera_matrices = []
    for camera_dist, polar, azim in zip(camera_dists, polars_rad, azims_rad):
        camera_position = spherical_to_cartesian(camera_dist, polar, azim)
        view_matrix = look_at(np.array(camera_position), target_position, up_vector)
        camera_matrices.append(view_matrix)

    return camera_matrices


def get_rays_for_frame(width, height, cx, cy, fl_x, fl_y, c2w):
    x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")

    local_dirs = torch.stack(
        [
            (x - cx + 0.5) / fl_x,
            -(y - cy + 0.5) / fl_y,
            -(torch.ones_like(x)),
        ],
        dim=-1,
    )

    local_dirs = local_dirs / torch.linalg.norm(local_dirs, dim=-1, keepdims=True)

    c2w = torch.tensor(c2w)

    world_dirs = (
        (local_dirs[:, :, None, :] * c2w[None, None, :3, :3]).sum(-1).squeeze(-1)
    )
    world_origs = (
        c2w[:3, -1]
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(world_dirs.shape[0], world_dirs.shape[1], 1)
    )

    return world_origs, world_dirs


def viz_grid(grid, TARGET_GRID_DIM, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.array(grid.cpu().numpy().nonzero())[:, 0:-1:5]
    ax.scatter(x, y, z, c="b", marker="o")
    ax.set_xlim(0, TARGET_GRID_DIM)
    ax.set_ylim(0, TARGET_GRID_DIM)
    ax.set_zlim(0, TARGET_GRID_DIM)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(filename, bbox_inches="tight")


def create_transforms(
    images_dir,
    polars_rad=np.deg2rad(90 - 5),
    azims_rad=None,
    camera_dists=3.5,  # 2
    fov_deg=33.9,
    radius=1,  # 2
    TARGET_GRID_DIM=128,  # 64
    visualize=False,
):
    images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if len(images) == 0:
        print(f"No images found in {images_dir}")
        return

    width, height = Image.open(images[0]).size
    assert width == height
    image_size = width

    img_data = [np.asarray(Image.open(image)) for image in images]
    masks = [
        torch.tensor(np.any(img[:, :, :3] <= [252, 252, 252], axis=-1))
        for img in img_data
    ]

    if visualize:
        for i, img in enumerate(img_data):
            imageio.imwrite(os.path.join(images_dir, f"CHECK_img{i}.jpg"), img)
        for i, mask in enumerate(masks):
            imageio.imwrite(
                os.path.join(images_dir, f"CHECK_mask{i}.jpg"),
                mask.numpy().astype(np.uint8) * 255,
            )

    if isinstance(polars_rad, float) or isinstance(polars_rad, int):
        polars_rad = [polars_rad] * len(images)

    if azims_rad is None:
        azims_rad = [2 * np.pi / len(images) * (i + 1) for i in range(len(images))]

    if isinstance(camera_dists, float) or isinstance(camera_dists, int):
        camera_dists = [camera_dists] * len(images)

    c = image_size * 0.5
    fl = 0.5 * image_size / np.tan(0.5 * (fov_deg * math.pi / 180))
    aabb = torch.tensor(
        [-radius, -radius, -radius, radius, radius, radius], dtype=torch.float32
    ).cuda()

    matrices = generate_camera_matrices(camera_dists, polars_rad, azims_rad)

    # make grid

    hit_o, hit_d, hit_indices = [], [], []
    for i, c2w in enumerate(matrices):
        o, d = get_rays_for_frame(image_size, image_size, c, c, fl, fl, c2w)
        mask = masks[i]
        hit_o.append(o[mask])
        hit_d.append(d[mask])
        hit_indices.append(torch.tensor([i]).repeat(o[mask].shape[0]))

    hit_o = torch.concat(hit_o).float()
    hit_d = torch.concat(hit_d).float()
    hit_indices = torch.concat(hit_indices).long()

    grid = ray_volume.get_volume(
        hit_o.cuda(),
        hit_d.cuda(),
        hit_indices.cuda(),
        aabb,
        TARGET_GRID_DIM,
        len(matrices),
    )
    # grid_occupancy = grid.count_nonzero().cpu()

    if visualize:
        viz_grid(grid, TARGET_GRID_DIM, os.path.join(images_dir, "CHECK_grid.jpg"))

    c = image_size * 0.5
    fl = 0.5 * image_size / np.tan(0.5 * (fov_deg * math.pi / 180))
    out_frames = []
    for matrix, image_path in zip(matrices, images):
        out_frame = {
            "fl_x": fl,
            "fl_y": fl,
            "cx": c,
            "cy": c,
            "w": image_size,
            "h": image_size,
            "file_path": f"./{os.path.basename(image_path)}",
            "transform_matrix": matrix.tolist(),
        }
        out_frames.append(out_frame)

    out = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": out_frames,
    }

    with open(os.path.join(images_dir, "transforms.json"), "w") as of:
        json.dump(out, of, indent=5)


if __name__ == "__main__":
    fire.Fire(create_transforms)
    print("done.")
