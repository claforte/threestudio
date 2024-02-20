import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mesh2sdf
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
from skimage.io import imread
from tqdm import tqdm


def nearest_dist(pts0, pts1, batch_size=512):
    pts0 = torch.from_numpy(pts0.astype(np.float32)).cuda()
    pts1 = torch.from_numpy(pts1.astype(np.float32)).cuda()
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    for i in tqdm(range(0, pn0, batch_size), desc="evaluating..."):
        dist = torch.norm(pts0[i : i + batch_size, None, :] - pts1[None, :, :], dim=-1)
        dists.append(torch.min(dist, 1)[0])
    dists = torch.cat(dists, 0)
    return dists.cpu().numpy()


def norm_coords(vertices):
    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    scale = 1 / np.max(max_pt - min_pt)
    vertices = vertices * scale

    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    center = (max_pt + min_pt) / 2
    # center = np.mean(vertices, 0)
    vertices = vertices - center[None, :]
    return vertices


def get_chamfer_iou(mesh_pr, mesh_gt, vis_align=False):
    pts_pr = np.asarray(mesh_pr.vertices)
    pts_gt = np.asarray(mesh_gt.vertices)

    if vis_align:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(pts_pr[:, 0], pts_pr[:, 1], pts_pr[:, 2], s=1)
        ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], s=1)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        plt.savefig("scatter.png")

    # compute iou
    size = 64
    sdf_pr = mesh2sdf.compute(
        mesh_pr.vertices, mesh_pr.triangles, size, fix=False, return_mesh=False
    )
    sdf_gt = mesh2sdf.compute(
        mesh_gt.vertices, mesh_gt.triangles, size, fix=False, return_mesh=False
    )
    vol_pr = sdf_pr < 0
    vol_gt = sdf_gt < 0
    iou = np.sum(vol_pr & vol_gt) / np.sum(vol_gt | vol_pr)

    dist0 = nearest_dist(pts_pr, pts_gt, batch_size=4096)
    dist1 = nearest_dist(pts_gt, pts_pr, batch_size=4096)

    chamfer = (np.mean(dist0) + np.mean(dist1)) / 2
    return chamfer, iou


def eval_one_mesh(
    mesh_pr, mesh_gt, azim=0, elev=0, rigid_align=False, icp=False, vis_align=False
):
    mesh_gt = o3d.io.read_triangle_mesh(mesh_gt)
    mesh_pr = o3d.io.read_triangle_mesh(mesh_pr)
    vertices_gt = np.asarray(mesh_gt.vertices)
    vertices_pr = np.asarray(mesh_pr.vertices)

    if rigid_align:
        print(azim, elev)
        # SV3D
        rot_mat = R.from_rotvec(np.array([0, 0, -np.pi / 2 + azim])).as_matrix()
        # One-2345
        # rot_mat = R.from_rotvec(np.array([np.pi/2, azim, -elev])).as_matrix()
        # rot_mat = R.from_rotvec(np.array([np.pi/2, -elev, azim])).as_matrix()
        # Shap-E
        # rot_mat = R.from_rotvec(np.array([0, 0, -np.pi/2+azim])).as_matrix()
        # DreamGaussian & One-2345
        # rot_mat = R.from_rotvec(np.array([np.pi/2, 0, 0])).as_matrix()
        # rot_mat2 = R.from_rotvec(np.array([0, 0, azim])).as_matrix()
        # rot_mat3 = R.from_rotvec(np.array([0, -elev, 0])).as_matrix()
        # rot_mat = np.matmul(rot_mat3, np.matmul(rot_mat2, rot_mat))

        trans_init = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        trans_init[:3, :3] = rot_mat
        pcd_pr = o3d.geometry.PointCloud()
        pcd_gt = o3d.geometry.PointCloud()
        pcd_pr.points = o3d.utility.Vector3dVector(vertices_pr)
        pcd_gt.points = o3d.utility.Vector3dVector(vertices_gt)

        if icp:
            threshold = 0.01
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_pr,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            )
            pcd_pr = pcd_pr.transform(reg_p2p.transformation)

        pcd_pr = pcd_pr.transform(trans_init)
        vertices_pr = pcd_pr.points

    vertices_gt = norm_coords(vertices_gt)
    vertices_pr = norm_coords(vertices_pr)
    mesh_gt.vertices = o3d.utility.Vector3dVector(vertices_gt)
    mesh_pr.vertices = o3d.utility.Vector3dVector(vertices_pr)

    return get_chamfer_iou(mesh_pr, mesh_gt, vis_align)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr_dir", type=str, required=True)
    parser.add_argument("--gt_mesh", type=str, required=True)
    parser.add_argument("--transform_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--vis_align", action="store_true", default=False, dest="vis_align"
    )
    parser.add_argument(
        "--rigid_align", action="store_true", default=False, dest="rigid_align"
    )
    parser.add_argument("--icp", action="store_true", default=False, dest="icp")
    args = parser.parse_args()

    chamfer_list = []
    iou_list = []
    results = ""
    gt_objs = sorted(glob.glob(args.gt_mesh))

    for i, gt_obj in enumerate(gt_objs):
        obj_name = gt_obj.split("/")[-2]
        pr_obj = os.path.join(args.pr_dir, obj_name, "save", "model.obj")
        # pr_obj = args.pr_dir.replace("*", obj_name)
        transform_file = os.path.join(args.transform_dir, obj_name, "frame_0020.json")
        if os.path.exists(pr_obj):
            transforms = json.load(open(transform_file, "r"))
            azim, elev = transforms["azimuth"], np.pi / 2 - transforms["polar"]
            chamfer, iou = eval_one_mesh(
                pr_obj, gt_obj, azim, elev, args.rigid_align, args.icp, args.vis_align
            )
            chamfer_list.append(chamfer)
            iou_list.append(iou)
            print_result = f"{obj_name}: Chamfer={chamfer:.5f}, Volume IOU={iou:.5f}\n"
            results += print_result
            print(print_result)

    print(len(chamfer_list))
    chamfer_mean = np.mean(np.array(chamfer_list))
    iou_mean = np.mean(np.array(iou_list))
    print_result = f"Mean: Chamfer={chamfer_mean:.5f}, Volume IOU={iou_mean:.5f}\n"
    results += print_result
    print(print_result)

    with open(args.output_file, "w") as f:
        f.write(results + "\n")


if __name__ == "__main__":
    main()
