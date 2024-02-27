from __future__ import annotations

import fast_simplification
import gpytoolbox
import numpy as np
import pymeshlab
import torch
import torch.nn.functional as F

import threestudio
from threestudio.utils.ops import dot
from threestudio.utils.typing import *


class Mesh:
    def __init__(
        self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"], **kwargs
    ) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = None
        self._t_tex_idx: Optional[Float[Tensor, "Nf 3"]] = None
        self._v_rgb: Optional[Float[Tensor, "Nv 3"]] = None
        self._edges: Optional[Integer[Tensor, "Ne 2"]] = None
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    def simplify(
        self, target_reduction: float = None, target_count: int = None
    ) -> Mesh:
        if target_reduction is None and target_count is None:
            raise ValueError(
                "Either target_reduction or target_count must be specified"
            )

        # Check that only self.v_pos and self.t_pos_idx are used
        if len(self.extras) > 0:
            threestudio.warn(
                f"The following extra attributes are ignored during simplification: {list(self.extras.keys())}"
            )
        if self._v_nrm is not None:
            threestudio.warn("Vertex normals are ignored during simplification")
        if self._v_tng is not None:
            threestudio.warn("Vertex tangents are ignored during simplification")
        if self._v_tex is not None:
            threestudio.warn(
                "Vertex texture coordinates are ignored during simplification"
            )
        if self._t_tex_idx is not None:
            threestudio.warn(
                "Triangle texture coordinates are ignored during simplification"
            )
        if self._v_rgb is not None:
            threestudio.warn("Vertex colors are ignored during simplification")
        if self._edges is not None:
            threestudio.warn("Edges are ignored during simplification")

        # Convert to numpy
        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float32)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int64)

        # Simplify
        points_out, faces_out = fast_simplification.simplify(
            v_pos,
            t_pos_idx,
            target_reduction=target_reduction,
            target_count=target_count,
        )

        # Convert back to torch
        v_pos = torch.from_numpy(points_out).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(faces_out).to(self.t_pos_idx).contiguous()

        # Create new mesh
        simplified_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        return simplified_mesh

    def fix_non_manifold_mesh(self) -> Mesh:
        # Check that only self.v_pos and self.t_pos_idx are used
        if len(self.extras) > 0:
            threestudio.warn(
                f"The following extra attributes are ignored during simplification: {list(self.extras.keys())}"
            )
        if self._v_nrm is not None:
            threestudio.warn("Vertex normals are ignored during simplification")
        if self._v_tng is not None:
            threestudio.warn("Vertex tangents are ignored during simplification")
        if self._v_tex is not None:
            threestudio.warn(
                "Vertex texture coordinates are ignored during simplification"
            )
        if self._t_tex_idx is not None:
            threestudio.warn(
                "Triangle texture coordinates are ignored during simplification"
            )
        if self._v_rgb is not None:
            threestudio.warn("Vertex colors are ignored during simplification")
        if self._edges is not None:
            threestudio.warn("Edges are ignored during simplification")

        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float64)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)

        # Load pylabmesh
        m = pymeshlab.Mesh(v_pos, t_pos_idx)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "generated")

        # Fix mesh
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices()

        # Apply fixes to new mesh
        m = ms.current_mesh()
        v_pos = m.vertex_matrix()
        t_pos_idx = m.face_matrix()

        v_pos = torch.from_numpy(v_pos).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(t_pos_idx).to(self.t_pos_idx).contiguous()
        return Mesh(v_pos, t_pos_idx)

    def clean_mesh(self) -> Mesh:
        if len(self.extras) > 0:
            threestudio.warn(
                f"The following extra attributes are ignored during simplification: {list(self.extras.keys())}"
            )
        if self._v_nrm is not None:
            threestudio.warn("Vertex normals are ignored during simplification")
        if self._v_tng is not None:
            threestudio.warn("Vertex tangents are ignored during simplification")
        if self._v_tex is not None:
            threestudio.warn(
                "Vertex texture coordinates are ignored during simplification"
            )
        if self._t_tex_idx is not None:
            threestudio.warn(
                "Triangle texture coordinates are ignored during simplification"
            )
        if self._v_rgb is not None:
            threestudio.warn("Vertex colors are ignored during simplification")
        if self._edges is not None:
            threestudio.warn("Edges are ignored during simplification")

        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float64)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)

        # Load pylabmesh
        m = pymeshlab.Mesh(v_pos, t_pos_idx)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "generated")

        # Handle floaters
        p = pymeshlab.PercentageValue(20)
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=p)

        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices()

        # Apply fixes to new mesh
        m = ms.current_mesh()
        v_pos = m.vertex_matrix()
        t_pos_idx = m.face_matrix()

        v_pos = torch.from_numpy(v_pos).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(t_pos_idx).to(self.t_pos_idx).contiguous()
        return Mesh(v_pos, t_pos_idx)

    def close_holes(self) -> Mesh:
        if len(self.extras) > 0:
            threestudio.warn(
                f"The following extra attributes are ignored during simplification: {list(self.extras.keys())}"
            )
        if self._v_nrm is not None:
            threestudio.warn("Vertex normals are ignored during simplification")
        if self._v_tng is not None:
            threestudio.warn("Vertex tangents are ignored during simplification")
        if self._v_tex is not None:
            threestudio.warn(
                "Vertex texture coordinates are ignored during simplification"
            )
        if self._t_tex_idx is not None:
            threestudio.warn(
                "Triangle texture coordinates are ignored during simplification"
            )
        if self._v_rgb is not None:
            threestudio.warn("Vertex colors are ignored during simplification")
        if self._edges is not None:
            threestudio.warn("Edges are ignored during simplification")

        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float64)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)

        # Load pylabmesh
        m = pymeshlab.Mesh(v_pos, t_pos_idx)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "generated")

        # num_eges = (t_pos_idx.shape[0] // 3) // 10
        ms.meshing_close_holes()  # maxholesize=num_eges)

        # Apply fixes to new mesh
        m = ms.current_mesh()
        v_pos = m.vertex_matrix()
        t_pos_idx = m.face_matrix()

        v_pos = torch.from_numpy(v_pos).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(t_pos_idx).to(self.t_pos_idx).contiguous()
        return Mesh(v_pos, t_pos_idx)

    def remesh(
        self, average_edge_length_multiplier: float = 1.0, remesh_steps: int = 10
    ) -> Mesh:
        # Check that only self.v_pos and self.t_pos_idx are used
        if len(self.extras) > 0:
            threestudio.warn(
                f"The following extra attributes are ignored during simplification: {list(self.extras.keys())}"
            )
        if self._v_nrm is not None:
            threestudio.warn("Vertex normals are ignored during simplification")
        if self._v_tng is not None:
            threestudio.warn("Vertex tangents are ignored during simplification")
        if self._v_tex is not None:
            threestudio.warn(
                "Vertex texture coordinates are ignored during simplification"
            )
        if self._t_tex_idx is not None:
            threestudio.warn(
                "Triangle texture coordinates are ignored during simplification"
            )
        if self._v_rgb is not None:
            threestudio.warn("Vertex colors are ignored during simplification")
        if self._edges is not None:
            threestudio.warn("Edges are ignored during simplification")

        # Convert to numpy
        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float64)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)

        # Remesh

        # Check if mesh is non manifold:
        ne = gpytoolbox.non_manifold_edges(t_pos_idx)
        if len(ne) > 0:
            # Try to fix with pymeshlab
            threestudio.warn(
                "Mesh is non manifold, trying to fix with pymeshlab. Non manifold edges: {}".format(
                    ne
                )
            )

            try:
                fixed_mesh = self.fix_non_manifold_mesh()
            except Exception as e:
                threestudio.warn(
                    f"Failed to fix non manifold mesh with pymeshlab: {e}. Returning original mesh"
                )
                return self
            v_pos = fixed_mesh.v_pos.detach().cpu().numpy().astype(np.float64)
            t_pos_idx = fixed_mesh.t_pos_idx.detach().cpu().numpy().astype(np.int32)

            ne = gpytoolbox.non_manifold_edges(t_pos_idx)
            if len(ne) > 0:
                threestudio.warn(
                    "Mesh is still non manifold after pymeshlab. Non manifold edges: {}".format(
                        ne
                    )
                )
                return self

        # Check if mesh is closed
        be = gpytoolbox.boundary_edges(t_pos_idx)
        if len(be) > 0:
            threestudio.warn(
                "Mesh is not closed, tying to fix with pymeshlab. Boundary edges: {}".format(
                    be
                )
            )

            try:
                fixed_mesh = self.close_holes()
            except Exception as e:
                threestudio.warn(
                    f"Failed to close holes with pymeshlab: {e}. Returning original mesh"
                )
                return self

            v_pos = fixed_mesh.v_pos.detach().cpu().numpy().astype(np.float64)
            t_pos_idx = fixed_mesh.t_pos_idx.detach().cpu().numpy().astype(np.int32)

            be = gpytoolbox.boundary_edges(t_pos_idx)
            if len(be) > 0:
                threestudio.warn(
                    "Mesh is still not closed after pymeshlab. Boundary edges: {}".format(
                        be
                    )
                )
                return self

        edges = self.edges
        average_edge_length = (
            torch.linalg.norm(self.v_pos[edges[:, 0]] - self.v_pos[edges[:, 1]], dim=1)
            .mean()
            .item()
        )

        threestudio.info(
            f"Remeshing with {average_edge_length * average_edge_length_multiplier} as target edge length and {remesh_steps} steps"
        )
        v_remesh, f_remesh = gpytoolbox.remesh_botsch(
            v_pos,
            t_pos_idx,
            remesh_steps,
            float(average_edge_length * average_edge_length_multiplier),
        )

        # Convert back to torch
        v_pos = torch.from_numpy(v_remesh).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(f_remesh).to(self.t_pos_idx).contiguous()

        # Create new mesh
        remeshed_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        return remeshed_mesh

    def remove_outlier(self, outlier_n_faces_threshold: Union[int, float]) -> Mesh:
        if self.requires_grad:
            threestudio.debug("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        threestudio.debug(
            "Mesh has {} components, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        threestudio.debug(
            "Removing components with less than {} faces".format(n_faces_threshold)
        )

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        threestudio.debug(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            threestudio.debug(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self):
        if self._v_tex is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self):
        if self._t_tex_idx is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._t_tex_idx

    @property
    def v_rgb(self):
        return self._v_rgb

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def _unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        threestudio.info("Using xatlas to perform UV unwrapping, may take a while ...")

        import xatlas

        atlas = xatlas.Atlas()
        atlas.add_mesh(
            self.v_pos.detach().cpu().numpy(),
            self.t_pos_idx.cpu().numpy(),
        )
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in xatlas_chart_options.items():
            setattr(co, k, v)
        for k, v in xatlas_pack_options.items():
            setattr(po, k, v)
        atlas.generate(co, po)
        vmapping, indices, uvs = atlas.get_mesh(0)
        vmapping = (
            torch.from_numpy(
                vmapping.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        uvs = torch.from_numpy(uvs).to(self.v_pos.device).float()
        indices = (
            torch.from_numpy(
                indices.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        return uvs, indices

    def unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        self._v_tex, self._t_tex_idx = self._unwrap_uv(
            xatlas_chart_options, xatlas_pack_options
        )

    def set_vertex_color(self, v_rgb):
        assert v_rgb.shape[0] == self.v_pos.shape[0]
        self._v_rgb = v_rgb

    def _compute_edges(self):
        # Compute edges
        edges = torch.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        return edges

    def normal_consistency(self) -> Float[Tensor, ""]:
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.v_nrm[self.edges]
        nc = (
            1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)
        ).mean()
        return nc

    def _laplacian_uniform(self):
        # from stable-dreamfusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/8fb3613e9e4cd1ded1066b46e80ca801dfb9fd06/nerf/renderer.py#L224
        verts, faces = self.v_pos, self.t_pos_idx

        V = verts.shape[0]
        F = faces.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(
            dim=1
        )
        adj_values = torch.ones(adj.shape[1]).to(verts)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    def laplacian(self) -> Float[Tensor, ""]:
        with torch.no_grad():
            L = self._laplacian_uniform()
        loss = L.mm(self.v_pos)
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss
