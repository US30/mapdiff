"""Gaussian Splat (.ply) -> cleaned coloured point cloud -> Poisson surface mesh.

    uv run python recon/splat_to_mesh.py recon/out/truck.ply --out recon/out

Steps
  1. Read 3DGS PLY: centres, opacity (logit), scales (log), SH DC colour.
  2. Keep confident, compact Gaussians (opacity > --min-opacity, scale below 99th pct):
     large/transparent Gaussians are background haze, not surface.
  3. Voxel downsample + statistical outlier removal (point-cloud fusion clean-up).
  4. Normals -> screened Poisson reconstruction; trim low-density vertices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load_splat(path: Path):
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    opacity = 1 / (1 + np.exp(-v["opacity"]))
    scale = np.exp(np.column_stack([v[f"scale_{i}"] for i in range(3)])).max(1)
    rgb = np.clip(0.5 + SH_C0 * np.column_stack([v[f"f_dc_{i}"] for i in range(3)]), 0, 1)
    return xyz, opacity, scale, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--out", type=Path, default=Path("recon/out"))
    ap.add_argument("--voxel", type=float, default=0.01, help="relative to scene extent")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--center-crop", type=float, default=1.0,
                    help="keep points within this fraction of the scene radius around the "
                         "densest point (e.g. 0.25 isolates the central object)")
    ap.add_argument("--min-opacity", type=float, default=0.2,
                    help="Brush regularises opacity, so most surface splats sit at 0.1-0.4")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    xyz, op, sc, rgb = load_splat(a.ply)
    keep = (op > a.min_opacity) & (sc <= np.percentile(sc, 99))
    if keep.sum() < 100:
        raise SystemExit(f"only {keep.sum()} confident Gaussians; check the PLY")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[keep]))
    pcd.colors = o3d.utility.Vector3dVector(rgb[keep])

    # crop to the central 98% of points to drop distant background floaters
    lo, hi = np.percentile(xyz[keep], [1, 99], axis=0)
    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    extent = float(np.linalg.norm(hi - lo))
    if a.center_crop < 1.0:
        P = np.asarray(pcd.points)
        # densest region = scene subject (cameras orbit it, so it gets most splats)
        H, edges = np.histogramdd(P, bins=20)
        idx = np.unravel_index(H.argmax(), H.shape)
        centre = np.array([(e[i] + e[i + 1]) / 2 for e, i in zip(edges, idx)])
        r = a.center_crop * extent / 2
        pcd = pcd.select_by_index(np.where(np.linalg.norm(P - centre, axis=1) < r)[0])
        extent = 2 * r
    pcd = pcd.voxel_down_sample(a.voxel * extent / 10)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=extent / 100, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(15)
    o3d.io.write_point_cloud(str(a.out / "points.ply"), pcd)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=a.depth, n_threads=1)  # multithreaded Poisson crashes on macOS arm64
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(a.out / "mesh.ply"), mesh)

    stats = {
        "gaussians": int(len(xyz)),
        "kept_gaussians": int(keep.sum()),
        "points_after_cleanup": len(pcd.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "watertight": bool(mesh.is_watertight()),
    }
    (a.out / "mesh_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
