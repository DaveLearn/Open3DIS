"""CLI entry point for Open3DIS external segmenter.

Usage (invoked by pixi task):
    python segment.py <observations_path> <scene_path> [--use-3d-proposals]

Outputs ``objects_path: <path>`` to stdout for the parent process to read.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import contextlib
import json
import logging
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import tyro
import yaml
import gdown
import certifi
import requests
import imageio.v2 as imageio
import open3d as o3d
from munch import Munch
from plyfile import PlyData, PlyElement

from initializerdefs import (
    InstanceMaskObjectsDef,
    ObjectSegmentations,
    Observations,
    ObservationFrame,
    SceneSetup,
)
from psdframe import Frame

from helpers.debug_visualize import DebugVisualizer


logger = logging.getLogger("open3dis-segmenter")

SEGMENTATOR_DIR = Path(__file__).resolve().parent.parent / "SAI3D" / "Segmentator"
ISBNET_SCANNET200_FILE_ID = "1ZEZgQeT6dIakljSTx4s5YZM0n2rwC3Kw"
ISBNET_SCANNET200_FILENAME = "isbnet_scannet200.pth"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_FILENAME = "sam_vit_h_4b8939.pth"
GROUNDING_DINO_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
RAM_PLUS_URL = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"


@dataclass
class Args:
    observations_path: tyro.conf.Positional[Path]
    """Path to the pickled Observations."""

    scene_path: tyro.conf.Positional[Path]
    """Path to the pickled SceneSetup."""

    dataset_mode: Literal["scannetpp", "scannet200"] = "scannetpp"
    """Dataset mode for Open3DIS config. Default scannetpp for custom data."""

    use_3d_proposals: bool = False
    """Merge 3D proposals with 2D proposals (requires external 3D proposals)."""

    use_superpoints: bool = False
    """Use superpoints in Open3DIS clustering (requires dc_features)."""

    dc_features_path: Optional[Path] = None
    """Override path to dc_features folder (.pth files)."""

    isbnet_config: Optional[Path] = None
    """Optional ISBNet config override for 3D backbone."""

    isbnet_checkpoint: Optional[Path] = None
    """ISBNet checkpoint for generating 3D proposals + dc_features."""

    cls_agnostic_3d_proposals_path: Optional[Path] = None
    """Override path to class-agnostic 3D proposals (.pth files)."""

    exp_name: str = "deg_open3dis"
    """Experiment name for Open3DIS outputs."""

    img_interval: int = 1
    """Image interval for Open3DIS 2D grounding and clustering."""

    k_thresh: float = 0.01
    """Segmentator kThresh parameter."""

    seg_min_verts: int = 20
    """Segmentator minimum vertices per segment."""

    mask2d_output: str = "mask_deg"
    """Output folder name for Open3DIS 2D masks."""

    output_dir: Optional[Path] = None
    """Optional override for output directory."""

    config_template: Optional[Path] = None
    """Optional override for Open3DIS config template."""


def get_dataset_frame_from_observation_frame(observation_frame: ObservationFrame) -> Frame:
    return Frame(
        id=observation_frame.id,
        name=observation_frame.name,
        color=torch.tensor(observation_frame.color).cuda(),
        X_WV=torch.tensor(observation_frame.X_WV),
        K=torch.tensor(observation_frame.K),
        depth=(
            torch.tensor(observation_frame.depth).cuda()
            if observation_frame.depth is not None
            else None
        ),
    )


def _to_cam_open3d(frame: Frame) -> o3d.camera.PinholeCameraParameters:
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        frame.w, frame.h, frame.fl_x, frame.fl_y, frame.cx, frame.cy
    )
    extrinsic = frame.X_VW_opencv.cpu().numpy()
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic
    return camera


def _post_process_mesh(
    mesh: o3d.geometry.TriangleMesh, cluster_to_keep: int = 1000
) -> o3d.geometry.TriangleMesh:
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, _ = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    logger.info("mesh vertices raw %d -> post %d", len(mesh.vertices), len(mesh_0.vertices))
    return mesh_0


@torch.no_grad()
def _extract_mesh_bounded(
    frames: List[Frame],
    voxel_size: float = 0.004,
    sdf_trunc: float = 0.02,
    depth_trunc: float = 3,
) -> o3d.geometry.TriangleMesh:
    logger.info(
        "TSDF integration: voxel_size=%.4f  sdf_trunc=%.4f  depth_trunc=%.2f",
        voxel_size,
        sdf_trunc,
        depth_trunc,
    )
    for frame in frames:
        assert frame.depth is not None and frame.color is not None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for frame in frames:
        rgb = frame.color.cpu().numpy()
        depth = frame.depth.cpu().numpy() if frame.depth is not None else None
        assert depth is not None

        ci = o3d.geometry.Image((rgb * 255).astype(np.uint8))
        di = o3d.geometry.Image(depth)
        cam = _to_cam_open3d(frame)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            ci, di, depth_trunc=depth_trunc, convert_rgb_to_intensity=False, depth_scale=1.0
        )
        volume.integrate(rgbd, intrinsic=cam.intrinsic, extrinsic=cam.extrinsic)

    return volume.extract_triangle_mesh()


def _extract_mesh_bounded_with_res(
    frames: List[Frame], depth_trunc: float = 2, mesh_res: int = 1024
) -> o3d.geometry.TriangleMesh:
    voxel_size = depth_trunc / mesh_res
    sdf_trunc = 5.0 * voxel_size
    raw_mesh = _extract_mesh_bounded(frames, voxel_size, sdf_trunc, depth_trunc)
    return _post_process_mesh(raw_mesh, cluster_to_keep=50)


def get_workspace_voxels(scene: SceneSetup) -> o3d.geometry.VoxelGrid:
    table_xyz = scene.ground_gaussians.xyz
    table_plane = scene.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    table_pcd_extruded = np.array(table_xyz).copy()

    desired_height = 1.0
    voxel_size = 0.05
    iters = int(desired_height / voxel_size)
    for i in range(iters):
        new_points = table_xyz + table_normal * voxel_size * i
        table_pcd_extruded = np.append(table_pcd_extruded, new_points, axis=0)
    for i in range(5):
        table_pcd_extruded = np.append(
            table_pcd_extruded, table_xyz - table_normal * voxel_size * (i + 1), axis=0
        )

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_pcd_extruded))
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size * 2)


def _crop_mesh_to_workspace_bbox(
    mesh: o3d.geometry.TriangleMesh,
    workspace_voxels: o3d.geometry.VoxelGrid,
    padding_m: float = 0.2,
) -> o3d.geometry.TriangleMesh:
    voxel_size = float(workspace_voxels.voxel_size)
    origin = np.asarray(workspace_voxels.origin, dtype=np.float32)
    voxels = workspace_voxels.get_voxels()
    if len(voxels) == 0:
        return mesh

    indices = np.array([v.grid_index for v in voxels], dtype=np.float32)
    min_corner = origin + indices.min(axis=0) * voxel_size - padding_m
    max_corner = origin + (indices.max(axis=0) + 1.0) * voxel_size + padding_m
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
    return mesh.crop(aabb)


def _filter_labels_by_workspace(
    vertices: np.ndarray,
    labels: np.ndarray,
    workspace_voxels: o3d.geometry.VoxelGrid,
) -> np.ndarray:
    labels = labels.copy()
    pcd = o3d.utility.Vector3dVector(vertices)
    valid_mask = np.array(workspace_voxels.check_if_included(pcd))

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl <= 0:
            continue
        seg_mask = labels == lbl
        total = np.sum(seg_mask)
        inside = np.sum(valid_mask & seg_mask)
        if total > 0 and inside / total < 0.9:
            labels[seg_mask] = 0

    return labels


def _ensure_segmentator_binary() -> Path:
    binary = SEGMENTATOR_DIR / "segmentator"
    if binary.exists():
        return binary
    logger.info("Building Segmentator binary")
    result = subprocess.run(["make", "-C", str(SEGMENTATOR_DIR)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Segmentator build failed:\n%s", result.stderr)
        raise RuntimeError(f"Failed to build Segmentator: {result.stderr}")
    assert binary.exists(), f"Expected binary at {binary} after build"
    logger.info("Segmentator binary built at %s", binary)
    return binary


def _ensure_segmentator_ply(ply_path: Path) -> Path:
    try:
        ply = PlyData.read(str(ply_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read PLY {ply_path}: {exc}")

    if "vertex" not in ply or "face" not in ply:
        raise RuntimeError(f"PLY missing vertex/face elements: {ply_path}")

    v = ply["vertex"].data
    if not all(name in v.dtype.names for name in ("x", "y", "z")):
        raise RuntimeError(f"PLY vertex missing x/y/z: {ply_path}")

    verts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    verts_ok = verts.dtype == np.float32

    f = ply["face"].data
    face_prop = None
    for name in ("vertex_indices", "vertex_index"):
        if name in f.dtype.names:
            face_prop = name
            break
    if face_prop is None:
        raise RuntimeError(f"PLY face missing vertex_indices: {ply_path}")

    faces = np.vstack(f[face_prop])
    faces_ok = faces.dtype == np.uint32 and faces.shape[1] == 3

    if verts_ok and faces_ok and face_prop == "vertex_indices":
        return ply_path

    fixed_path = ply_path.with_suffix("")
    fixed_path = fixed_path.with_name(f"{fixed_path.name}.segmentator.ply")

    verts = verts.astype(np.float32)
    faces = faces[:, :3].astype(np.uint32)

    verts_el = np.empty(verts.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts_el["x"], verts_el["y"], verts_el["z"] = verts[:, 0], verts[:, 1], verts[:, 2]

    face_el = np.empty(faces.shape[0], dtype=[("vertex_indices", "u4", (3,))])
    face_el["vertex_indices"] = faces

    PlyData(
        [PlyElement.describe(verts_el, "vertex"), PlyElement.describe(face_el, "face")],
        text=False,
    ).write(str(fixed_path))
    logger.info("Wrote Segmentator-friendly PLY to %s", fixed_path)
    return fixed_path


def _run_segmentator(ply_path: Path, k_thresh: float, seg_min_verts: int) -> Path:
    binary = _ensure_segmentator_binary()
    ply_path = _ensure_segmentator_ply(ply_path)
    expected_output = ply_path.parent / f"{ply_path.stem}.{k_thresh}.segs.json"
    if expected_output.exists():
        logger.info("Superpoints already computed: %s", expected_output)
        return expected_output

    logger.info(
        "Running Segmentator on %s (kThresh=%s, segMinVerts=%d)",
        ply_path,
        k_thresh,
        seg_min_verts,
    )
    result = subprocess.run(
        [str(binary), str(ply_path), str(k_thresh), str(seg_min_verts)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Segmentator failed:\n%s", result.stderr)
        raise RuntimeError(f"Segmentator failed: {result.stderr}")

    if not expected_output.exists():
        segs_files = list(ply_path.parent.glob("*.segs.json"))
        if segs_files:
            expected_output = segs_files[0]
        else:
            raise RuntimeError(
                f"Segmentator did not produce expected output {expected_output}"
            )

    logger.info("Superpoints saved to %s", expected_output)
    return expected_output


def _rle_decode(rle: Dict) -> np.ndarray:
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except Exception:
        s = rle["counts"]
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def _decode_instance_masks(mask_entries: List) -> np.ndarray:
    masks = []
    for entry in mask_entries:
        if isinstance(entry, dict):
            masks.append(_rle_decode(entry))
        else:
            if hasattr(entry, "numpy"):
                mask = entry.numpy()
            else:
                mask = np.array(entry)
            if mask.dtype != np.uint8:
                mask = (mask == 1).astype(np.uint8)
            masks.append(mask)
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.stack(masks, axis=0)


def _build_point_labels(masks: np.ndarray, confidences: Optional[np.ndarray]) -> np.ndarray:
    if masks.ndim != 2:
        raise ValueError(f"Expected masks shape (K, N), got {masks.shape}")
    n_instances, n_points = masks.shape
    labels = np.zeros(n_points, dtype=np.int32)
    if confidences is None or len(confidences) != n_instances:
        scores = masks.sum(axis=1)
    else:
        scores = confidences

    order = np.argsort(scores)[::-1]
    next_id = 1
    for idx in order:
        mask = masks[idx].astype(bool)
        assign = mask & (labels == 0)
        if not np.any(assign):
            continue
        labels[assign] = next_id
        next_id += 1

    return labels


def _triangle_labels_from_vertices(
    mesh: o3d.geometry.TriangleMesh, labels: np.ndarray
) -> np.ndarray:
    triangles = np.asarray(mesh.triangles)
    tri_labels = labels[triangles]
    a = tri_labels[:, 0]
    b = tri_labels[:, 1]
    c = tri_labels[:, 2]
    return np.where((a == b) | (a == c), a, np.where(b == c, b, a))


def _render_instance_id_masks(
    mesh: o3d.geometry.TriangleMesh,
    labels: np.ndarray,
    frames: List[Frame],
) -> Optional[Dict[str, np.ndarray]]:
    def _to_legacy_mesh(input_mesh):
        if isinstance(input_mesh, o3d.geometry.TriangleMesh):
            return input_mesh
        if hasattr(input_mesh, "to_legacy"):
            try:
                return input_mesh.to_legacy()
            except Exception:
                pass
        legacy = o3d.geometry.TriangleMesh()
        legacy.vertices = o3d.utility.Vector3dVector(np.asarray(input_mesh.vertices))
        legacy.triangles = o3d.utility.Vector3iVector(np.asarray(input_mesh.triangles))
        return legacy

    def _to_tensor_mesh(input_mesh: o3d.geometry.TriangleMesh) -> "o3d.t.geometry.TriangleMesh":
        return o3d.t.geometry.TriangleMesh.from_legacy(input_mesh)

    def _raycast_instance_id_masks(
        mesh_legacy: o3d.geometry.TriangleMesh,
        tri_labels: np.ndarray,
        frames: List[Frame],
    ) -> Dict[str, np.ndarray]:
        scene = o3d.t.geometry.RaycastingScene()
        tmesh = _to_tensor_mesh(mesh_legacy)
        scene.add_triangles(tmesh)

        result: Dict[str, np.ndarray] = {}
        for frame in frames:
            h, w = frame.h, frame.w
            k = frame.K.cpu().numpy()
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])

            u, v = np.meshgrid(
                np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
            )
            u = u + 0.5
            v = v + 0.5
            dirs_cam = np.stack(
                [(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1
            )
            dirs_cam = dirs_cam.reshape(-1, 3)
            dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

            x_vw = frame.X_VW_opencv.cpu().numpy()
            x_wv = np.linalg.inv(x_vw)
            r = x_wv[:3, :3]
            t = x_wv[:3, 3]
            dirs_world = dirs_cam @ r.T
            origins = np.broadcast_to(t, dirs_world.shape)

            rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
            ans = scene.cast_rays(o3d.core.Tensor(rays))
            prim_ids = ans["primitive_ids"].numpy().reshape(h, w)

            mask = np.zeros((h, w), dtype=np.int32)
            if np.issubdtype(prim_ids.dtype, np.unsignedinteger):
                invalid = np.iinfo(prim_ids.dtype).max
                hit = prim_ids != invalid
            else:
                hit = prim_ids >= 0
            if np.any(hit):
                prim_ids_valid = prim_ids[hit].astype(np.int64)
                mask[hit] = tri_labels[prim_ids_valid]
            result[frame.name] = mask

        return result

    mesh_legacy = _to_legacy_mesh(copy.deepcopy(mesh))
    tri_labels = _triangle_labels_from_vertices(mesh_legacy, labels)
    return _raycast_instance_id_masks(mesh_legacy, tri_labels, frames)


def _get_instance_id_mask_for_frame(
    instance_id: int, masks: Dict[str, np.ndarray], frame: Frame
) -> torch.Tensor:
    frame_mask = masks[frame.name]
    instance_mask = frame_mask == instance_id
    return torch.tensor(instance_mask, device=frame.color.device, dtype=torch.bool)


def determine_table_instance_id(
    frames: List[Frame],
    masks: Dict[str, np.ndarray],
    table_plane: Tuple[float, float, float, float],
    object_ids: np.ndarray,
) -> int:
    instance_ids = object_ids
    if len(instance_ids) == 0:
        return -1

    table_instance_candidates: List[int] = []
    table_instance_counts: List[int] = []

    for frame in frames:
        assert frame.depth is not None
        h, w = frame.depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=frame.depth.device),
            torch.arange(w, device=frame.depth.device),
            indexing="ij",
        )
        valid_mask = frame.depth > 0
        z = frame.depth
        x_world = (x - frame.cx) * z / frame.fl_x
        y_world = (y - frame.cy) * z / frame.fl_y
        points = torch.stack([x_world, y_world, z, torch.ones_like(z)], dim=0)
        points = frame.X_WV_opencv.cuda() @ points.reshape(4, -1)
        points = points.reshape(4, h, w)

        a, b, c, d = table_plane
        plane_dist = (a * points[0] + b * points[1] + c * points[2] + d) / math.sqrt(
            a * a + b * b + c * c
        )
        table_mask = torch.abs(plane_dist) < 0.02
        table_mask = table_mask & valid_mask

        for instance_id in instance_ids:
            inst_id_val = int(instance_id.item()) if hasattr(instance_id, "item") else int(instance_id)
            instance_mask = _get_instance_id_mask_for_frame(inst_id_val, masks, frame)
            instance_mask_valid = instance_mask & valid_mask
            instance_mask_near_table = instance_mask & table_mask
            valid_count = instance_mask_valid.sum()
            if valid_count > 0 and instance_mask_near_table.sum() / valid_count > 0.7:
                table_instance_candidates.append(inst_id_val)
                table_instance_counts.append(instance_mask_near_table.sum().item())

    if len(table_instance_candidates) == 0:
        logger.warning("No table candidates found")
        return -1

    best_idx = int(np.argmax(table_instance_counts))
    return table_instance_candidates[best_idx]


def _sanitize_scene_id(raw_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch if ch in allowed else "_" for ch in raw_id)
    cleaned = cleaned.strip("_")
    return cleaned or "scene"


def _write_scene_files(
    frames: List[Frame],
    scene_id: str,
    mesh: o3d.geometry.TriangleMesh,
    dataset_mode: str,
    data_root: Path,
    split_3d: str,
    split_path_root: Path,
    k_thresh: float,
    seg_min_verts: int,
) -> Tuple[Path, Path, Path, Path, Path]:
    dataset_root = data_root
    if dataset_mode == "scannetpp":
        dataset_name = "Scannetpp"
        dataset_2d_name = "Scannetpp_2D_5interval"
        dataset_3d_name = "Scannetpp_3D"
    else:
        dataset_name = "Scannet200"
        dataset_2d_name = "Scannet200_2D_5interval"
        dataset_3d_name = "Scannet200_3D"

    scene_2d_root = dataset_root / dataset_name / dataset_2d_name / "val" / scene_id
    scene_3d_root = dataset_root / dataset_name / dataset_3d_name / split_3d

    color_dir = scene_2d_root / "color"
    depth_dir = scene_2d_root / "depth"
    pose_dir = scene_2d_root / "pose"
    intrinsic_dir = scene_2d_root / "intrinsic"

    for path in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        path.mkdir(parents=True, exist_ok=True)

    original_ply_dir = scene_3d_root / "original_ply_files"
    superpoints_dir = scene_3d_root / "superpoints"
    groundtruth_dir = scene_3d_root / "groundtruth"

    for path in (original_ply_dir, superpoints_dir, groundtruth_dir):
        path.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise RuntimeError("No frames to export")

    first_k = frames[0].K.cpu().numpy()
    np.savetxt(scene_2d_root / "intrinsic.txt", first_k, fmt="%.8f")

    for idx, frame in enumerate(frames):
        frame_id = f"{idx:05d}"
        color = (frame.color.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        depth = frame.depth.cpu().numpy() if frame.depth is not None else None
        if depth is None:
            raise RuntimeError("Depth is required for Open3DIS pipeline")

        imageio.imwrite(color_dir / f"{frame_id}.jpg", color)

        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        imageio.imwrite(depth_dir / f"{frame_id}.png", depth_mm)

        pose = frame.X_WV_opencv.cpu().numpy()
        np.savetxt(pose_dir / f"{frame_id}.txt", pose, fmt="%.8f")

        k = frame.K.cpu().numpy()
        np.savetxt(intrinsic_dir / f"{frame_id}.txt", k, fmt="%.8f")

    ply_path = original_ply_dir / f"{scene_id}.ply"
    o3d.io.write_triangle_mesh(str(ply_path), mesh)

    segs_json_path = _run_segmentator(ply_path, k_thresh=k_thresh, seg_min_verts=seg_min_verts)
    with open(segs_json_path, "r") as f:
        seg_data = json.load(f)
    seg_ids = np.array(seg_data["segIndices"], dtype=np.int32)
    seg_ids = np.unique(seg_ids, return_inverse=True)[1]

    superpoints_path = superpoints_dir / f"{scene_id}.pth"
    torch.save(seg_ids.astype(np.int32), superpoints_path)

    vertices = np.asarray(mesh.vertices).astype(np.float32)
    colors = np.asarray(mesh.vertex_colors)
    if colors.size == 0:
        colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    else:
        colors = colors.astype(np.float32)
    sem_gt = np.zeros(vertices.shape[0], dtype=np.int32)
    inst_gt = np.zeros(vertices.shape[0], dtype=np.int32)
    groundtruth_path = groundtruth_dir / f"{scene_id}.pth"
    torch.save((vertices, colors, sem_gt, inst_gt), groundtruth_path)

    split_path = split_path_root / "scenes.txt"
    split_path.write_text(f"{scene_id}\n")

    return scene_2d_root.parent, original_ply_dir, superpoints_dir, groundtruth_dir, split_path


def _build_open3dis_config(
    args: Args,
    scene_id: str,
    scene_root_2d: Path,
    original_ply_dir: Path,
    superpoints_dir: Path,
    groundtruth_dir: Path,
    split_path: Path,
    exp_dir: Path,
    img_dim: Tuple[int, int],
    rgb_dim: Tuple[int, int],
    output_dir: Path,
) -> Path:
    project_root = Path(__file__).resolve().parent
    if args.config_template is not None:
        template_path = args.config_template
    else:
        template_path = project_root / "configs" / f"{args.dataset_mode}.yaml"

    cfg = yaml.safe_load(template_path.read_text())

    cfg["data"]["dataset_name"] = args.dataset_mode
    cfg["data"]["split_path"] = str(split_path)
    cfg["data"]["datapath"] = str(scene_root_2d)
    cfg["data"]["gt_pth"] = str(groundtruth_dir)
    cfg["data"]["original_ply"] = str(original_ply_dir)
    cfg["data"]["spp_path"] = str(superpoints_dir)
    cfg["data"]["img_dim"] = [int(img_dim[0]), int(img_dim[1])]
    cfg["data"]["rgb_img_dim"] = [int(rgb_dim[0]), int(rgb_dim[1])]
    cfg["data"]["img_interval"] = int(args.img_interval)
    cfg["data"]["dataset_name"] = "scannetpp"

    if args.cls_agnostic_3d_proposals_path is not None:
        cfg["data"]["cls_agnostic_3d_proposals_path"] = str(
            args.cls_agnostic_3d_proposals_path
        )
    if args.dc_features_path is not None:
        cfg["data"]["dc_features_path"] = str(args.dc_features_path)

    cfg["exp"]["save_dir"] = str(exp_dir)
    cfg["exp"]["exp_name"] = args.exp_name
    cfg["exp"]["mask2d_output"] = args.mask2d_output

    cfg["final_instance"]["spp_level"] = bool(args.use_superpoints)
    cfg["proposals"]["p2d"] = True
    cfg["proposals"]["p3d"] = bool(args.use_3d_proposals)
    cfg["proposals"]["agnostic"] = True

    if args.use_3d_proposals:
        cfg["cluster"]["simi"] = max(cfg["cluster"].get("simi", 0.0), 0.9)

    def _abs_path(path_value: str) -> str:
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        return str((project_root / path).resolve())

    if "foundation_model" in cfg:
        for key in (
            "ram_checkpoint",
            "grounded_config_file",
            "grounded_checkpoint",
            "yoloworld_config_file",
            "yoloworld_checkpoint",
            "sam_checkpoint",
        ):
            if key in cfg["foundation_model"]:
                cfg["foundation_model"][key] = _abs_path(cfg["foundation_model"][key])

    config_path = output_dir / "open3dis_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return config_path


def _run_open3dis_pipeline(
    project_root: Path, config_path: Path, work_dir: Path, debug_dir: Path
) -> None:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    def _run(cmd: List[str]) -> None:
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(work_dir), env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Open3DIS command failed: {' '.join(cmd)}")

    tracker_path = debug_dir / "tracker_2d.txt"
    if not tracker_path.exists():
        tracker_path.write_text("")
    env["TRACKER_2D_PATH"] = str(tracker_path)

    _run(
        [
            "python3",
            str((project_root / "tools" / "grounding_2d.py").resolve()),
            "--config",
            str(config_path),
        ]
    )
    _run(
        [
            "python3",
            str((project_root / "tools" / "generate_3d_inst.py").resolve()),
            "--config",
            str(config_path),
        ]
    )


def _ensure_clip_weights(config_path: Path) -> None:
    cfg = Munch.fromDict(yaml.safe_load(config_path.read_text()))
    clip_model = None
    if hasattr(cfg, "foundation_model"):
        clip_model = cfg.foundation_model.get("clip_model")
    if not clip_model:
        logger.info("No CLIP model specified in config; skipping CLIP download")
        return

    try:
        import clip  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenAI CLIP not installed; install openai-clip") from exc

    logger.info("Ensuring CLIP weights for model %s", clip_model)
    model, _ = clip.load(clip_model, device="cpu")
    del model


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    logger.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _ensure_foundation_checkpoints(config_path: Path, project_root: Path) -> None:
    cfg = Munch.fromDict(yaml.safe_load(config_path.read_text()))
    if not hasattr(cfg, "foundation_model"):
        return

    cert = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cert)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert)

    base_dir = project_root / "pretrains" / "foundation_models"
    base_dir.mkdir(parents=True, exist_ok=True)

    sam_ckpt = Path(cfg.foundation_model.get("sam_checkpoint"))
    if not sam_ckpt.is_absolute():
        sam_ckpt = (project_root / sam_ckpt).resolve()
    if not sam_ckpt.exists():
        sam_ckpt = (project_root / "pretrains" / "foundation_models" / SAM_CHECKPOINT_FILENAME).resolve()
        _download_file(SAM_CHECKPOINT_URL, sam_ckpt)
        cfg.foundation_model["sam_checkpoint"] = str(sam_ckpt)

    gdino_ckpt = Path(cfg.foundation_model.get("grounded_checkpoint"))
    if not gdino_ckpt.is_absolute():
        gdino_ckpt = (project_root / gdino_ckpt).resolve()
    if not gdino_ckpt.exists():
        _download_file(GROUNDING_DINO_URL, gdino_ckpt)

    ram_ckpt = Path(cfg.foundation_model.get("ram_checkpoint"))
    if not ram_ckpt.is_absolute():
        ram_ckpt = (project_root / ram_ckpt).resolve()
    if not ram_ckpt.exists():
        ram_ckpt = (project_root / "pretrains" / "foundation_models" / "ram_plus_swin_large_14m.pth").resolve()
        _download_file(RAM_PLUS_URL, ram_ckpt)
        cfg.foundation_model["ram_checkpoint"] = str(ram_ckpt)

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _ensure_nltk_data(output_dir: Path) -> None:
    try:
        import nltk  # type: ignore
    except Exception as exc:
        raise RuntimeError("nltk not installed; install nltk") from exc

    download_dir = output_dir / "nltk_data"
    download_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.append(str(download_dir))

    for resource in ("punkt", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, download_dir=str(download_dir), quiet=True)


def _build_isbnet_config(
    project_root: Path,
    args: Args,
    dataset_mode: str,
    groundtruth_dir: Path,
    output_dir: Path,
) -> Path:
    if args.isbnet_config is not None:
        template_path = args.isbnet_config
    else:
        if dataset_mode == "scannetpp":
            template_path = project_root / "segmenter3d" / "ISBNet" / "configs" / "scannetpp" / "isbnet_scannetpp.yaml"
        else:
            template_path = project_root / "segmenter3d" / "ISBNet" / "configs" / "scannet200" / "isbnet_scannet200.yaml"

    cfg = yaml.safe_load(template_path.read_text())

    data_root = str(groundtruth_dir.parent)
    cfg["data"]["train"]["data_root"] = data_root
    cfg["data"]["test"]["data_root"] = data_root
    cfg["data"]["train"]["prefix"] = "groundtruth"
    cfg["data"]["test"]["prefix"] = "groundtruth"
    cfg["data"]["train"]["suffix"] = ".pth"
    cfg["data"]["test"]["suffix"] = ".pth"
    cfg["data"]["train"]["type"] = dataset_mode
    cfg["data"]["test"]["type"] = dataset_mode

    cfg_path = output_dir / "isbnet_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _run_isbnet_backbone(
    project_root: Path,
    args: Args,
    dataset_mode: str,
    groundtruth_dir: Path,
    output_dir: Path,
    scene_id: str,
) -> Tuple[Path, Path]:
    isbnet_root = project_root / "segmenter3d" / "ISBNet"
    if args.isbnet_checkpoint is None:
        ckpt_dir = project_root / "pretrains" / "isbnet"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / ISBNET_SCANNET200_FILENAME
        if not ckpt_path.exists():
            logger.info("Downloading ISBNet checkpoint (ScanNet200) to %s", ckpt_path)
            gdown.download(
                id=ISBNET_SCANNET200_FILE_ID,
                output=str(ckpt_path),
                quiet=False,
            )
        args.isbnet_checkpoint = ckpt_path

    isbnet_root = project_root / "segmenter3d" / "ISBNet"
    cfg_path = _build_isbnet_config(
        project_root=project_root,
        args=args,
        dataset_mode=dataset_mode,
        groundtruth_dir=groundtruth_dir,
        output_dir=output_dir,
    )

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONPATH"] = f"{isbnet_root}:{env.get('PYTHONPATH', '')}"

    def _run(cmd: List[str]) -> None:
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(isbnet_root), env=env)
        if result.returncode != 0:
            raise RuntimeError(f"ISBNet command failed: {' '.join(cmd)}")

    data_link = project_root / "data"
    if data_link.is_symlink():
        target = data_link.readlink()
        if not target.is_absolute():
            target = (data_link.parent / target).resolve()
        target.mkdir(parents=True, exist_ok=True)
    else:
        data_link.mkdir(parents=True, exist_ok=True)

    if dataset_mode == "scannetpp":
        dc_features_path = project_root / "data" / "Scannetpp" / "Scannetpp_3D" / "test" / "dc_feat_scannetpp"
        proposals_path = project_root / "data" / "Scannetpp" / "Scannetpp_3D" / "test" / "isbnet_clsagnostic_scannetpp"
    else:
        dc_features_path = project_root / "data" / "Scannet200" / "Scannet200_3D" / "val" / "dc_feat_scannet200"
        proposals_path = project_root / "data" / "Scannet200" / "Scannet200_3D" / "val" / "isbnet_clsagnostic_scannet200"

    dc_features_path.mkdir(parents=True, exist_ok=True)
    proposals_path.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python3",
            "tools/test.py",
            str(cfg_path),
            str(args.isbnet_checkpoint),
        ]
    )

    if not (dc_features_path / f"{scene_id}.pth").exists():
        raise RuntimeError(f"ISBNet did not produce dc_features for {scene_id}")
    if not (proposals_path / f"{scene_id}.pth").exists():
        raise RuntimeError(f"ISBNet did not produce 3D proposals for {scene_id}")

    return dc_features_path, proposals_path


def _load_open3dis_instances(
    cluster_path: Path,
    use_3d_proposals: bool,
    proposals_3d_path: Optional[Path],
    scene_id: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    cluster_dict = torch.load(cluster_path)
    masks_2d = _decode_instance_masks(cluster_dict["ins"])
    conf_2d = cluster_dict.get("conf", None)
    if conf_2d is not None:
        if hasattr(conf_2d, "cpu"):
            conf_2d = conf_2d.cpu().numpy()
        else:
            conf_2d = np.array(conf_2d)

    if not use_3d_proposals:
        return masks_2d, conf_2d

    if proposals_3d_path is None:
        raise RuntimeError("use_3d_proposals requested but no proposals path provided")

    proposal_file = proposals_3d_path / f"{scene_id}.pth"
    if not proposal_file.exists():
        raise RuntimeError(f"3D proposals file missing: {proposal_file}")

    proposal_dict = torch.load(proposal_file)
    masks_3d = _decode_instance_masks(proposal_dict["ins"])
    conf_3d = proposal_dict.get("conf", None)
    if conf_3d is not None:
        if hasattr(conf_3d, "cpu"):
            conf_3d = conf_3d.cpu().numpy()
        else:
            conf_3d = np.array(conf_3d)

    if masks_2d.size == 0:
        return masks_3d, conf_3d
    if masks_3d.size == 0:
        return masks_2d, conf_2d

    masks = np.concatenate([masks_2d, masks_3d], axis=0)
    if conf_2d is None and conf_3d is None:
        conf = None
    else:
        conf_2d = conf_2d if conf_2d is not None else np.zeros(masks_2d.shape[0])
        conf_3d = conf_3d if conf_3d is not None else np.zeros(masks_3d.shape[0])
        conf = np.concatenate([conf_2d, conf_3d], axis=0)
    return masks, conf


def run() -> None:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))
    logger.addHandler(ch)

    args = tyro.cli(Args)

    with contextlib.redirect_stdout(sys.stderr):
        logger.info("--------------")
        logger.info("Starting Open3DIS initialization")
        logger.info("params: %s", args)

        logger.info("Loading observations from %s", args.observations_path)
        dataset: Observations = Observations.load(args.observations_path)
        logger.info("Observations loaded.")

        logger.info("Loading scene setup from %s", args.scene_path)
        scene = SceneSetup.load(args.scene_path)
        logger.info("Scene loaded.")

        if dataset.id is None:
            logger.info("Dataset has no id, using transient id")
            dataset.id = f"transient_{time.strftime('%Y%m%d-%H%M%S')}"


        scene_id = _sanitize_scene_id(dataset.id)
        project_root = Path(__file__).resolve().parent
        if args.output_dir is None:
            output_dir = project_root / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{scene_id}"
        else:
            output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        debug_dir = output_dir / "debug"
        dbg = DebugVisualizer(debug_dir)

        logger.info("Converting %d observation frames", len(dataset.frames))
        frames = [get_dataset_frame_from_observation_frame(f) for f in dataset.frames]
        dbg.save_frames(frames)

        logger.info("Reconstructing TSDF mesh")
        mesh = _extract_mesh_bounded_with_res(frames, depth_trunc=2, mesh_res=1024)
        logger.info("Mesh has %d vertices, %d triangles", len(mesh.vertices), len(mesh.triangles))
        dbg.save_mesh(mesh)

        workspace_voxels = get_workspace_voxels(scene)
        dbg.save_workspace_voxels(workspace_voxels)
        mesh = _crop_mesh_to_workspace_bbox(mesh, workspace_voxels)
        logger.info("Cropped mesh has %d vertices, %d triangles", len(mesh.vertices), len(mesh.triangles))
        dbg.save_mesh(mesh, filename="mesh_tsdf_cropped.ply")

        data_root = project_root / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        split_3d = "test"
        scene_root_2d, original_ply_dir, superpoints_dir, groundtruth_dir, split_path = _write_scene_files(
            frames=frames,
            scene_id=scene_id,
            mesh=mesh,
            dataset_mode=args.dataset_mode,
            data_root=data_root,
            split_3d=split_3d,
            split_path_root=output_dir,
            k_thresh=args.k_thresh,
            seg_min_verts=args.seg_min_verts,
        )
        if dbg.enabled:
            superpoints_path = superpoints_dir / f"{scene_id}.pth"
            if superpoints_path.exists():
                superpoints = torch.load(superpoints_path)
                if hasattr(superpoints, "cpu"):
                    superpoints = superpoints.cpu().numpy()
                else:
                    superpoints = np.array(superpoints)
                dbg.save_superpoints(mesh, superpoints)

        exp_dir = output_dir / "open3dis_exp"
        exp_dir.mkdir(parents=True, exist_ok=True)

        img_dim = (frames[0].w, frames[0].h)
        rgb_dim = (frames[0].w, frames[0].h)
        config_path = _build_open3dis_config(
            args,
            scene_id=scene_id,
            scene_root_2d=scene_root_2d,
            original_ply_dir=original_ply_dir,
            superpoints_dir=superpoints_dir,
            groundtruth_dir=groundtruth_dir,
            split_path=split_path,
            exp_dir=exp_dir,
            img_dim=img_dim,
            rgb_dim=rgb_dim,
            output_dir=output_dir,
        )

        if args.use_3d_proposals or args.use_superpoints:
            dc_features_path = args.dc_features_path
            proposals_3d_path_override = args.cls_agnostic_3d_proposals_path
            needs_dc = args.use_superpoints and dc_features_path is None
            needs_props = args.use_3d_proposals and proposals_3d_path_override is None
            if needs_dc or needs_props:
                logger.info("Running ISBNet to produce dc_features and 3D proposals")
                dc_features_path, proposals_3d_path_override = _run_isbnet_backbone(
                    project_root=project_root,
                    args=args,
                    dataset_mode=args.dataset_mode,
                    groundtruth_dir=groundtruth_dir,
                    output_dir=output_dir,
                    scene_id=scene_id,
                )
                args.dc_features_path = dc_features_path
                args.cls_agnostic_3d_proposals_path = proposals_3d_path_override

                config_path = _build_open3dis_config(
                    args,
                    scene_id=scene_id,
                    scene_root_2d=scene_root_2d,
                    original_ply_dir=original_ply_dir,
                    superpoints_dir=superpoints_dir,
                    groundtruth_dir=groundtruth_dir,
                    split_path=split_path,
                    exp_dir=exp_dir,
                    img_dim=img_dim,
                    rgb_dim=rgb_dim,
                    output_dir=output_dir,
                )

        _ensure_clip_weights(config_path)
        _ensure_nltk_data(output_dir)
        _ensure_foundation_checkpoints(config_path, project_root)

        logger.info("Running Open3DIS pipeline")
        _run_open3dis_pipeline(
            project_root=project_root,
            config_path=config_path,
            work_dir=output_dir,
            debug_dir=debug_dir,
        )

        cfg = Munch.fromDict(yaml.safe_load(config_path.read_text()))
        if dbg.enabled:
            mask2d_path = Path(cfg.exp.save_dir) / cfg.exp.exp_name / cfg.exp.mask2d_output / f"{scene_id}.pth"
            if mask2d_path.exists():
                mask_data = torch.load(mask2d_path)
                dbg.save_masks_2d(frames, mask_data)
        cluster_path = (
            Path(cfg.exp.save_dir)
            / cfg.exp.exp_name
            / cfg.exp.clustering_3d_output
            / f"{scene_id}.pth"
        )
        if not cluster_path.exists():
            raise RuntimeError(f"Open3DIS output missing: {cluster_path}")

        proposals_3d_path = (
            args.cls_agnostic_3d_proposals_path
            if args.cls_agnostic_3d_proposals_path is not None
            else Path(cfg.data.cls_agnostic_3d_proposals_path)
            if hasattr(cfg.data, "cls_agnostic_3d_proposals_path")
            else None
        )

        masks, conf = _load_open3dis_instances(
            cluster_path=cluster_path,
            use_3d_proposals=args.use_3d_proposals,
            proposals_3d_path=proposals_3d_path,
            scene_id=scene_id,
        )

        if masks.size == 0:
            raise RuntimeError("Open3DIS produced no instance masks")

        labels = _build_point_labels(masks, conf)
        if dbg.enabled:
            dbg.save_segmented_mesh(mesh, labels)
        mesh_vertices = np.asarray(mesh.vertices).astype(np.float32)
        labels = _filter_labels_by_workspace(mesh_vertices, labels, workspace_voxels)

        instance_groups = _render_instance_id_masks(mesh, labels, frames)
        if instance_groups is None:
            raise RuntimeError("Instance mask rendering failed")

        all_label_ids = np.unique(labels)
        all_label_ids = all_label_ids[all_label_ids > 0]

        frame_counts = {lbl: 0 for lbl in all_label_ids}
        for _, mask in instance_groups.items():
            for lbl in all_label_ids:
                if np.any(mask == lbl):
                    frame_counts[lbl] += 1

        valid_ids = np.array([lbl for lbl, cnt in frame_counts.items() if cnt >= 3])
        logger.info("Labels in >= 3 frames: %d / %d", len(valid_ids), len(all_label_ids))

        for name in instance_groups:
            instance_groups[name][~np.isin(instance_groups[name], valid_ids)] = 0

        table_id = determine_table_instance_id(frames, instance_groups, scene.ground_plane, valid_ids)
        logger.info("Table instance id: %d", table_id)
        if table_id > 0:
            valid_ids = valid_ids[valid_ids != table_id]
            for name in instance_groups:
                instance_groups[name][instance_groups[name] == table_id] = 0

        vertex_labels_filtered = labels.copy()
        vertex_labels_filtered[~np.isin(vertex_labels_filtered, valid_ids)] = 0
        if dbg.enabled:
            dbg.save_filtered_mesh(mesh, labels, vertex_labels_filtered, table_id, valid_ids)

        frame_ids: List[int] = []
        pixel_masks: List[np.ndarray] = []
        for obs_frame in dataset.frames:
            frame_ids.append(obs_frame.id)
            mask = instance_groups.get(
                obs_frame.name,
                np.zeros((frames[0].h, frames[0].w), dtype=np.int32),
            )
            pixel_masks.append(mask)

        instance_mask_objects = InstanceMaskObjectsDef(
            frame_ids=frame_ids,
            pixel_object_ids=pixel_masks,
        )

        objects = ObjectSegmentations(object_segmentations=instance_mask_objects)
        if dbg.enabled:
            dbg.save_pixel_masks(frames, instance_groups)
        output_path = output_dir / "objectsdef.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        objects.save(output_path)

        logger.info("Objects saved to %s", output_path)

    print(f"objects_path: {output_path}")


if __name__ == "__main__":
    run()
