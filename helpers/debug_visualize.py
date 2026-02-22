"""Debug visualization utilities for the Open3DIS external segmenter.

Gated by the OPEN3DIS_DEBUG environment variable. When OPEN3DIS_DEBUG=1,
checkpoints save artifacts (images, PLY meshes, numpy arrays) to a debug
output directory. When disabled, all methods are no-ops with minimal overhead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("open3dis-segmenter")


def is_debug_enabled() -> bool:
    return os.environ.get("OPEN3DIS_DEBUG", "0") == "1"


def _random_colors_for_labels(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    labels_int = labels.astype(np.int64)
    colors = np.empty((labels_int.shape[0], 3), dtype=np.float64)

    bg_mask = labels_int <= 0
    colors[bg_mask] = np.array([0.9, 0.9, 0.9])

    fg = labels_int[~bg_mask]
    if fg.size:
        hashed = (fg * 2654435761 + seed) & 0xFFFFFFFF
        r = ((hashed >> 16) & 0xFF) / 255.0
        g = ((hashed >> 8) & 0xFF) / 255.0
        b = (hashed & 0xFF) / 255.0
        colors[~bg_mask] = np.stack([r, g, b], axis=-1)

    return colors


def _scale_depth(depth: np.ndarray) -> np.ndarray:
    if depth.size == 0:
        return depth
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros_like(depth, dtype=np.uint8)
    lo = float(np.percentile(valid, 1))
    hi = float(np.percentile(valid, 99))
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.uint8)
    scaled = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


class DebugVisualizer:
    """Saves debug artifacts for each pipeline stage when OPEN3DIS_DEBUG=1."""

    def __init__(self, output_dir: Path):
        self.enabled = is_debug_enabled()
        self.output_dir = output_dir
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug visualizer ENABLED -- output dir: %s", self.output_dir)
        else:
            logger.debug("Debug visualizer disabled (set OPEN3DIS_DEBUG=1 to enable)")

    def save_frames(self, frames, max_frames: int = 5) -> None:
        if not self.enabled:
            return
        import imageio.v2 as imageio

        out = self.output_dir / "frames"
        out.mkdir(exist_ok=True)

        for frame in frames[:max_frames]:
            rgb = frame.color.cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(out / f"{frame.name}_rgb.png", rgb_uint8)

            if frame.depth is not None:
                depth = frame.depth.cpu().numpy()
                depth_vis = _scale_depth(depth)
                imageio.imwrite(out / f"{frame.name}_depth.png", depth_vis)

        logger.info("[DEBUG] Saved %d frame visualizations to %s", min(len(frames), max_frames), out)

    def save_mesh(self, mesh, filename: str = "mesh_tsdf.ply") -> None:
        if not self.enabled:
            return
        import open3d as o3d

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh)
        logger.info(
            "[DEBUG] Saved mesh: %d verts, %d tris -> %s",
            len(mesh.vertices),
            len(mesh.triangles),
            path,
        )

    def save_workspace_voxels(self, workspace_voxels, filename: str = "workspace_voxels.ply") -> None:
        if not self.enabled:
            return
        import open3d as o3d

        voxels = workspace_voxels.get_voxels()
        origin = workspace_voxels.origin
        vs = workspace_voxels.voxel_size
        centers = np.array([origin + np.asarray(v.grid_index) * vs + vs / 2 for v in voxels])

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
        pcd.paint_uniform_color([0.2, 0.7, 0.3])

        path = self.output_dir / filename
        o3d.io.write_point_cloud(str(path), pcd)
        logger.info("[DEBUG] Saved workspace voxels (%d voxels) -> %s", len(voxels), path)

    def save_superpoints(self, mesh, seg_ids: np.ndarray, filename: str = "mesh_superpoints.ply") -> None:
        if not self.enabled:
            return
        import copy
        import open3d as o3d

        mesh_copy = copy.deepcopy(mesh)
        colors = _random_colors_for_labels(seg_ids, seed=123)
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh_copy)

        unique_sp = np.unique(seg_ids)
        sizes = np.array([np.sum(seg_ids == s) for s in unique_sp])
        logger.info(
            "[DEBUG] Superpoints: %d unique, size min=%d median=%d max=%d -> %s",
            len(unique_sp),
            int(sizes.min()) if sizes.size else 0,
            int(np.median(sizes)) if sizes.size else 0,
            int(sizes.max()) if sizes.size else 0,
            path,
        )

    def save_masks_2d(self, frames, mask_dict: Dict[str, Dict], max_frames: int = 6) -> None:
        if not self.enabled:
            return
        import imageio.v2 as imageio
        import pycocotools.mask

        out = self.output_dir / "masks_2d"
        out.mkdir(exist_ok=True)

        saved = 0
        for idx, frame in enumerate(frames):
            if saved >= max_frames:
                break
            entry = None
            candidates = [
                frame.name,
                f"{idx:05d}",
                str(idx),
                f"frame_{idx}",
                f"frame_{idx * 10}",
            ]
            for key in candidates:
                if key in mask_dict:
                    entry = mask_dict[key]
                    break
            if entry is None and isinstance(mask_dict, dict) and "masks" in mask_dict:
                maybe_list = mask_dict["masks"]
                if isinstance(maybe_list, (list, tuple)) and idx < len(maybe_list):
                    entry = maybe_list[idx]

            if entry is None:
                continue

            if isinstance(entry, dict):
                masks = entry.get("masks")
            else:
                masks = entry
            if masks is None:
                continue

            instance_map = np.zeros((frame.h, frame.w), dtype=np.int32)
            for inst_id, mask_rle in enumerate(masks, start=1):
                mask = pycocotools.mask.decode(mask_rle).astype(bool)
                instance_map[mask] = inst_id

            colors = _random_colors_for_labels(instance_map.ravel(), seed=42 + saved).reshape(instance_map.shape + (3,))
            colors_uint8 = (colors * 255).astype(np.uint8)
            rgb = frame.color.cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            imageio.imwrite(out / f"{frame.name}_rgb.png", rgb_uint8)
            imageio.imwrite(out / f"{frame.name}_masks.png", colors_uint8)
            saved += 1

        logger.info("[DEBUG] Saved %d 2D mask visualizations -> %s", saved, out)

    def save_segmented_mesh(self, mesh, vertex_labels: np.ndarray, filename: str = "mesh_segmented.ply") -> None:
        if not self.enabled:
            return
        import copy
        import open3d as o3d

        mesh_copy = copy.deepcopy(mesh)
        colors = _random_colors_for_labels(vertex_labels, seed=77)
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh_copy)

        n_labels = len(np.unique(vertex_labels[vertex_labels > 0]))
        logger.info("[DEBUG] Segmented mesh: %d instances -> %s", n_labels, path)

    def save_filtered_mesh(
        self,
        mesh,
        vertex_labels_before: np.ndarray,
        vertex_labels_after: np.ndarray,
        table_id: int,
        valid_ids: np.ndarray,
        filename: str = "mesh_filtered.ply",
    ) -> None:
        if not self.enabled:
            return
        import copy
        import open3d as o3d

        mesh_copy = copy.deepcopy(mesh)
        colors = _random_colors_for_labels(vertex_labels_after, seed=77)
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh_copy)

        before_ids = set(np.unique(vertex_labels_before)) - {0}
        after_ids = set(np.unique(vertex_labels_after)) - {0}
        removed = before_ids - after_ids
        logger.info(
            "[DEBUG] Filtering: %d -> %d instances (removed %s, table_id=%d) -> %s",
            len(before_ids),
            len(after_ids),
            removed if removed else "none",
            table_id,
            path,
        )

    def save_pixel_masks(self, frames, instance_groups: Dict[str, np.ndarray], max_frames: int = 6) -> None:
        if not self.enabled:
            return
        import imageio.v2 as imageio

        out = self.output_dir / "pixel_masks"
        out.mkdir(exist_ok=True)

        saved = 0
        for frame in frames:
            if saved >= max_frames:
                break
            mask = instance_groups.get(frame.name)
            if mask is None:
                continue

            colors = _random_colors_for_labels(mask.ravel(), seed=42).reshape(mask.shape + (3,))
            colors_uint8 = (colors * 255).astype(np.uint8)
            rgb = frame.color.cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            imageio.imwrite(out / f"{frame.name}_rgb.png", rgb_uint8)
            imageio.imwrite(out / f"{frame.name}_pixel_masks.png", colors_uint8)
            saved += 1

        logger.info("[DEBUG] Saved %d back-projected pixel mask visualizations -> %s", saved, out)
