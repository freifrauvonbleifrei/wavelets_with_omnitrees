#!/usr/bin/env python3
"""Compare OpenVDB cloud model against omnitree wavelet compression.

Loads a VDB file, determines the grid resolution from the file, builds an
adaptive omnitree (refined only where VDB has data), and computes compression
statistics by analyzing wavelet coefficients at adjustable error bounds.

Usage:
    python compare_cloud.py wdas_cloud.vdb --threshold 0.0 0.01 0.1
    python compare_cloud.py wdas_cloud.vdb --grid density --max-level 8
"""

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import bitarray as ba
import numpy as np

import openvdb as vdb

import dyada
import dyada.descriptor
import dyada.discretization
import dyada.linearization

try:
    from wavelets_with_omnitrees.dyada_cache_patch import install as _install_cache
except ModuleNotFoundError:
    from dyada_cache_patch import install as _install_cache  # type: ignore
_install_cache()

try:
    from wavelets_with_omnitrees.wavelets_with_omnitrees import (
        transform_to_all_wavelet_coefficients,
        get_leaf_scalings,
        _compute_node_levels,
        compress_by_omnitree_coarsening,
        compress_by_downsplit_coarsening,
        omnitree_from_vdb,
        read_vdb_grid,
    )
except ModuleNotFoundError:
    from wavelets_with_omnitrees import (  # type: ignore
        transform_to_all_wavelet_coefficients,
        get_leaf_scalings,
        _compute_node_levels,
        compress_by_omnitree_coarsening,
        compress_by_downsplit_coarsening,
        omnitree_from_vdb,
        read_vdb_grid,
    )


# ── VDB helpers ─────────────────────────────────────────────────────────────


def vdb_grid_extent(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bbox_min, bbox_max_exclusive, extent_per_dim) in index space."""
    bbox_min, bbox_max = grid.evalActiveVoxelBoundingBox()
    bbox_min = np.array(bbox_min, dtype=np.int64)
    bbox_max = np.array(bbox_max, dtype=np.int64) + 1  # exclusive
    extent = bbox_max - bbox_min
    return bbox_min, bbox_max, extent


def levels_from_extent(extent: np.ndarray) -> np.ndarray:
    """Compute per-dimension level = ceil(log2(extent)) for power-of-2 padding."""
    return np.array([int(math.ceil(math.log2(max(e, 1)))) for e in extent])


_LEAF_3D = ba.frozenbitarray("000")
_FULL_SPLIT_3D = ba.frozenbitarray("111")


# ── Stats ──────────────────────────────────────────────────────────────────


def descriptor_stats(desc: dyada.descriptor.RefinementDescriptor) -> dict:
    """Compute topology stats for an omnitree descriptor."""
    d = desc.get_num_dimensions()
    n_nodes = len(desc)
    n_boxes = desc.get_num_boxes()
    topo_bits = n_nodes * d
    ref_counts = Counter(ref.to01() for ref in desc)
    return {
        "nodes": n_nodes,
        "boxes": n_boxes,
        "topo_bits": topo_bits,
        "ref_counts": dict(ref_counts),
    }


def vdb_topology_stats(vdb_path: str, grid_name: str | None = None) -> dict:
    """Get VDB topology stats using the C++ tool if available."""
    tool = Path(__file__).parent / "vdb_topology_bits"
    if tool.exists():
        cmd = [str(tool), vdb_path]
        if grid_name:
            cmd.append(grid_name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            stats = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "summary" in obj:
                    stats["total_topology_bits"] = obj["total_topology_bits"]
                    stats["total_topology_bytes"] = obj["total_topology_bytes"]
                elif "value_summary" in obj:
                    stats["active_leaf_voxels"] = obj["active_leaf_voxels"]
                    stats["total_coefficients"] = obj.get(
                        "total_coefficients",
                        obj.get("total_leaf_coefficients", 0),
                    )
            return stats
    return {}


def vdb_topology_stats_python(grid) -> dict:
    """Compute VDB topology stats from the Python grid object."""
    active_voxels = grid.activeVoxelCount()
    leaf_count = grid.leafCount()
    non_leaf_count = grid.nonLeafCount()
    log2_dims = grid.nodeLog2Dims()
    mem_usage = grid.memUsage()

    # OpenVDB tree structure (for FloatGrid): Root -> Internal5 -> Internal4 -> Leaf3
    # Leaf: 8^3=512 voxels, value mask = 512 bits
    # Internal4: 16^3=4096 children, child mask + value mask = 2*4096 = 8192 bits
    # Internal5: 32^3=32768 children, child mask + value mask = 2*32768 = 65536 bits
    # We approximate topology bits from node counts
    leaf_bits = leaf_count * 512  # value mask per leaf
    # non_leaf_count includes both Internal4 and Internal5 nodes
    # We can't distinguish them from Python API alone, so estimate
    internal_bits = non_leaf_count * (8192 + 65536) // 2  # rough average

    return {
        "active_voxels": active_voxels,
        "leaf_count": leaf_count,
        "non_leaf_count": non_leaf_count,
        "leaf_topo_bits": leaf_bits,
        "internal_topo_bits_approx": internal_bits,
        "total_topo_bits_approx": leaf_bits + internal_bits,
        "mem_usage_bytes": mem_usage,
        "node_log2_dims": log2_dims,
    }


# ── Wavelet-based compression analysis ────────────────────────────────────


def analyze_wavelet_compression(
    discretization: dyada.discretization.Discretization,
    coefficients: list,
    thresholds: list[float],
) -> dict[float, dict]:
    """Analyze wavelet compression at multiple thresholds without restructuring.

    For each threshold, walks the tree bottom-up and counts how many subtrees
    can be coarsened (all wavelet detail coefficients within local error bound).
    Returns stats per threshold.
    """
    descriptor = discretization.descriptor
    num_dimensions = descriptor.get_num_dimensions()
    n = len(descriptor)

    # Compute node levels as a 2D array for vectorized access
    print(f"  Computing node levels for {n:,} nodes...", end="", flush=True)
    levels_2d = np.zeros((n, num_dimensions), dtype=np.int8)
    current_level = np.zeros(num_dimensions, dtype=np.int8)
    level_stack: list = []
    for i in range(n):
        ref = descriptor[i]
        levels_2d[i] = current_level
        if ref.count() > 0:
            child_level = current_level.copy()
            for d in range(num_dimensions):
                if ref[d]:
                    child_level[d] += 1
            level_stack.append([1 << ref.count(), current_level])
            current_level = child_level
        else:
            while level_stack:
                level_stack[-1][0] -= 1
                if level_stack[-1][0] > 0:
                    break
                current_level = level_stack[-1][1]
                level_stack.pop()
    print(" done.")

    # Precompute per-node box volumes (only for inner nodes)
    print(f"  Computing box volumes...", end="", flush=True)
    box_volumes = np.prod(np.power(2.0, -levels_2d.astype(np.float64)), axis=1)
    print(" done.")

    # Precompute detail coefficient sums for inner nodes
    print(f"  Computing detail coefficient sums...", end="", flush=True)
    detail_sums = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if len(coefficients[i]) > 1:
            detail_sums[i] = sum(abs(v) for v in coefficients[i][1:])
    print(" done.")

    results = {}
    for threshold in thresholds:
        print(f"  Threshold {threshold}...", end="", flush=True)
        # Single bottom-up pass: detect coarsenable nodes and count survivors
        coarsenable = set()
        sub_nodes = np.ones(n, dtype=np.int64)
        sub_boxes = np.ones(n, dtype=np.int64)

        desc_i = n - 1
        stack: list[int] = []
        coarsenable_count = 0
        for ref in reversed(descriptor):
            num_ref = ref.count()
            if num_ref == 0:
                stack.append(desc_i)
                desc_i -= 1
                continue

            num_children = 1 << num_ref
            child_indices = [stack.pop() for _ in range(num_children)]

            all_children_simple = all(
                descriptor[c].count() == 0 or c in coarsenable
                for c in child_indices
            )

            can_coarsen = False
            if all_children_simple and len(coefficients[desc_i]) > 1:
                local_bound = threshold * box_volumes[desc_i]
                if detail_sums[desc_i] <= local_bound:
                    can_coarsen = True
                    coarsenable.add(desc_i)
                    coarsenable_count += 1

            if can_coarsen:
                sub_nodes[desc_i] = 1
                sub_boxes[desc_i] = 1
            else:
                sub_nodes[desc_i] = 1 + sum(int(sub_nodes[c]) for c in child_indices)
                sub_boxes[desc_i] = sum(int(sub_boxes[c]) for c in child_indices)

            stack.append(desc_i)
            desc_i -= 1

        results[threshold] = {
            "nodes": int(sub_nodes[0]),
            "boxes": int(sub_boxes[0]),
            "topo_bits": int(sub_nodes[0]) * num_dimensions,
            "coarsenable_nodes": coarsenable_count,
        }
        print(f" {coarsenable_count:,} coarsenable")

    return results


def _build_subtree_end(descriptor) -> np.ndarray:
    """Precompute subtree_end[i] = first index after the subtree rooted at i.

    For leaves, subtree_end[i] = i+1.
    For inner nodes, subtree_end[i] = subtree_end[last_child].
    Computed in a single backward pass.
    """
    n = len(descriptor)
    subtree_end = np.empty(n, dtype=np.int64)
    i = n - 1
    while i >= 0:
        ref = descriptor[i]
        if ref.count() == 0:
            subtree_end[i] = i + 1
        else:
            # The last child of i is at subtree_end[i] - 1 (determined by
            # walking forward), but we can compute it by scanning children.
            # Children of i are at i+1, then subtree_end[i+1], etc.
            child_start = i + 1
            for _ in range(1 << ref.count()):
                child_start = int(subtree_end[child_start])
            subtree_end[i] = child_start
        i -= 1
    return subtree_end


def compute_max_error_at_threshold(
    grid,
    bbox_min: np.ndarray,
    discretization: dyada.discretization.Discretization,
    coefficients: list,
    coarsenable: set[int],
    per_dim_levels: np.ndarray,
    subtree_end: np.ndarray | None = None,
) -> float:
    """Compute max error for a wavelet-thresholded reconstruction.

    Uses an iterative stack-based tree walk.  Reads VDB data on demand via
    copyToArray for each leaf/coarsened region — no full dense array needed.
    """
    grid_shape = tuple(1 << int(l) for l in per_dim_levels)
    descriptor = discretization.descriptor
    background = float(grid.background)
    gx, gy, gz = int(bbox_min[0]), int(bbox_min[1]), int(bbox_min[2])

    if subtree_end is None:
        subtree_end = _build_subtree_end(descriptor)

    max_err = 0.0
    # Stack entries: (desc_i, origin_x, origin_y, origin_z, size_x, size_y, size_z)
    stack = [(0, 0, 0, 0, grid_shape[0], grid_shape[1], grid_shape[2])]

    while stack:
        desc_i, ox, oy, oz, sx, sy, sz = stack.pop()
        ref = descriptor[desc_i]

        if ref.count() == 0 or desc_i in coarsenable:
            scaling = float(coefficients[desc_i][0])
            # Read just this sub-region from VDB
            buf = np.full((sx, sy, sz), background, dtype=np.float32)
            grid.copyToArray(buf, ijk=(gx + ox, gy + oy, gz + oz))
            err = float(np.max(np.abs(buf - scaling)))
            if err > max_err:
                max_err = err
            continue

        # Inner node: push children in reverse order
        csx, csy, csz = sx, sy, sz
        if ref[0]:
            csx //= 2
        if ref[1]:
            csy //= 2
        if ref[2]:
            csz //= 2

        num_children = 1 << ref.count()
        child_desc = desc_i + 1
        child_entries = []
        for child_idx in range(num_children):
            cox, coy, coz = ox, oy, oz
            local_bit = 0
            for d in range(3):
                if ref[d]:
                    if (child_idx >> local_bit) & 1:
                        if d == 0:
                            cox += csx
                        elif d == 1:
                            coy += csy
                        else:
                            coz += csz
                    local_bit += 1
            child_entries.append((child_desc, cox, coy, coz, csx, csy, csz))
            child_desc = int(subtree_end[child_desc])

        for entry in reversed(child_entries):
            stack.append(entry)

    return max_err


def find_coarsenable_nodes(
    discretization: dyada.discretization.Discretization,
    coefficients: list,
    threshold: float,
) -> set[int]:
    """Find nodes that can be coarsened at the given threshold."""
    descriptor = discretization.descriptor
    node_levels = _compute_node_levels(descriptor)

    children_of: dict[int, list[int]] = {}
    stack: list[int] = []
    desc_i = len(descriptor) - 1
    for ref in reversed(descriptor):
        num_ref = ref.count()
        if num_ref == 0:
            stack.append(desc_i)
        else:
            num_children = 1 << num_ref
            children = [stack.pop() for _ in range(num_children)]
            children_of[desc_i] = children
            stack.append(desc_i)
        desc_i -= 1

    coarsenable = set()
    desc_i = len(descriptor) - 1
    stack_pass: list[int] = []
    for ref in reversed(descriptor):
        num_ref = ref.count()
        if num_ref == 0:
            stack_pass.append(desc_i)
            desc_i -= 1
            continue

        num_children = 1 << num_ref
        child_indices = [stack_pass.pop() for _ in range(num_children)]

        all_children_simple = all(
            descriptor[c].count() == 0 or c in coarsenable
            for c in child_indices
        )

        if all_children_simple and len(coefficients[desc_i]) > 1:
            branch_level = node_levels[desc_i].astype(np.int64)
            box_volume = float(np.prod(np.power(2.0, -branch_level)))
            local_bound = threshold * box_volume
            detail_sum = sum(abs(v) for v in coefficients[desc_i][1:])
            if detail_sum <= local_bound:
                coarsenable.add(desc_i)

        stack_pass.append(desc_i)
        desc_i -= 1

    return coarsenable


def omnitree_to_vdb(
    discretization: dyada.discretization.Discretization,
    leaf_values: np.ndarray,
    per_dim_levels: np.ndarray,
    bbox_min: np.ndarray,
    grid_name: str = "density",
    background: float = 0.0,
    source_grid=None,
) -> "vdb.FloatGrid":
    """Convert an omnitree discretization + leaf values into an OpenVDB FloatGrid.

    Each omnitree leaf becomes a VDB tile (uniform-value region) via grid.fill().
    VDB internally stores uniform regions as tiles, so the resulting grid is
    compact — proportional to the number of omnitree leaves, not the dense grid.

    If *source_grid* is given, its transform is copied so the output VDB has
    the same world-space coordinate system as the original.
    """
    grid = vdb.FloatGrid(background)
    grid.name = grid_name
    if source_grid is not None:
        grid.transform = source_grid.transform

    descriptor = discretization.descriptor
    grid_shape = tuple(1 << int(l) for l in per_dim_levels)
    gx, gy, gz = int(bbox_min[0]), int(bbox_min[1]), int(bbox_min[2])

    subtree_end = _build_subtree_end(descriptor)

    # Stack-based DFS: (desc_i, origin_x, origin_y, origin_z, size_x, size_y, size_z)
    stack = [(0, 0, 0, 0, grid_shape[0], grid_shape[1], grid_shape[2])]
    leaf_idx = 0

    while stack:
        desc_i, ox, oy, oz, sx, sy, sz = stack.pop()
        ref = descriptor[desc_i]

        if ref.count() == 0:
            val = float(leaf_values[leaf_idx])
            leaf_idx += 1
            if val != background:
                # fill() uses inclusive min/max in VDB index space
                grid.fill(
                    (gx + ox, gy + oy, gz + oz),
                    (gx + ox + sx - 1, gy + oy + sy - 1, gz + oz + sz - 1),
                    val,
                )
            continue

        # Inner node: push children in reverse order
        csx, csy, csz = sx, sy, sz
        if ref[0]:
            csx //= 2
        if ref[1]:
            csy //= 2
        if ref[2]:
            csz //= 2

        num_children = 1 << ref.count()
        child_desc = desc_i + 1
        child_entries = []
        for child_idx in range(num_children):
            cox, coy, coz = ox, oy, oz
            local_bit = 0
            for d in range(3):
                if ref[d]:
                    if (child_idx >> local_bit) & 1:
                        if d == 0:
                            cox += csx
                        elif d == 1:
                            coy += csy
                        else:
                            coz += csz
                    local_bit += 1
            child_entries.append((child_desc, cox, coy, coz, csx, csy, csz))
            child_desc = int(subtree_end[child_desc])

        for entry in reversed(child_entries):
            stack.append(entry)

    assert leaf_idx == len(leaf_values), (
        f"leaf traversal mismatch: visited {leaf_idx}, expected {len(leaf_values)}"
    )
    return grid


def _full_tree_nodes(per_dim_levels) -> int:
    """Count nodes in a fully-refined tree with given per-dimension levels."""
    max_depth = int(max(per_dim_levels))
    total = 0
    nodes_at_depth = 1
    for d in range(max_depth + 1):
        total += nodes_at_depth
        if d < max_depth:
            n_refined = sum(1 for l in per_dim_levels if int(l) > d)
            nodes_at_depth *= 1 << n_refined
    return total


def _compute_leaf_max_error(grid, bbox_min, disc, coefficients, per_dim_levels):
    """Compute max error between reconstructed leaf values and VDB ground truth."""
    root_scaling = coefficients[0][0]
    coeff_copy = [list(c) for c in coefficients]
    for c in coeff_copy:
        c[0] = np.nan
    coeff_copy[0][0] = root_scaling
    leaf_values = get_leaf_scalings(disc, coeff_copy)

    subtree_end = _build_subtree_end(disc.descriptor)
    coeff_for_err: list[list[float]] = [
        [np.nan] for _ in range(len(disc.descriptor))
    ]
    leaf_idx = 0
    for i in range(len(disc.descriptor)):
        if disc.descriptor[i].count() == 0:
            coeff_for_err[i] = [float(leaf_values[leaf_idx])]
            leaf_idx += 1
    return compute_max_error_at_threshold(
        grid, bbox_min, disc, coeff_for_err, set(), per_dim_levels,
        subtree_end=subtree_end,
    )


def _get_leaf_values(disc, coefficients):
    """Extract leaf values from a discretization with wavelet coefficients."""
    root_scaling = coefficients[0][0]
    coeff_copy = [list(c) for c in coefficients]
    for c in coeff_copy:
        c[0] = np.nan
    coeff_copy[0][0] = root_scaling
    return get_leaf_scalings(disc, coeff_copy)


def _save_stage(output_dir, basename, disc, coefficients, per_dim_levels,
                bbox_min, grid, grid_name):
    """Save descriptor + wavelet coefficients + back-transformed VDB + nonzero mask."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    disc.descriptor.to_file(str(out / basename))
    values = _get_leaf_values(disc, coefficients)
    bg = float(grid.background)
    nonbg = values != bg
    n_nonbg = int(np.count_nonzero(nonbg))
    # Save only non-background values + a packed mask for reconstruction.
    # To reconstruct:
    #   mask = np.unpackbits(np.load("..._nonbg_mask.npy"))[:n_leaves].astype(bool)
    #   values = np.full(n_leaves, bg)
    #   values[mask] = np.load("..._values.npy")
    np.save(str(out / f"{basename}_values.npy"), values[nonbg])
    np.save(str(out / f"{basename}_nonbg_mask.npy"), np.packbits(nonbg))
    out_grid = omnitree_to_vdb(
        disc, values, per_dim_levels, bbox_min,
        grid_name=grid_name, background=bg, source_grid=grid,
    )
    vdb.write(str(out / f"{basename}.vdb"), grids=[out_grid])
    print(f"    -> saved {basename} (descriptor + coefficients + .vdb + mask, "
          f"{n_nonbg}/{len(values)} non-background)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare OpenVDB cloud vs omnitree wavelet compression.",
    )
    parser.add_argument("vdb_file", help="Path to the OpenVDB .vdb file")
    parser.add_argument(
        "--grid",
        default="density",
        help="Name of the VDB grid to use (default: 'density')",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="Cap the refinement level (default: auto from VDB extent). "
        "Useful for large grids that don't fit in memory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.0],
        help="Coarsening threshold(s) for error-bounded compression. "
        "0.0 = lossless (default). Multiple values can be given.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save descriptor files (default: no save)",
    )
    parser.add_argument(
        "--skip-error",
        action="store_true",
        help="Skip max error computation (faster for large grids)",
    )
    args = parser.parse_args()

    vdb_path = args.vdb_file
    if not Path(vdb_path).exists():
        parser.error(f"File not found: {vdb_path}")

    grid = read_vdb_grid(vdb_path, args.grid)
    bbox_min, bbox_max, extent = vdb_grid_extent(grid)
    native_per_dim_levels = levels_from_extent(extent)
    if args.max_level:
        per_dim_levels = np.minimum(native_per_dim_levels, args.max_level)
    else:
        per_dim_levels = native_per_dim_levels
    grid_shape = tuple(1 << int(l) for l in per_dim_levels)
    total_voxels = int(np.prod(grid_shape))

    mem_gb = (total_voxels * 8) / (1 << 30)
    shape_str = " × ".join(str(s) for s in grid_shape)
    print(f"VDB file: {vdb_path}")
    print(f"Grid: {args.grid}")
    print(f"Active bounding box: {bbox_min} -> {bbox_max}")
    print(f"Extent: {extent}")
    print(f"Per-dim levels: {per_dim_levels} (native: {native_per_dim_levels})")
    print(f"Padded grid: {shape_str} = {total_voxels:,} voxels ({mem_gb:.1f} GB dense)")
    print(f"Active voxels: {grid.activeVoxelCount():,}")
    print(f"Background: {grid.background}")
    print()

    # (No dense array needed VDB is queried directly during tree construction)

    print("Building adaptive omnitree from VDB...")
    sys.stdout.flush()
    adaptive_disc, leaf_values = omnitree_from_vdb(grid, bbox_min, per_dim_levels)
    adaptive_stats = descriptor_stats(adaptive_disc.descriptor)
    full_grid_nodes = _full_tree_nodes(per_dim_levels)
    print(
        f"  Adaptive tree: {adaptive_stats['nodes']:,} nodes, "
        f"{adaptive_stats['boxes']:,} boxes "
        f"(vs {full_grid_nodes:,} nodes for full tree)"
    )
    print(
        f"  Compression from adaptive construction: "
        f"{adaptive_stats['nodes'] / full_grid_nodes:.4f}x nodes"
    )
    print()

    # ── Hierarchize ─────────────────────────────────────────────────────────
    print("Hierarchizing (Haar wavelet transform on adaptive tree)...")
    sys.stdout.flush()
    coefficients = transform_to_all_wavelet_coefficients(
        adaptive_disc, leaf_values
    )
    root_scaling = coefficients[0][0]
    print(f"  Root scaling coefficient: {root_scaling:.6f}")

    levels_str = "x".join(str(int(l)) for l in per_dim_levels)
    if args.output_dir:
        _save_stage(args.output_dir, f"cloud_l{levels_str}_octree",
                    adaptive_disc, coefficients, per_dim_levels, bbox_min,
                    grid, args.grid)
    print()

    compute_error = not args.skip_error

    print("OpenVDB topology:")
    vdb_stats = vdb_topology_stats(vdb_path, args.grid)
    vdb_py_stats = vdb_topology_stats_python(grid)
    if vdb_stats:
        for k, v in vdb_stats.items():
            print(f"  {k}: {v:,}")
    else:
        print(f"  leaf nodes: {vdb_py_stats['leaf_count']:,}")
        print(f"  non-leaf nodes: {vdb_py_stats['non_leaf_count']:,}")
        print(f"  active voxels: {vdb_py_stats['active_voxels']:,}")
        print(f"  leaf topo bits (value masks): {vdb_py_stats['leaf_topo_bits']:,}")
        print(f"  internal topo bits (approx): {vdb_py_stats['internal_topo_bits_approx']:,}")
        print(f"  total topo bits (approx): {vdb_py_stats['total_topo_bits_approx']:,}")
        print(f"  memory usage: {vdb_py_stats['mem_usage_bytes']:,} bytes")
        print(f"  node log2 dims: {vdb_py_stats['node_log2_dims']}")
    print()

    for threshold in args.threshold:
        label = "(lossless)" if threshold == 0.0 else ""
        print(f"=== Threshold: {threshold} {label} ===")

        print("  Omnitree coarsening...", end="", flush=True)
        sys.stdout.flush()
        disc_can, coeff_can = compress_by_omnitree_coarsening(
            adaptive_disc,
            [list(c) for c in coefficients],
            coarsening_threshold=threshold,
        )
        can_stats = descriptor_stats(disc_can.descriptor)
        print(f" {can_stats['nodes']:,} nodes, {can_stats['boxes']:,} boxes")

        if args.output_dir:
            _save_stage(args.output_dir, f"cloud_l{levels_str}_t{threshold}_omnitree",
                        disc_can, coeff_can, per_dim_levels, bbox_min, grid, args.grid)

        print("  Downsplit coarsening...", end="", flush=True)
        sys.stdout.flush()
        disc_ds, coeff_ds = compress_by_downsplit_coarsening(
            disc_can,
            [list(c) for c in coeff_can],
            coarsening_threshold=threshold,
        )
        ds_stats = descriptor_stats(disc_ds.descriptor)
        print(f" {ds_stats['nodes']:,} nodes, {ds_stats['boxes']:,} boxes")

        if args.output_dir:
            _save_stage(args.output_dir, f"cloud_l{levels_str}_t{threshold}_downsplit",
                        disc_ds, coeff_ds, per_dim_levels, bbox_min, grid, args.grid)

        if threshold == 0.0:
            oct_err = can_err = ds_err = 0.0
        elif not compute_error:
            oct_err = can_err = ds_err = float("nan")
        else:
            oct_err = 0.0  # adaptive tree is always exact
            print("  Computing omnitree error...", end="", flush=True)
            can_err = _compute_leaf_max_error(
                grid, bbox_min, disc_can, coeff_can, per_dim_levels,
            )
            print(f" {can_err:.6f}")
            print("  Computing downsplit error...", end="", flush=True)
            ds_err = _compute_leaf_max_error(
                grid, bbox_min, disc_ds, coeff_ds, per_dim_levels,
            )
            print(f" {ds_err:.6f}")

        bits_per_coeff = 32  # float32

        header = (
            f"  {'method':<22s} {'nodes':>10s} {'boxes':>10s} {'coeffs':>10s} "
            f"{'topo bits':>12s} {'coeff bits':>12s} {'total bits':>12s} {'max err':>10s}"
        )
        sep = f"  {'-' * 94}"
        print(header)
        print(sep)

        def _print_row(name, stats, err):
            topo = stats["topo_bits"]
            n_coeffs = stats["boxes"]
            coeff_bits = n_coeffs * bits_per_coeff
            total = topo + coeff_bits
            err_str = f"{err:>10.6f}" if not math.isnan(err) else "       N/A"
            print(
                f"  {name:<22s} {stats['nodes']:>10,d} "
                f"{stats['boxes']:>10,d} {n_coeffs:>10,d} "
                f"{topo:>12,d} {coeff_bits:>12,d} "
                f"{total:>12,d} {err_str}"
            )

        _print_row("octree", adaptive_stats, oct_err)
        _print_row("omnitree", can_stats, can_err)
        _print_row("downsplit", ds_stats, ds_err)

        # VDB reference
        if vdb_stats:
            vdb_topo = vdb_stats.get("total_topology_bits", 0)
            vdb_coeffs_count = vdb_stats.get("total_coefficients", 0)
            vdb_coeff_bits = vdb_coeffs_count * bits_per_coeff
            vdb_total = vdb_topo + vdb_coeff_bits
        else:
            vdb_topo = vdb_py_stats["total_topo_bits_approx"]
            vdb_active = vdb_py_stats["active_voxels"]
            vdb_coeffs_count = vdb_active
            vdb_coeff_bits = vdb_active * bits_per_coeff
            vdb_total = vdb_topo + vdb_coeff_bits

        print(
            f"  {'openvdb':<22s} {'--':>10s} {'--':>10s} {vdb_coeffs_count:>10,d} "
            f"{vdb_topo:>12,d} {vdb_coeff_bits:>12,d} "
            f"{vdb_total:>12,d} {'0.000000':>10s}"
        )

        # Full dense grid reference
        full_bits = total_voxels * bits_per_coeff
        print(
            f"  {'full dense grid':<22s} {'--':>10s} {total_voxels:>10,d} {total_voxels:>10,d} "
            f"{'0':>12s} {full_bits:>12,d} "
            f"{full_bits:>12,d} {'0.000000':>10s}"
        )

        print()

        def _total_bits(stats):
            return stats["topo_bits"] + stats["boxes"] * bits_per_coeff

        oct_total = _total_bits(adaptive_stats)
        can_total = _total_bits(can_stats)
        ds_total = _total_bits(ds_stats)

        print(f"  Compression ratios (total bits vs full dense):")
        for name, total in [("octree", oct_total), ("omnitree", can_total),
                            ("downsplit", ds_total), ("openvdb", vdb_total)]:
            print(f"    {name + ':':<22s} {total / full_bits:.6f}x  ({full_bits / total:.1f}:1)")

        print(f"  Downsplit vs OpenVDB:")
        print(f"    topo bits:  {ds_stats['topo_bits'] / vdb_topo:.4f}x" if vdb_topo > 0 else "    topo bits:  N/A")
        print(f"    total bits: {ds_total / vdb_total:.4f}x" if vdb_total > 0 else "    total bits: N/A")
        print()


if __name__ == "__main__":
    main()
