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
    from wavelets_with_omnitrees.wavelets_with_omnitrees import (
        transform_to_all_wavelet_coefficients,
        get_leaf_scalings,
        _compute_node_levels,
    )
except ModuleNotFoundError:
    from wavelets_with_omnitrees import (  # type: ignore
        transform_to_all_wavelet_coefficients,
        get_leaf_scalings,
        _compute_node_levels,
    )


# ── VDB helpers ─────────────────────────────────────────────────────────────


def read_vdb_grid(vdb_path: str, grid_name: str):
    """Read a named grid from a VDB file."""
    grids, _metadata = vdb.readAll(vdb_path)
    for g in grids:
        if g.name == grid_name:
            return g
    names = [g.name for g in grids]
    raise ValueError(
        f"Grid '{grid_name}' not found in {vdb_path}. Available: {names}"
    )


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


def vdb_to_dense_array(grid, bbox_min: np.ndarray, dims: int) -> np.ndarray:
    """Copy VDB grid data into a dense numpy array."""
    background = float(grid.background)
    array = np.full((dims, dims, dims), background, dtype=np.float64)

    try:
        buf = np.full((dims, dims, dims), background, dtype=np.float32)
        grid.copyToArray(buf, ijk=tuple(int(c) for c in bbox_min))
        array[:] = buf
        return array
    except (AttributeError, TypeError):
        pass

    print("  (copyToArray not available, using accessor fallback — may be slow)")
    _, active_max_incl = grid.evalActiveVoxelBoundingBox()
    active_max = np.array(active_max_incl, dtype=np.int64) + 1
    active_min = np.array(grid.evalActiveVoxelBoundingBox()[0], dtype=np.int64)

    rel_min = np.maximum(active_min - bbox_min, 0).astype(int)
    rel_max = np.minimum(active_max - bbox_min, dims).astype(int)

    acc = grid.getAccessor()
    for x in range(rel_min[0], rel_max[0]):
        for y in range(rel_min[1], rel_max[1]):
            for z in range(rel_min[2], rel_max[2]):
                ijk = (
                    int(bbox_min[0]) + x,
                    int(bbox_min[1]) + y,
                    int(bbox_min[2]) + z,
                )
                val, active = acc.probeValue(ijk)
                if active:
                    array[x, y, z] = float(val)
    return array


# ── Adaptive octree from VDB (sparse — no full dense array) ───────────────

_LEAF_3D = ba.frozenbitarray("000")
_FULL_SPLIT_3D = ba.frozenbitarray("111")

# Maximum sub-region size for which we copy data to check uniformity.
# 128^3 × 4 bytes = 8 MB per buffer — small enough for on-stack use.
_MAX_CHECK_SIZE = 128


def omnitree_from_vdb(
    grid,
    bbox_min: np.ndarray,
    level: int,
) -> tuple[dyada.discretization.Discretization, np.ndarray]:
    """Build an adaptive omnitree from a VDB grid.

    Queries VDB directly without allocating the full dense array.
    Regions outside the active bounding box are immediately collapsed to
    background-valued leaves.  Small sub-regions are checked for uniformity
    via copyToArray into a temporary buffer.
    """
    dims = 1 << level
    background = float(grid.background)

    # Active bounding box in *local* coordinates (relative to bbox_min)
    ab_min_global, ab_max_global = grid.evalActiveVoxelBoundingBox()
    ab_min = np.array(ab_min_global, dtype=np.int64) - bbox_min
    ab_max = np.array(ab_max_global, dtype=np.int64) + 1 - bbox_min  # exclusive

    print(f"  Grid: {dims}^3, active local range: {ab_min} .. {ab_max}")

    # Global offset for copyToArray ijk parameter
    gx, gy, gz = int(bbox_min[0]), int(bbox_min[1]), int(bbox_min[2])

    descriptor_bits = ba.bitarray()
    leaf_values: list[float] = []

    def _recurse(x0: int, y0: int, z0: int, size: int) -> None:
        # Fast path: entirely outside active bounding box → background leaf
        if (x0 >= ab_max[0] or x0 + size <= ab_min[0] or
            y0 >= ab_max[1] or y0 + size <= ab_min[1] or
            z0 >= ab_max[2] or z0 + size <= ab_min[2]):
            descriptor_bits.extend(_LEAF_3D)
            leaf_values.append(background)
            return

        if size == 1:
            descriptor_bits.extend(_LEAF_3D)
            # Single voxel — probe directly
            acc = grid.getConstAccessor()
            val = acc.getValue((gx + x0, gy + y0, gz + z0))
            leaf_values.append(float(val))
            return

        # For small-enough regions, copy and check uniformity
        if size <= _MAX_CHECK_SIZE:
            buf = np.full((size, size, size), background, dtype=np.float32)
            grid.copyToArray(buf, ijk=(gx + x0, gy + y0, gz + z0))
            if buf.min() == buf.max():
                descriptor_bits.extend(_LEAF_3D)
                leaf_values.append(float(buf.flat[0]))
                return

        # Non-uniform (or too large to check cheaply): split
        descriptor_bits.extend(_FULL_SPLIT_3D)
        half = size >> 1
        for child_idx in range(8):
            cx = x0 + half * (child_idx & 1)
            cy = y0 + half * ((child_idx >> 1) & 1)
            cz = z0 + half * ((child_idx >> 2) & 1)
            _recurse(cx, cy, cz, half)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), level + 100))
    _recurse(0, 0, 0, dims)

    descriptor = dyada.descriptor.RefinementDescriptor.from_binary(
        3, descriptor_bits
    )
    discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(), descriptor
    )
    values = np.array(leaf_values, dtype=np.float64)

    assert len(values) == len(discretization), (
        f"leaf count mismatch: {len(values)} values vs "
        f"{len(discretization)} boxes in descriptor"
    )

    return discretization, values


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
    level: int,
    subtree_end: np.ndarray | None = None,
) -> float:
    """Compute max error for a wavelet-thresholded reconstruction.

    Uses an iterative stack-based tree walk.  Reads VDB data on demand via
    copyToArray for each leaf/coarsened region — no full dense array needed.
    """
    dims = 1 << level
    descriptor = discretization.descriptor
    background = float(grid.background)
    gx, gy, gz = int(bbox_min[0]), int(bbox_min[1]), int(bbox_min[2])

    if subtree_end is None:
        subtree_end = _build_subtree_end(descriptor)

    max_err = 0.0
    # Stack entries: (desc_i, origin_x, origin_y, origin_z, size_x, size_y, size_z)
    stack = [(0, 0, 0, 0, dims, dims, dims)]

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


# ── Main ────────────────────────────────────────────────────────────────────


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

    # ── Read VDB and determine resolution ───────────────────────────────────
    grid = read_vdb_grid(vdb_path, args.grid)
    bbox_min, bbox_max, extent = vdb_grid_extent(grid)
    per_dim_levels = levels_from_extent(extent)
    native_level = int(per_dim_levels.max())
    level = min(native_level, args.max_level) if args.max_level else native_level
    dims = 1 << level

    mem_gb = (dims**3 * 8) / (1 << 30)
    print(f"VDB file: {vdb_path}")
    print(f"Grid: {args.grid}")
    print(f"Active bounding box: {bbox_min} -> {bbox_max}")
    print(f"Extent: {extent}  (per-dim levels: {per_dim_levels})")
    print(f"Level: {level} (native: {native_level})")
    print(f"Padded grid: {dims}^3 = {dims**3:,} voxels ({mem_gb:.1f} GB dense)")
    print(f"Active voxels: {grid.activeVoxelCount():,}")
    print(f"Background: {grid.background}")
    print()

    # (No dense array needed — VDB is queried directly during tree construction)

    # ── Build adaptive omnitree from VDB ────────────────────────────────────
    print("Building adaptive omnitree from VDB...")
    sys.stdout.flush()
    adaptive_disc, leaf_values = omnitree_from_vdb(grid, bbox_min, level)
    adaptive_stats = descriptor_stats(adaptive_disc.descriptor)
    full_grid_nodes = sum(8**i for i in range(level + 1))
    print(
        f"  Adaptive tree: {adaptive_stats['nodes']:,} nodes, "
        f"{adaptive_stats['boxes']:,} boxes "
        f"(vs {full_grid_nodes:,} nodes for full octree)"
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
    print()

    # ── Error computation flag ───────────────────────────────────────────────
    compute_error = not args.skip_error

    # ── VDB topology stats ──────────────────────────────────────────────────
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

    # ── Precompute subtree end table for fast tree traversal ──────────────
    subtree_end = _build_subtree_end(adaptive_disc.descriptor) if compute_error else None

    # ── Analyze wavelet compression at each threshold ──────────────��───────
    print("Analyzing wavelet compression...")
    sys.stdout.flush()
    compression_stats = analyze_wavelet_compression(
        adaptive_disc, coefficients, args.threshold
    )

    bits_per_coeff = 32  # float32

    header = (
        f"  {'method':<22s} {'nodes':>10s} {'boxes':>10s} {'topo bits':>12s} "
        f"{'coeff bits':>12s} {'total bits':>12s} {'max err':>10s}"
    )
    sep = f"  {'-' * 82}"

    for threshold in args.threshold:
        label = "(lossless)" if threshold == 0.0 else ""
        print(f"\n=== Threshold: {threshold} {label} ===")
        print(header)
        print(sep)

        stats = compression_stats[threshold]

        # Compute max error
        if threshold == 0.0:
            max_err = 0.0
        elif not compute_error:
            max_err = float("nan")
        else:
            coarsenable = find_coarsenable_nodes(
                adaptive_disc, coefficients, threshold
            )
            max_err = compute_max_error_at_threshold(
                grid, bbox_min, adaptive_disc, coefficients, coarsenable, level,
                subtree_end=subtree_end,
            )

        omni_topo = stats["topo_bits"]
        omni_coeffs = stats["boxes"] * bits_per_coeff
        omni_total = omni_topo + omni_coeffs

        err_str = f"{max_err:>10.6f}" if not math.isnan(max_err) else "       N/A"
        print(
            f"  {'omnitree+wavelet':<22s} {stats['nodes']:>10,d} "
            f"{stats['boxes']:>10,d} "
            f"{omni_topo:>12,d} {omni_coeffs:>12,d} "
            f"{omni_total:>12,d} {err_str}"
        )

        # Adaptive octree reference (before wavelet coarsening)
        adapt_topo = adaptive_stats["topo_bits"]
        adapt_coeffs = adaptive_stats["boxes"] * bits_per_coeff
        adapt_total = adapt_topo + adapt_coeffs
        print(
            f"  {'adaptive octree':<22s} "
            f"{adaptive_stats['nodes']:>10,d} "
            f"{adaptive_stats['boxes']:>10,d} "
            f"{adapt_topo:>12,d} {adapt_coeffs:>12,d} "
            f"{adapt_total:>12,d} {'0.000000':>10s}"
        )

        # VDB reference
        if vdb_stats:
            vdb_topo = vdb_stats.get("total_topology_bits", 0)
            vdb_coeffs_count = vdb_stats.get("total_coefficients", 0)
            vdb_coeff_bits = vdb_coeffs_count * bits_per_coeff
            vdb_total = vdb_topo + vdb_coeff_bits
        else:
            vdb_topo = vdb_py_stats["total_topo_bits_approx"]
            vdb_active = vdb_py_stats["active_voxels"]
            vdb_coeff_bits = vdb_active * bits_per_coeff
            vdb_total = vdb_topo + vdb_coeff_bits

        print(
            f"  {'openvdb':<22s} {'--':>10s} {'--':>10s} "
            f"{vdb_topo:>12,d} {vdb_coeff_bits:>12,d} "
            f"{vdb_total:>12,d} {'0.000000':>10s}"
        )

        # Full dense grid reference
        full_voxels = dims**3
        full_bits = full_voxels * bits_per_coeff
        print(
            f"  {'full dense grid':<22s} {'--':>10s} {full_voxels:>10,d} "
            f"{'0':>12s} {full_bits:>12,d} "
            f"{full_bits:>12,d} {'0.000000':>10s}"
        )

        print()

        # ── Ratios ──────────────────────────────────────────────────────────
        print(f"  Compression ratios (total bits vs full dense):")
        print(f"    omnitree+wavelet: {omni_total / full_bits:.6f}x  ({full_bits / omni_total:.1f}:1)")
        print(f"    adaptive octree:  {adapt_total / full_bits:.6f}x  ({full_bits / adapt_total:.1f}:1)")
        print(f"    openvdb:          {vdb_total / full_bits:.6f}x  ({full_bits / vdb_total:.1f}:1)")

        if threshold > 0.0:
            print(f"  Wavelet coarsening vs adaptive:")
            print(f"    nodes: {stats['nodes'] / adaptive_stats['nodes']:.4f}")
            print(f"    boxes: {stats['boxes'] / adaptive_stats['boxes']:.4f}")
            print(f"    coarsened: {stats['coarsenable_nodes']:,} nodes")

        print(f"  Omnitree vs OpenVDB:")
        print(f"    topo bits:  {omni_topo / vdb_topo:.4f}x" if vdb_topo > 0 else "    topo bits:  N/A")
        print(f"    total bits: {omni_total / vdb_total:.4f}x" if vdb_total > 0 else "    total bits: N/A")
        print()

    # ── Save descriptors ────────────────────────────────────────────────────
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        adaptive_disc.descriptor.to_file(str(out / f"cloud_l{level}_adaptive"))
        print(f"Saved adaptive descriptor to {out}/")


if __name__ == "__main__":
    main()
