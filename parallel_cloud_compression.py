#!/usr/bin/env python3
"""Demonstrate parallel omnitree compression of the WDAS cloud.

Partitions the grid into sub-regions, compresses each independently
(simulating worker threads), then stitches the partial descriptors
via compose_grid for a final round of coarsening.

Usage:
    python -O parallel_cloud_compression.py [--workers 8]
"""

import argparse
import json
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bitarray as ba
import numpy as np

import dyada
import dyada.descriptor
import dyada.discretization
import dyada.linearization
from dyada.descriptor_builder import compose_grid
from dyada.linearization import flat_to_coord

try:
    from dyada_cache_patch import install as _install_cache
    _install_cache()
except ImportError:
    pass

from compare_cloud import (
    read_vdb_grid, vdb_grid_extent, levels_from_extent, omnitree_from_vdb,
)
from wavelets_with_omnitrees import (
    transform_to_all_wavelet_coefficients,
    get_leaf_scalings,
    compress_by_downsplit_coarsening,
    compress_by_omnitree_coarsening,
)


# ── Worker function ────────────────────────────────────────────────────────

def partition_paths(work_dir, part_idx):
    """Return (descriptor_file, values_file) for a given partition index."""
    return (
        f"{work_dir}/part{part_idx:05d}_3d.bin",
        f"{work_dir}/part{part_idx:05d}_values.npy",
    )


_WORKER_GRID = None  # per-process cache of the VDB grid (set by _init_worker)


def _init_worker(vdb_path=None, grid_name=None):
    """Pool initializer: set up imports and optionally load the VDB grid."""
    global _WORKER_GRID
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from dyada_cache_patch import install
        install()
    except ImportError:
        pass
    if vdb_path is not None:
        from compare_cloud import read_vdb_grid
        _WORKER_GRID = read_vdb_grid(vdb_path, grid_name)


def compress_partition(task):
    """Compress a single partition.  Runs in a worker process.

    task is a dict with keys:
      part_idx, work_dir, threshold  (always present)
      input_dir                      (cascaded: load from previous output)
      vdb_path, grid_name,
      sub_bbox_min, sub_levels       (VDB mode: build omnitree from grid)
    """
    part_idx = task["part_idx"]
    work_dir = task["work_dir"]
    threshold = task["threshold"]
    input_dir = task.get("input_dir")

    # Skip if another process already wrote this partition
    desc_file, vals_file = partition_paths(work_dir, part_idx)
    if os.path.exists(desc_file) and os.path.exists(vals_file):
        desc = dyada.descriptor.RefinementDescriptor.from_file(desc_file)
        return (part_idx, {
            "initial_nodes": -1,
            "compressed_nodes": len(desc),
            "compressed_boxes": desc.get_num_boxes(),
            "skipped": True,
        })

    # Obtain discretization + leaf values
    if input_dir is not None:
        # Cascaded: load from previous threshold's output
        in_desc, in_vals = partition_paths(input_dir, part_idx)
        desc = dyada.descriptor.RefinementDescriptor.from_file(in_desc)
        leaf_values = np.load(in_vals)
        disc = dyada.discretization.Discretization(
            dyada.linearization.MortonOrderLinearization(), desc
        )
    else:
        # VDB mode: build omnitree from the per-worker grid
        from compare_cloud import read_vdb_grid, omnitree_from_vdb
        global _WORKER_GRID
        if _WORKER_GRID is None:
            _WORKER_GRID = read_vdb_grid(task["vdb_path"], task["grid_name"])
        disc, leaf_values = omnitree_from_vdb(
            _WORKER_GRID,
            np.array(task["sub_bbox_min"]),
            np.array(task["sub_levels"]),
        )

    n_initial = len(disc.descriptor)
    coefficients = transform_to_all_wavelet_coefficients(disc, leaf_values)
    coeff_list = [list(c) for c in coefficients]
    disc_ds, coeff_ds, _ = compress_by_downsplit_coarsening(
        disc, coeff_list, coarsening_threshold=threshold
    )

    # Extract leaf values for later recomposition
    root_scaling = coeff_ds[0][0]
    coeff_copy = [list(c) for c in coeff_ds]
    for c in coeff_copy:
        c[0] = np.nan
    coeff_copy[0][0] = root_scaling
    out_leaf_values = get_leaf_scalings(disc_ds, coeff_copy)

    # Persist descriptor and values
    disc_ds.descriptor.to_file(desc_file.rsplit("_3d.bin", 1)[0])
    np.save(vals_file, out_leaf_values)

    return (
        part_idx,
        {
            "initial_nodes": n_initial,
            "compressed_nodes": len(disc_ds.descriptor),
            "compressed_boxes": disc_ds.descriptor.get_num_boxes(),
        },
    )


def load_partition(work_dir, part_idx):
    """Load a previously-saved partition.  Returns (descriptor, leaf_values, stats)."""
    import dyada.descriptor
    desc_file, vals_file = partition_paths(work_dir, part_idx)
    desc = dyada.descriptor.RefinementDescriptor.from_file(desc_file)
    leaf_values = np.load(vals_file)
    stats = {
        "initial_nodes": -1,  # unknown after reload
        "compressed_nodes": len(desc),
        "compressed_boxes": desc.get_num_boxes(),
    }
    return desc, leaf_values, stats


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel omnitree compression of WDAS cloud"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--vdb", type=str,
                        default="/workspace/wdas_cloud/wdas_cloud_sixteenth.vdb")
    parser.add_argument("--grid-name", type=str, default="density")
    parser.add_argument("--max-level", type=int, default=6)
    parser.add_argument("--only-subs", action="store_true",
                        help="run only the (parallel) sub-descriptor parts")
    parser.add_argument("--skip-serial", action="store_true",
                        help="Skip the serial baseline (faster for testing)")
    parser.add_argument("--split-levels", type=int, nargs="+", default=None,
                        help="Top-level splits per dim.  Pass a single int to apply "
                        "the same level to all dims (e.g. 3 -> 8^3=512 partitions), "
                        "or one int per dim (e.g. 3 3 4 -> 8x8x16=1024 partitions). "
                        "Default: auto (~4-8x more partitions than workers).")
    parser.add_argument("--work-dir", type=str, default="/tmp/parallel_cloud",
                        help="Directory for per-partition cache files (resumable).")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with previous threshold's partitions. "
                        "Workers load from here instead of VDB (cascading mode).")
    args = parser.parse_args()

    vdb_path = args.vdb
    cascading = args.input_dir is not None

    if cascading:
        if os.path.realpath(args.input_dir) == os.path.realpath(args.work_dir):
            parser.error("--input-dir and --work-dir must be different directories")
        meta_path = os.path.join(args.input_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        bbox_min = np.array(meta["bbox_min"])
        per_dim_levels = np.array(meta["per_dim_levels"])
        print(f"Cascading from {args.input_dir}")
        print(f"  per-dim levels: {per_dim_levels}")
    else:
        grid = read_vdb_grid(vdb_path, args.grid_name)
        bbox_min, _, extent = vdb_grid_extent(grid)
        per_dim_levels = np.minimum(levels_from_extent(extent), args.max_level)
        print(f"Grid extent: {extent}, per-dim levels: {per_dim_levels}")
    print(f"Workers: {args.workers}, threshold: {args.threshold}")

    # ── Determine partition layout ─────────────────────────────────────────
    # Aim for ~8x more partitions than workers, so unbalanced load averages out.
    num_dims = len(per_dim_levels)
    if args.split_levels is None:
        target_partitions = max(args.workers * 8, 8)
        s = max(1, int(np.ceil(np.log2(target_partitions) / num_dims)))
        grid_levels = [s] * num_dims
    elif len(args.split_levels) == 1:
        grid_levels = [args.split_levels[0]] * num_dims
    elif len(args.split_levels) == num_dims:
        grid_levels = list(args.split_levels)
    else:
        parser.error(
            f"--split-levels takes 1 or {num_dims} values, got {len(args.split_levels)}"
        )
    # Cap each dim so its sub-region keeps at least 3 levels of refinement
    for d in range(num_dims):
        max_split_d = max(1, int(per_dim_levels[d]) - 3)
        if grid_levels[d] > max_split_d:
            print(f"  Capping split_levels[{d}] {grid_levels[d]} -> {max_split_d} "
                  f"(per-dim level {d} is {int(per_dim_levels[d])})")
            grid_levels[d] = max_split_d
    grid_shape = tuple(1 << lv for lv in grid_levels)
    n_partitions = 1
    for s in grid_shape:
        n_partitions *= s
    sub_levels = per_dim_levels - np.array(grid_levels)

    # Compute sub-region origins (Fortran-order flat index → grid coord → bbox offset)
    full_grid_shape = np.array([1 << int(l) for l in per_dim_levels])
    sub_grid_shape = np.array([1 << int(l) for l in sub_levels])

    # Build partition argument tuples
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    partition_args = []
    for flat_idx in range(n_partitions):
        task = {"part_idx": flat_idx, "threshold": args.threshold, "work_dir": work_dir}
        if cascading:
            task["input_dir"] = args.input_dir
        else:
            coord = flat_to_coord(flat_idx, grid_shape)
            sub_origin = np.array(coord) * sub_grid_shape
            sub_bbox_min = bbox_min + sub_origin
            task.update(vdb_path=vdb_path, grid_name=args.grid_name,
                        sub_bbox_min=sub_bbox_min.tolist(),
                        sub_levels=sub_levels.tolist())
        partition_args.append(task)

    # Skip partitions whose output already exists on disk
    pending = []
    cached = []
    for a in partition_args:
        flat_idx = a["part_idx"]
        desc_file, vals_file = partition_paths(work_dir, flat_idx)
        if os.path.exists(desc_file) and os.path.exists(vals_file):
            cached.append(flat_idx)
        else:
            pending.append(a)

    print(f"\nPartitioned into {n_partitions} sub-regions "
          f"(split_levels={grid_levels}, grid={grid_shape}, sub_levels={sub_levels})")
    print(f"Resume cache in {work_dir}/: {len(cached)} cached, {len(pending)} pending")
    if not cascading:
        del grid

    # ── Serial baseline (optional, skip with --skip-serial) ──────────────
    t_serial = None
    disc_serial = None
    if not args.skip_serial and not cascading:
        print("\n=== Serial baseline ===")
        t_serial_start = time.perf_counter()
        grid = read_vdb_grid(vdb_path, args.grid_name)
        disc, vals = omnitree_from_vdb(grid, bbox_min, per_dim_levels)
        coefficients = transform_to_all_wavelet_coefficients(disc, vals)
        coeff_list = [list(c) for c in coefficients]
        disc_serial, coeff_serial, _ = compress_by_downsplit_coarsening(
            disc, coeff_list, coarsening_threshold=args.threshold
        )
        t_serial = time.perf_counter() - t_serial_start
        print(f"  {len(disc_serial.descriptor)} nodes, "
              f"{disc_serial.descriptor.get_num_boxes()} boxes in {t_serial:.1f}s")
        del grid
    else:
        print("\n=== Serial baseline skipped ===")

    # ── Parallel compression ───────────────────────────────────────────────
    print(f"\n=== Parallel ({args.workers} workers, "
          f"{len(pending)} pending, {len(cached)} cached) ===")
    t_parallel_start = time.perf_counter()

    # Don't hold partition data in the main process during the parallel phase —
    # workers write to disk anyway.  Just track stats; load from disk only when
    # compose_grid actually needs them.  This keeps main-process RAM bounded
    # even for hundreds of fine-threshold partitions.
    child_stats = [None] * n_partitions

    if pending:
        if cascading:
            pool_kwargs = dict(max_workers=args.workers, initializer=_init_worker)
        else:
            pool_kwargs = dict(max_workers=args.workers, initializer=_init_worker,
                               initargs=(vdb_path, args.grid_name))
        with ProcessPoolExecutor(**pool_kwargs) as pool:
            futures = {pool.submit(compress_partition, a): a["part_idx"]
                       for a in pending}
            for future in as_completed(futures):
                part_idx, stats = future.result()
                child_stats[part_idx] = stats
                if stats.get("skipped"):
                    print(f"  Worker {part_idx:5d}: {stats['compressed_nodes']:7d} nodes "
                          f"(skipped, already on disk)")
                else:
                    print(f"  Worker {part_idx:5d}: {stats['compressed_nodes']:7d} nodes, "
                          f"{stats['compressed_boxes']:7d} boxes "
                          f"(from {stats['initial_nodes']})")

    # Save metadata for cascading to next threshold
    meta = {
        "bbox_min": bbox_min.tolist() if hasattr(bbox_min, 'tolist') else list(bbox_min),
        "per_dim_levels": per_dim_levels.tolist() if hasattr(per_dim_levels, 'tolist') else list(per_dim_levels),
        "grid_levels": list(grid_levels),
        "n_partitions": n_partitions,
    }
    with open(os.path.join(work_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)

    if args.only_subs:
        return

    # Load all partitions (cached + freshly computed) from disk for compose.
    child_descriptors = [None] * n_partitions
    child_values = [None] * n_partitions
    for part_idx in range(n_partitions):
        desc, vals, stats = load_partition(work_dir, part_idx)
        child_descriptors[part_idx] = desc
        child_values[part_idx] = vals
        if child_stats[part_idx] is None:
            child_stats[part_idx] = stats

    print(f"  Loaded {n_partitions} partitions from {work_dir}/ for compose")

    t_workers = time.perf_counter() - t_parallel_start

    # ── Compose via compose_grid ───────────────────────────────────────────
    t_compose_start = time.perf_counter()
    composed_desc, node_mappings = compose_grid(grid_levels, child_descriptors)
    composed_disc = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(), composed_desc
    )

    # Compose leaf values in matching order: walk composed descriptor leaves
    # and pull values from child_values via node_mappings
    all_leaf_parts = []
    for flat_idx in range(n_partitions):
        if flat_idx in node_mappings:
            # The mapping tells us which composed nodes came from this sub-descriptor
            sub_desc = child_descriptors[flat_idx]
            sub_vals = child_values[flat_idx]
            all_leaf_parts.append(sub_vals)
        else:
            # Partition had no sub-descriptor (None) → single background leaf
            all_leaf_parts.append(np.array([0.0]))

    # compose_grid uses Z-order internally; we need to reorder values to match
    # the composed descriptor's leaf ordering. For now, concatenate in Z-order
    # since compose_grid arranges sub-descriptors in Z-order.
    # The base descriptor is a regular grid with grid_levels, so its leaves
    # are in Z-order = Morton order. compose_descriptors splices each sub-tree
    # into the base leaf position, preserving DFS order within each sub-tree.
    # So the composed leaf values are just the concatenation in Z-order.
    base_desc = dyada.descriptor.RefinementDescriptor(3, list(grid_levels))
    base_disc = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(), base_desc
    )
    # Get the Z-order of flat indices
    from dyada.linearization import grid_coord_to_z_index
    z_order = []
    for flat_idx in range(n_partitions):
        coord = flat_to_coord(flat_idx, grid_shape)
        z_idx = grid_coord_to_z_index(coord, grid_levels)
        z_order.append((z_idx, flat_idx))
    z_order.sort()  # sort by Z-index

    composed_vals = np.concatenate([child_values[flat_idx] for _, flat_idx in z_order])

    print(f"\n  Composed tree: {len(composed_desc)} nodes, "
          f"{composed_desc.get_num_boxes()} boxes, "
          f"{len(composed_vals)} leaf values")

    # ── Final coarsening round (omnitree canonical, no downsplit) ──────────
    coefficients = transform_to_all_wavelet_coefficients(composed_disc, composed_vals)
    coeff_list = [list(c) for c in coefficients]
    disc_final, coeff_final, _ = compress_by_omnitree_coarsening(
        composed_disc, coeff_list, coarsening_threshold=args.threshold
    )
    t_compose_and_final = time.perf_counter() - t_compose_start
    t_parallel_total = time.perf_counter() - t_parallel_start

    print(f"  After final coarsening: {len(disc_final.descriptor)} nodes, "
          f"{disc_final.descriptor.get_num_boxes()} boxes")

    # ── Save final result ─────────────────────────────────────────────────
    final_prefix = os.path.join(work_dir, "final")
    disc_final.descriptor.to_file(final_prefix)
    root_scaling = coeff_final[0][0]
    coeff_copy = [list(c) for c in coeff_final]
    for c in coeff_copy:
        c[0] = np.nan
    coeff_copy[0][0] = root_scaling
    final_leaf_values = get_leaf_scalings(disc_final, coeff_copy)
    np.save(os.path.join(work_dir, "final_values.npy"), final_leaf_values)
    desc_file = f"{final_prefix}_3d.bin"
    vals_file = os.path.join(work_dir, "final_values.npy")
    desc_bytes = os.path.getsize(desc_file)
    vals_bytes = os.path.getsize(vals_file)
    print(f"  Saved: {desc_file} ({desc_bytes:,} bytes) + "
          f"{vals_file} ({vals_bytes:,} bytes) = {desc_bytes + vals_bytes:,} bytes total")

    # ── Summary ────────────────────────────────────────────────────────────
    fn = len(disc_final.descriptor)
    fb = disc_final.descriptor.get_num_boxes()
    total_worker_nodes = sum(s["compressed_nodes"] for s in child_stats)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if disc_serial is not None:
        sn = len(disc_serial.descriptor)
        sb = disc_serial.descriptor.get_num_boxes()
        print(f"Serial:            {sn:6d} nodes, {sb:6d} boxes, {t_serial:.1f}s")
    print(f"Workers total:     {total_worker_nodes:6d} nodes, {t_workers:.1f}s wall")
    print(f"Composed:          {len(composed_desc):6d} nodes, {composed_desc.get_num_boxes():6d} boxes")
    print(f"Final:             {fn:6d} nodes, {fb:6d} boxes, "
          f"+{t_compose_and_final:.1f}s compose+coarsen")
    print(f"")
    print(f"Parallel total:    {t_parallel_total:.1f}s "
          f"(workers: {t_workers:.1f}s + compose+coarsen: {t_compose_and_final:.1f}s)")
    if t_serial is not None:
        print(f"Speedup:           {t_serial / t_parallel_total:.2f}x")
    if disc_serial is not None:
        print(f"Quality vs serial: {fn/sn:.3f}x nodes ({fn - sn:+d}), "
              f"{fb/sb:.3f}x boxes ({fb - sb:+d})")


if __name__ == "__main__":
    main()
