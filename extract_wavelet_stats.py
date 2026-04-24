#!/usr/bin/env python
"""Extract storage statistics from wavelet compression output files.

Scans a directory for {thingi_id}_l{level}_canonical_3d.bin and
{thingi_id}_l{level}_pushdown_3d.bin files (and optionally openvdb), 
extracts tree statistics, and writes results to a CSV file.

Skips (thingi_id, level) combinations already present in the CSV.

Usage:
    python extract_wavelet_stats.py [--dir output/] [--csv wavelet_stats.csv]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from filelock import FileLock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dyada import Discretization, MortonOrderLinearization, RefinementDescriptor


DIMENSIONALITY = 3

CSV_FILENAME = "wavelet_stats.csv"

COLUMNS = [
    "thingi_id",
    "level",
    # Canonical coarsening
    "can_nodes",
    "can_boxes",
    "can_topo_bits",
    "can_ref_counts",
    # Downsplit (pushdown) coarsening
    "ds_nodes",
    "ds_boxes",
    "ds_topo_bits",
    "ds_ref_counts",
    # OpenVDB (optional)
    "vdb_topo_bits",
    "vdb_topo_bytes",
    "vdb_active_leaf_voxels",
    "vdb_dense_value_bits",
    "vdb_active_value_bits",
    "vdb_active_leaf",
    "vdb_inactive_leaf",
    "vdb_active_tiles",
    "vdb_total_leaf_coefficients",
    "vdb_total_coefficients",
    "vdb_file_bytes",
    # Compressed omnitree sizes (canonical)
    "can_raw_total_bytes",
    "can_blosc2_total_bytes",
    "can_desc_raw_bytes",
    "can_desc_blosc2_bytes",
    "can_coeffs_raw_bytes",
    "can_coeffs_blosc2_bytes",
    # Compressed omnitree sizes (downsplit)
    "ds_raw_total_bytes",
    "ds_blosc2_total_bytes",
    "ds_desc_raw_bytes",
    "ds_desc_blosc2_bytes",
    "ds_coeffs_raw_bytes",
    "ds_coeffs_blosc2_bytes",
]


class WaveletStatsFile:
    def __init__(self, filename=CSV_FILENAME):
        self.filename = filename
        self.lockfile = filename + ".lock"

        with FileLock(self.lockfile):
            try:
                with open(self.filename, "x") as f:
                    df = pd.DataFrame(columns=COLUMNS)
                    df.to_csv(f, index=False)
                    print(f"Created {self.filename}")
            except FileExistsError:
                pass

    def append_row(self, row_dict):
        df = pd.DataFrame([row_dict])
        df = df.reindex(columns=COLUMNS)
        with FileLock(self.lockfile):
            df.to_csv(self.filename, mode="a", index=False, header=False)

    def existing_keys(self):
        """Return set of (thingi_id, level) already in CSV."""
        keys = set()
        try:
            for chunk in pd.read_csv(self.filename, chunksize=1024):
                for _, row in chunk.iterrows():
                    keys.add((int(row["thingi_id"]), int(row["level"])))
        except (pd.errors.EmptyDataError, KeyError, FileNotFoundError):
            pass
        return keys


def load_descriptor(path):
    """Load a RefinementDescriptor from a .bin file."""
    desc = RefinementDescriptor.from_file(str(path))
    disc = Discretization(MortonOrderLinearization(), desc)
    return disc


def descriptor_stats(disc):
    """Extract statistics from a Discretization."""
    desc = disc.descriptor
    n_nodes = len(desc)
    n_boxes = desc.get_num_boxes()
    topo_bits = n_nodes * DIMENSIONALITY  # bits in the descriptor
    ref_counts = Counter(ref.to01() for ref in desc)
    ref_counts.pop("0" * DIMENSIONALITY, None)  # remove leaf entries
    return {
        "nodes": n_nodes,
        "boxes": n_boxes,
        "topo_bits": topo_bits,
        "ref_counts": json.dumps(dict(ref_counts), sort_keys=True),
    }


def openvdb_topology_bits(vdb_file):
    """Query OpenVDB topology stats using the vdb_topology_bits tool."""
    tool = Path(__file__).parent / "vdb_topology_bits"
    if not tool.exists():
        return None

    result = subprocess.run(
        [str(tool), str(vdb_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  vdb_topology_bits failed: {result.stderr.strip()}")
        return None

    # Parse JSON lines output
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
            stats["inactive_leaf_voxels"] = obj.get("inactive_leaf_voxels")
            stats["active_tiles"] = obj.get("active_tiles")
            stats["leaf_count"] = obj.get("leaf_count")
            stats["total_leaf_coefficients"] = obj.get("total_leaf_coefficients")
            stats["total_coefficients"] = obj.get("total_coefficients")
            stats["dense_value_bits"] = obj["dense_value_bits"]
            stats["active_value_bits"] = obj["active_value_bits"]
    return stats if stats else None





def _sample_leaf_values_from_vdb(disc, vdb_path, level):
    """Sample the VDB BoolGrid at a finest-level midpoint within each leaf.

    Walks the descriptor tree to compute leaf coordinates directly (no
    per-box coordinate lookup).  The sample point is the leaf's lower-bound
    corner in VDB index space, matching _sample_at_finest_midpoints.

    Returns a bitarray of length n_boxes with 1 for inside, 0 for outside.
    """
    import bitarray as ba
    import openvdb as vdb

    grids, _ = vdb.readAll(vdb_path)
    grid = grids[0]
    acc = grid.getConstAccessor()

    descriptor = disc.descriptor
    dims = 1 << level
    n_boxes = descriptor.get_num_boxes()
    values = ba.bitarray(n_boxes)
    values.setall(0)

    # Build subtree-end array for O(1) sibling skipping
    import numpy as np
    n = len(descriptor)
    subtree_end = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        r = descriptor[i]
        if r.count() == 0:
            subtree_end[i] = i + 1
        else:
            c = i + 1
            for _ in range(1 << r.count()):
                c = int(subtree_end[c])
            subtree_end[i] = c

    # DFS walk: stack of (desc_index, ox, oy, oz, sx, sy, sz)
    stack = [(0, 0, 0, 0, dims, dims, dims)]
    leaf_idx = 0

    while stack:
        desc_i, ox, oy, oz, sx, sy, sz = stack.pop()
        ref = descriptor[desc_i]

        if ref.count() == 0:
            # Leaf: sample at lower-bound corner (= finest-level midpoint)
            if acc.isValueOn((ox, oy, oz)):
                values[leaf_idx] = 1
            leaf_idx += 1
            continue

        # Inner node: compute child sizes and push in reverse order
        csx = sx // 2 if ref[0] else sx
        csy = sy // 2 if ref[1] else sy
        csz = sz // 2 if ref[2] else sz

        # Precompute subtree-end for children
        num_children = 1 << ref.count()
        child_desc = desc_i + 1
        child_entries = []
        for child_idx in range(num_children):
            cox, coy, coz = ox, oy, oz
            local_bit = 0
            for d in range(3):
                if ref[d]:
                    if (child_idx >> local_bit) & 1:
                        if d == 0: cox += csx
                        elif d == 1: coy += csy
                        else: coz += csz
                    local_bit += 1
            child_entries.append((child_desc, cox, coy, coz, csx, csy, csz))
            child_desc = int(subtree_end[child_desc])

        for entry in reversed(child_entries):
            stack.append(entry)

    return values


def compressed_omnitree_sizes(descriptor_path: str, vdb_path: str = None,
                              level: int = None) -> dict:
    """Write binary coefficients and compress both descriptor and coefficients.

    Samples actual leaf occupancy values from the VDB BoolGrid.  If the VDB
    is not available, falls back to all-ones (assumes every leaf is inside).

    Writes:
      - {base}_coeffs.bin           (raw packed 1-bit coefficients)
      - {base}_3d.bin.blosc2        (blosc2-compressed descriptor)
      - {base}_coeffs.bin.blosc2    (blosc2-compressed coefficients)

    Returns dict with raw / blosc2 descriptor, coefficient, and total bytes.
    """
    import bitarray as ba

    disc = load_descriptor(descriptor_path)
    n_boxes = disc.descriptor.get_num_boxes()

    # Raw descriptor bytes
    desc_data = disc.descriptor.get_data().tobytes()

    # Sample actual leaf values from VDB
    if vdb_path is not None and level is not None:
        try:
            coeffs = _sample_leaf_values_from_vdb(disc, vdb_path, level)
        except Exception as e:
            print(f"    warning: could not sample VDB ({e}), falling back to all-ones")
            coeffs = ba.bitarray(n_boxes)
            coeffs.setall(1)
    else:
        coeffs = ba.bitarray(n_boxes)
        coeffs.setall(1)
    coeffs_data = coeffs.tobytes()

    # Write raw coefficients file
    base = descriptor_path.rsplit("_3d.bin", 1)[0]
    coeffs_path = base + "_coeffs.bin"
    with open(coeffs_path, "wb") as f:
        f.write(coeffs_data)

    result = {
        "raw_descriptor_bytes": len(desc_data),
        "raw_coeffs_bytes": len(coeffs_data),
        "raw_total_bytes": len(desc_data) + len(coeffs_data),
        "blosc2_descriptor_bytes": float("nan"),
        "blosc2_coeffs_bytes": float("nan"),
        "blosc2_total_bytes": float("nan"),
    }

    # Blosc2 compression (optional)
    try:
        import blosc2
        desc_blosc2 = blosc2.compress(desc_data, typesize=1)
        coeffs_blosc2 = blosc2.compress(coeffs_data, typesize=1)
        with open(descriptor_path + ".blosc2", "wb") as f:
            f.write(desc_blosc2)
        with open(coeffs_path + ".blosc2", "wb") as f:
            f.write(coeffs_blosc2)
        result["blosc2_descriptor_bytes"] = len(desc_blosc2)
        result["blosc2_coeffs_bytes"] = len(coeffs_blosc2)
        result["blosc2_total_bytes"] = len(desc_blosc2) + len(coeffs_blosc2)
    except ImportError:
        pass

    return result


def compressed_cloud_sizes(descriptor_path: str, values_path: str,
                           mask_path: str) -> dict:
    """Compress real-valued cloud omnitree data (descriptor + values + mask).

    The cloud representation stores:
      - descriptor (_3d.bin): tree topology
      - values (_values.npy): float64 non-background leaf values
      - mask (_nonbg_mask.npy): packed bits indicating which leaves are non-bg

    Returns dict with raw and blosc2 total sizes.
    """
    import numpy as np

    desc_data = open(descriptor_path, "rb").read()
    values_data = np.load(values_path).astype(np.float32).tobytes()
    mask_data = np.load(mask_path).tobytes()

    result = {
        "raw_descriptor_bytes": len(desc_data),
        "raw_values_bytes": len(values_data),
        "raw_mask_bytes": len(mask_data),
        "raw_total_bytes": len(desc_data) + len(values_data) + len(mask_data),
        "blosc2_total_bytes": float("nan"),
    }

    try:
        import blosc2
        d_c = blosc2.compress(desc_data, typesize=1)
        v_c = blosc2.compress(values_data, typesize=4)
        m_c = blosc2.compress(mask_data, typesize=1)
        result["blosc2_total_bytes"] = len(d_c) + len(v_c) + len(m_c)
    except ImportError:
        pass

    return result


def discover_cloud_files(directory):
    """Discover cloud output files: {name}_{variant}_3d.bin + _values.npy + _nonbg_mask.npy.

    Returns dict keyed by (grid_name, threshold, variant) with file paths.
    """
    pattern = re.compile(
        r"^(cloud_l[\dx]+)_t([\d.]+)_(\w+)_3d\.bin$"
    )
    octree_pattern = re.compile(
        r"^(cloud_l[\dx]+)_(octree)_3d\.bin$"
    )
    entries = {}
    for f in os.listdir(directory):
        m = pattern.match(f) or octree_pattern.match(f)
        if not m:
            continue
        grid_name = m.group(1)
        if m.lastindex == 3:
            threshold, variant = m.group(2), m.group(3)
        else:
            threshold, variant = "0.0", m.group(2)

        base = f.rsplit("_3d.bin", 1)[0]
        values_file = os.path.join(directory, base + "_values.npy")
        mask_file = os.path.join(directory, base + "_nonbg_mask.npy")
        vdb_file = os.path.join(directory, base + ".vdb")

        if not os.path.exists(values_file) or not os.path.exists(mask_file):
            continue

        key = (grid_name, threshold, variant)
        entries[key] = {
            "descriptor": os.path.join(directory, f),
            "values": values_file,
            "mask": mask_file,
            "vdb": vdb_file if os.path.exists(vdb_file) else None,
        }

    return entries


CLOUD_COLUMNS = [
    "grid_name", "threshold", "variant",
    "nodes", "boxes", "topo_bits",
    "raw_total_bytes", "blosc2_total_bytes",
    "vdb_file_bytes",
    "n_nonbg_values",
]

SWEEP_COLUMNS = [
    "threshold",
    "nodes", "boxes", "topo_bits",
    "n_leaves", "n_nonbg_leaves",
    "Linf_error", "L1_error", "L2_error",
    "desc_raw_bytes", "values_raw_bytes", "raw_total_bytes",
    "desc_blosc2_bytes", "values_blosc2_bytes", "blosc2_total_bytes",
    "vdb_orig_file_bytes",
    "vdb_orig_topo_bits", "vdb_orig_active_voxels", "vdb_orig_total_coefficients",
    "vdb_recon_file_bytes",
    "ratio_raw_vs_vdb", "ratio_blosc2_vs_vdb",
]


def process_cloud(grid_name, threshold, variant, files):
    """Extract stats for one cloud variant."""
    import numpy as np

    row = {
        "grid_name": grid_name,
        "threshold": threshold,
        "variant": variant,
    }

    disc = load_descriptor(files["descriptor"])
    s = descriptor_stats(disc)
    row["nodes"] = s["nodes"]
    row["boxes"] = s["boxes"]
    row["topo_bits"] = s["topo_bits"]

    vals = np.load(files["values"])
    row["n_nonbg_values"] = len(vals)

    cs = compressed_cloud_sizes(files["descriptor"], files["values"], files["mask"])
    row["raw_total_bytes"] = cs["raw_total_bytes"]
    row["blosc2_total_bytes"] = cs["blosc2_total_bytes"]

    if files.get("vdb"):
        row["vdb_file_bytes"] = os.path.getsize(files["vdb"])

    return row


def discover_files(directory):
    """Discover (thingi_id, level) pairs from canonical and pushdown files."""
    pattern = re.compile(r"^(\d+)_l(\d+)_canonical_3d\.bin$")
    pairs = {}
    for f in os.listdir(directory):
        m = pattern.match(f)
        if m:
            thingi_id = int(m.group(1))
            level = int(m.group(2))
            key = (thingi_id, level)
            pairs[key] = {
                "canonical": os.path.join(directory, f),
            }

    # Match pushdown files
    for key in pairs:
        thingi_id, level = key
        pd_file = os.path.join(directory, f"{thingi_id}_l{level}_pushdown_3d.bin")
        if os.path.exists(pd_file):
            pairs[key]["pushdown"] = pd_file
        ds_file = os.path.join(directory, f"{thingi_id}_l{level}_downsplit_3d.bin")
        if os.path.exists(ds_file):
            pairs[key]["downsplit"] = ds_file

        vdb_file = os.path.join(directory, f"{thingi_id}_l{level}_openvdb.vdb")
        if os.path.exists(vdb_file):
            pairs[key]["openvdb"] = vdb_file

    return pairs


def process_one(thingi_id, level, files):
    """Extract stats for one (thingi_id, level) combination."""
    row = {"thingi_id": thingi_id, "level": level}

    # Canonical
    if "canonical" in files:
        disc = load_descriptor(files["canonical"])
        s = descriptor_stats(disc)
        row["can_nodes"] = s["nodes"]
        row["can_boxes"] = s["boxes"]
        row["can_topo_bits"] = s["topo_bits"]
        row["can_ref_counts"] = s["ref_counts"]
        vdb_file = files.get("openvdb")
        cs = compressed_omnitree_sizes(files["canonical"], vdb_file, level)
        row["can_raw_total_bytes"] = cs["raw_total_bytes"]
        row["can_blosc2_total_bytes"] = cs["blosc2_total_bytes"]
        row["can_desc_raw_bytes"] = cs["raw_descriptor_bytes"]
        row["can_desc_blosc2_bytes"] = cs["blosc2_descriptor_bytes"]
        row["can_coeffs_raw_bytes"] = cs["raw_coeffs_bytes"]
        row["can_coeffs_blosc2_bytes"] = cs["blosc2_coeffs_bytes"]

    # Downsplit (pushdown)
    ds_file = None
    if "pushdown" in files or "downsplit" in files:
        ds_file = files.get("pushdown") or files.get("downsplit")
        disc = load_descriptor(ds_file)
        s = descriptor_stats(disc)
        row["ds_nodes"] = s["nodes"]
        row["ds_boxes"] = s["boxes"]
        row["ds_topo_bits"] = s["topo_bits"]
        row["ds_ref_counts"] = s["ref_counts"]
        vdb_file = files.get("openvdb")
        cs = compressed_omnitree_sizes(ds_file, vdb_file, level)
        row["ds_raw_total_bytes"] = cs["raw_total_bytes"]
        row["ds_blosc2_total_bytes"] = cs["blosc2_total_bytes"]
        row["ds_desc_raw_bytes"] = cs["raw_descriptor_bytes"]
        row["ds_desc_blosc2_bytes"] = cs["blosc2_descriptor_bytes"]
        row["ds_coeffs_raw_bytes"] = cs["raw_coeffs_bytes"]
        row["ds_coeffs_blosc2_bytes"] = cs["blosc2_coeffs_bytes"]
    
    # OpenVDB (all stats from the C++ vdb_topology_bits tool)
    if "openvdb" in files:
        row["vdb_file_bytes"] = os.path.getsize(files["openvdb"])
        vdb_stats = openvdb_topology_bits(files["openvdb"])
        if vdb_stats:
            row["vdb_topo_bits"] = vdb_stats.get("total_topology_bits")
            row["vdb_topo_bytes"] = vdb_stats.get("total_topology_bytes")
            row["vdb_active_leaf_voxels"] = vdb_stats.get("active_leaf_voxels")
            row["vdb_dense_value_bits"] = vdb_stats.get("dense_value_bits")
            row["vdb_active_value_bits"] = vdb_stats.get("active_value_bits")
            row["vdb_active_leaf"] = vdb_stats.get("active_leaf_voxels")
            row["vdb_inactive_leaf"] = vdb_stats.get("inactive_leaf_voxels")
            row["vdb_active_tiles"] = vdb_stats.get("active_tiles")
            row["vdb_total_leaf_coefficients"] = vdb_stats.get("total_leaf_coefficients")
            row["vdb_total_coefficients"] = vdb_stats.get("total_coefficients")

    return row


def discover_sweep_dirs(sweep_dir):
    """Discover thr_* directories with final_3d.bin + final_values.npy."""
    import glob
    dirs = []
    for d in sorted(glob.glob(os.path.join(sweep_dir, "thr_*"))):
        if not os.path.isdir(d):
            continue
        desc = os.path.join(d, "final_3d.bin")
        vals = os.path.join(d, "final_values.npy")
        meta = os.path.join(d, "metadata.json")
        if os.path.exists(desc) and os.path.exists(vals) and os.path.exists(meta):
            thr_label = os.path.basename(d).removeprefix("thr_")
            dirs.append((thr_label, d))
    return dirs


def process_sweep_threshold(thr_label, work_dir, grid, bbox_min, per_dim_levels,
                            vdb_orig_file_bytes, vdb_orig_stats):
    """Extract stats for one threshold in a cloud sweep."""
    import numpy as np

    row = {"threshold": thr_label}

    desc_file = os.path.join(work_dir, "final_3d.bin")
    vals_file = os.path.join(work_dir, "final_values.npy")

    desc = RefinementDescriptor.from_file(desc_file)
    disc = Discretization(MortonOrderLinearization(), desc)
    leaf_values = np.load(vals_file)
    background = float(grid.background)

    n_nodes = len(desc)
    n_boxes = desc.get_num_boxes()
    row["nodes"] = n_nodes
    row["boxes"] = n_boxes
    row["topo_bits"] = n_nodes * DIMENSIONALITY
    row["n_leaves"] = n_boxes
    row["n_nonbg_leaves"] = int(np.count_nonzero(leaf_values != background))

    # ── Error vs original VDB ─────────────────────────────────────────────
    from compare_cloud import _build_subtree_end
    grid_shape = tuple(1 << int(l) for l in per_dim_levels)
    gx, gy, gz = int(bbox_min[0]), int(bbox_min[1]), int(bbox_min[2])
    subtree_end = _build_subtree_end(desc)

    max_err = 0.0
    sum_abs_err = 0.0
    sum_sq_err = 0.0
    n_voxels = 0

    stack = [(0, 0, 0, 0, grid_shape[0], grid_shape[1], grid_shape[2])]
    leaf_idx = 0
    while stack:
        desc_i, ox, oy, oz, sx, sy, sz = stack.pop()
        ref = desc[desc_i]

        if ref.count() == 0:
            val = float(leaf_values[leaf_idx])
            leaf_idx += 1
            buf = np.full((sx, sy, sz), background, dtype=np.float32)
            grid.copyToArray(buf, ijk=(gx + ox, gy + oy, gz + oz))
            diff = np.abs(buf - val)
            err = float(diff.max())
            if err > max_err:
                max_err = err
            count = sx * sy * sz
            sum_abs_err += float(diff.sum())
            sum_sq_err += float((diff * diff).sum())
            n_voxels += count
            continue

        csx, csy, csz = sx, sy, sz
        if ref[0]: csx //= 2
        if ref[1]: csy //= 2
        if ref[2]: csz //= 2

        num_children = 1 << ref.count()
        child_desc = desc_i + 1
        child_entries = []
        for child_idx in range(num_children):
            cox, coy, coz = ox, oy, oz
            local_bit = 0
            for d in range(3):
                if ref[d]:
                    if (child_idx >> local_bit) & 1:
                        if d == 0: cox += csx
                        elif d == 1: coy += csy
                        else: coz += csz
                    local_bit += 1
            child_entries.append((child_desc, cox, coy, coz, csx, csy, csz))
            child_desc = int(subtree_end[child_desc])
        for entry in reversed(child_entries):
            stack.append(entry)

    # Function-space norms (volume-weighted, normalized by domain volume |Ω|):
    #   L_inf = max_x |e(x)|
    #   L1    = (1/|Ω|) * integral |e| dΩ  =  (1/N) * sum_i |e_i|
    #   L2    = sqrt( (1/|Ω|) * integral |e|^2 dΩ )  =  sqrt( (1/N) * sum_i |e_i|^2 )
    # Each voxel has equal volume |Ω|/N, so per-voxel summation is volume-weighted.
    row["Linf_error"] = max_err
    row["L1_error"] = sum_abs_err / n_voxels if n_voxels else 0.0
    row["L2_error"] = (sum_sq_err / n_voxels) ** 0.5 if n_voxels else 0.0

    # ── Raw and compressed sizes ──────────────────────────────────────────
    desc_data = open(desc_file, "rb").read()
    values_data = leaf_values.astype(np.float32).tobytes()

    row["desc_raw_bytes"] = len(desc_data)
    row["values_raw_bytes"] = len(values_data)
    row["raw_total_bytes"] = len(desc_data) + len(values_data)

    try:
        import blosc2
        d_c = blosc2.compress(desc_data, typesize=1)
        v_c = blosc2.compress(values_data, typesize=4)
        row["desc_blosc2_bytes"] = len(d_c)
        row["values_blosc2_bytes"] = len(v_c)
        row["blosc2_total_bytes"] = len(d_c) + len(v_c)
    except ImportError:
        row["desc_blosc2_bytes"] = float("nan")
        row["values_blosc2_bytes"] = float("nan")
        row["blosc2_total_bytes"] = float("nan")

    # ── VDB reference stats ───────────────────────────────────────────────
    row["vdb_orig_file_bytes"] = vdb_orig_file_bytes
    row["vdb_orig_topo_bits"] = vdb_orig_stats.get("total_topology_bits", float("nan"))
    row["vdb_orig_active_voxels"] = vdb_orig_stats.get("active_leaf_voxels", float("nan"))
    row["vdb_orig_total_coefficients"] = vdb_orig_stats.get(
        "total_coefficients", grid.activeVoxelCount()
    )

    # Reconstructed VDB file size (if present)
    recon_vdb = None
    for name in ("final.vdb",):
        p = os.path.join(work_dir, name)
        if os.path.exists(p):
            recon_vdb = p
            break
    # Also check for final_thr_*.vdb or final_perm*.vdb
    if recon_vdb is None:
        import glob
        candidates = glob.glob(os.path.join(work_dir, "final*.vdb"))
        if candidates:
            recon_vdb = candidates[0]
    row["vdb_recon_file_bytes"] = os.path.getsize(recon_vdb) if recon_vdb else float("nan")

    # Compression ratios
    vdb_bytes = vdb_orig_file_bytes
    row["ratio_raw_vs_vdb"] = row["raw_total_bytes"] / vdb_bytes if vdb_bytes else float("nan")
    blosc2 = row.get("blosc2_total_bytes", float("nan"))
    row["ratio_blosc2_vs_vdb"] = blosc2 / vdb_bytes if vdb_bytes and not pd.isna(blosc2) else float("nan")

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Extract storage statistics from wavelet compression output files."
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Directory containing thingi10k .bin and .vdb files",
    )
    parser.add_argument(
        "--cloud-dir",
        default=None,
        help="Directory containing cloud output files (_3d.bin, _values.npy, _nonbg_mask.npy)",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Base directory of a threshold sweep (contains thr_* subdirs with "
        "final_3d.bin + final_values.npy + metadata.json)",
    )
    parser.add_argument(
        "--vdb",
        default=None,
        help="Original VDB file (required for --sweep-dir error computation)",
    )
    parser.add_argument(
        "--grid-name",
        default="density",
        help="VDB grid name (default: density)",
    )
    parser.add_argument(
        "--csv",
        default=CSV_FILENAME,
        help=f"Output CSV file (default: {CSV_FILENAME})",
    )
    args = parser.parse_args()

    if args.dir is None and args.cloud_dir is None and args.sweep_dir is None:
        parser.error("at least one of --dir, --cloud-dir, or --sweep-dir is required")
    if args.sweep_dir is not None and args.vdb is None:
        parser.error("--vdb is required with --sweep-dir")

    # ── Thingi10k binary occupancy stats ───────────────────────────────────
    if args.dir is not None:
        csv_file = WaveletStatsFile(args.csv)
        existing = csv_file.existing_keys()

        pairs = discover_files(args.dir)
        print(f"Found {len(pairs)} (thingi_id, level) pairs in {args.dir}/")

        n_skipped = 0
        n_processed = 0

        for (thingi_id, level), files in sorted(pairs.items()):
            if (thingi_id, level) in existing:
                n_skipped += 1
                continue

            stages = [k for k in ("canonical", "pushdown", "downsplit", "openvdb") if k in files]
            print(f"  thingi={thingi_id}, level={level}: {', '.join(stages)}")

            row = process_one(thingi_id, level, files)
            csv_file.append_row(row)
            n_processed += 1

        print(f"\nDone: {n_processed} processed, {n_skipped} skipped (already in CSV).")
        print(f"Results in: {args.csv}")

    # ── Cloud real-valued stats ────────────────────────────────────────────
    if args.cloud_dir is not None:
        cloud_entries = discover_cloud_files(args.cloud_dir)
        if not cloud_entries:
            print(f"No cloud files found in {args.cloud_dir}/")
            return

        cloud_csv = args.csv.rsplit(".", 1)[0] + "_cloud.csv"
        print(f"\nFound {len(cloud_entries)} cloud variants in {args.cloud_dir}/")

        rows = []
        for (grid_name, threshold, variant), files in sorted(cloud_entries.items()):
            print(f"  {grid_name} t={threshold} {variant}")
            row = process_cloud(grid_name, threshold, variant, files)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.reindex(columns=CLOUD_COLUMNS)
        df.to_csv(cloud_csv, index=False)
        print(f"\nCloud results in: {cloud_csv}")

        # Print summary
        print(f"\n{'grid':>18s}  {'thresh':>6s}  {'variant':>10s}  "
              f"{'nodes':>8s}  {'boxes':>8s}  "
              f"{'raw':>10s}  {'blosc2':>10s}  {'vdb_file':>10s}  {'blosc2/vdb':>10s}")
        print("-" * 100)
        for _, r in df.iterrows():
            vdb = r.get("vdb_file_bytes", float("nan"))
            ratio = r["blosc2_total_bytes"] / vdb if pd.notna(vdb) and vdb > 0 else float("nan")
            print(f"{r['grid_name']:>18s}  {r['threshold']:>6s}  {r['variant']:>10s}  "
                  f"{int(r['nodes']):8d}  {int(r['boxes']):8d}  "
                  f"{int(r['raw_total_bytes']):10d}  "
                  f"{int(r['blosc2_total_bytes']) if pd.notna(r['blosc2_total_bytes']) else 'N/A':>10}  "
                  f"{int(vdb) if pd.notna(vdb) else 'N/A':>10}  "
                  f"{ratio:9.3f}x" if pd.notna(ratio) else "")

    # ── Cloud sweep stats (--sweep-dir) ───────────────────────────────────
    if args.sweep_dir is not None:
        import numpy as np
        from compare_cloud import read_vdb_grid, vdb_grid_extent, levels_from_extent

        sweep_dirs = discover_sweep_dirs(args.sweep_dir)
        if not sweep_dirs:
            print(f"No thr_* directories with final results in {args.sweep_dir}/")
            return

        print(f"\n=== Cloud sweep: {args.sweep_dir} ===")
        print(f"Source VDB: {args.vdb}")
        print(f"Thresholds: {len(sweep_dirs)}")

        # Load source VDB once
        grid = read_vdb_grid(args.vdb, args.grid_name)
        bbox_min, _, extent = vdb_grid_extent(grid)
        per_dim_levels = levels_from_extent(extent)
        # Cap to metadata's per_dim_levels if available
        with open(os.path.join(sweep_dirs[0][1], "metadata.json")) as f:
            meta = json.load(f)
        per_dim_levels = np.array(meta["per_dim_levels"])
        bbox_min = np.array(meta["bbox_min"])
        print(f"Grid: bbox_min={bbox_min.tolist()}, per_dim_levels={per_dim_levels.tolist()}")

        # VDB reference stats
        vdb_orig_file_bytes = os.path.getsize(args.vdb)
        vdb_orig_stats = openvdb_topology_bits(args.vdb) or {}

        # Python-side stats (always available)
        from compare_cloud import vdb_topology_stats_python
        py_stats = vdb_topology_stats_python(grid)

        def _v(key, fallback="?"):
            """Get a value from the C++ tool stats, or '?' if unavailable."""
            return vdb_orig_stats.get(key, fallback)

        grid_shape = tuple(1 << int(l) for l in per_dim_levels)
        total_voxels = 1
        for s in grid_shape:
            total_voxels *= s

        topo_bits = _v("total_topology_bits")
        topo_bytes = _v("total_topology_bytes")
        active_leaf_voxels = _v("active_leaf_voxels")
        active_tiles = _v("active_tiles")
        total_coeffs = _v("total_coefficients",
                          py_stats["active_voxels"])  # fallback: assumes no active tiles

        def _p(label, val, suffix=""):
            """Print a stat line, formatting ints with commas or showing '?' as-is."""
            if isinstance(val, int):
                print(f"  {label:<22s}{val:>12,}{suffix}")
            else:
                print(f"  {label:<22s}{str(val):>12}{suffix}")

        print(f"\n--- Original VDB: {args.vdb} ---")
        _p("File size:", vdb_orig_file_bytes, " bytes")
        print(f"  Grid name:              {args.grid_name}")
        print(f"  Background:             {grid.background}")
        print(f"  Active bbox:            {grid.evalActiveVoxelBoundingBox()}")
        print(f"  Extent:                 {extent.tolist()}")
        shape_str = " x ".join(str(s) for s in grid_shape)
        print(f"  Padded grid:            {shape_str} = {total_voxels:,} voxels")
        _p("Active leaf voxels:", active_leaf_voxels)
        _p("Active tiles:", active_tiles)
        _p("Total coefficients:", total_coeffs)
        _p("Topology bits:", topo_bits)
        _p("Topology bytes:", topo_bytes, " bytes")
        if isinstance(total_coeffs, int):
            coeff_bits = total_coeffs * 32
            _p("Value bits (f32):", coeff_bits)
            if isinstance(topo_bits, int):
                total_bits = topo_bits + coeff_bits
                dense_bits = total_voxels * 32
                _p("Total bits:", total_bits)
                _p("Dense grid bits:", dense_bits)
                print(f"  {'VDB vs dense:':<22s}{total_bits / dense_bits:>11.4f}x")
        # Python-side stats (always available, from pyopenvdb API)
        print(f"  --- pyopenvdb ---")
        _p("activeVoxelCount:", py_stats["active_voxels"])
        _p("Leaf nodes:", py_stats["leaf_count"])
        _p("Non-leaf nodes:", py_stats["non_leaf_count"])
        _p("Memory usage:", py_stats["mem_usage_bytes"], " bytes")
        print(f"  {'Node log2 dims:':<22s}{py_stats['node_log2_dims']}")
        print()

        rows = []
        for thr_label, work_dir in sweep_dirs:
            print(f"  thr={thr_label} ...", end="", flush=True)
            row = process_sweep_threshold(
                thr_label, work_dir, grid, bbox_min, per_dim_levels,
                vdb_orig_file_bytes, vdb_orig_stats,
            )
            rows.append(row)
            print(f" {row['nodes']:,} nodes, {row['boxes']:,} boxes, "
                  f"Linf={row['Linf_error']:.6f}, L1={row['L1_error']:.6f}, "
                  f"L2={row['L2_error']:.6f}")

        sweep_csv = args.csv.rsplit(".", 1)[0] + "_sweep.csv"
        df = pd.DataFrame(rows)
        df = df.reindex(columns=SWEEP_COLUMNS)
        df.to_csv(sweep_csv, index=False)
        print(f"\nSweep results in: {sweep_csv}")

        # Print summary table
        print(f"\n{'thresh':>8s}  {'nodes':>8s}  {'boxes':>8s}  "
              f"{'Linf':>10s}  {'L1':>10s}  {'L2':>10s}  "
              f"{'raw':>10s}  {'blosc2':>10s}  "
              f"{'raw/vdb':>8s}  {'bl2/vdb':>8s}")
        print("-" * 106)
        def _fmt(val, fmt="{:10d}", na="N/A"):
            try:
                if pd.isna(val):
                    return f"{na:>10}"
            except (TypeError, ValueError):
                pass
            return fmt.format(val)

        for _, r in df.iterrows():
            bl2 = r.get("blosc2_total_bytes", float("nan"))
            raw_ratio = r.get("ratio_raw_vs_vdb", float("nan"))
            bl2_ratio = r.get("ratio_blosc2_vs_vdb", float("nan"))
            bl2_s = _fmt(bl2, "{:10.0f}")
            raw_r_s = _fmt(raw_ratio, "{:7.3f}x", "    N/A")
            bl2_r_s = _fmt(bl2_ratio, "{:7.3f}x", "    N/A")
            print(
                f"{r['threshold']:>8s}  "
                f"{int(r['nodes']):8d}  {int(r['boxes']):8d}  "
                f"{r['Linf_error']:10.6f}  {r['L1_error']:10.6f}  {r['L2_error']:10.6f}  "
                f"{int(r['raw_total_bytes']):10d}  "
                f"{bl2_s}  {raw_r_s}  {bl2_r_s}"
            )


if __name__ == "__main__":
    main()
