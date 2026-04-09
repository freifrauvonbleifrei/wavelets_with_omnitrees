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
    # Level sweep coarsening (optional)
    "ls_nodes",
    "ls_boxes",
    "ls_topo_bits",
    "ls_ref_counts",
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
    "can_lz4_total_bytes",
    # Compressed omnitree sizes (downsplit)
    "ds_raw_total_bytes",
    "ds_blosc2_total_bytes",
    "ds_lz4_total_bytes",
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
      - {base}_3d.bin.lz4           (lz4-compressed descriptor)
      - {base}_coeffs.bin.lz4       (lz4-compressed coefficients)

    Returns dict with keys:
      raw_total_bytes, blosc2_total_bytes, lz4_total_bytes
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
        "blosc2_total_bytes": float("nan"),
        "lz4_total_bytes": float("nan"),
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
        result["blosc2_total_bytes"] = len(desc_blosc2) + len(coeffs_blosc2)
    except ImportError:
        pass

    # LZ4 compression (optional)
    try:
        import lz4.frame
        desc_lz4 = lz4.frame.compress(desc_data)
        coeffs_lz4 = lz4.frame.compress(coeffs_data)
        with open(descriptor_path + ".lz4", "wb") as f:
            f.write(desc_lz4)
        with open(coeffs_path + ".lz4", "wb") as f:
            f.write(coeffs_lz4)
        result["lz4_total_bytes"] = len(desc_lz4) + len(coeffs_lz4)
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

    Returns dict with raw, blosc2, and lz4 total sizes.
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
        "lz4_total_bytes": float("nan"),
    }

    try:
        import blosc2
        d_c = blosc2.compress(desc_data, typesize=1)
        v_c = blosc2.compress(values_data, typesize=4)
        m_c = blosc2.compress(mask_data, typesize=1)
        result["blosc2_total_bytes"] = len(d_c) + len(v_c) + len(m_c)
    except ImportError:
        pass

    try:
        import lz4.frame
        d_c = lz4.frame.compress(desc_data)
        v_c = lz4.frame.compress(values_data)
        m_c = lz4.frame.compress(mask_data)
        result["lz4_total_bytes"] = len(d_c) + len(v_c) + len(m_c)
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
    "raw_total_bytes", "blosc2_total_bytes", "lz4_total_bytes",
    "vdb_file_bytes",
    "n_nonbg_values",
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
    row["lz4_total_bytes"] = cs["lz4_total_bytes"]

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
        row["can_lz4_total_bytes"] = cs["lz4_total_bytes"]

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
        row["ds_lz4_total_bytes"] = cs["lz4_total_bytes"]
    
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
        "--csv",
        default=CSV_FILENAME,
        help=f"Output CSV file (default: {CSV_FILENAME})",
    )
    args = parser.parse_args()

    if args.dir is None and args.cloud_dir is None:
        parser.error("at least one of --dir or --cloud-dir is required")

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

        import pandas as pd
        df = pd.DataFrame(rows)
        df = df.reindex(columns=CLOUD_COLUMNS)
        df.to_csv(cloud_csv, index=False)
        print(f"\nCloud results in: {cloud_csv}")

        # Print summary
        print(f"\n{'grid':>18s}  {'thresh':>6s}  {'variant':>10s}  "
              f"{'nodes':>8s}  {'boxes':>8s}  "
              f"{'raw':>10s}  {'blosc2':>10s}  {'lz4':>10s}  {'vdb_file':>10s}  {'blosc2/vdb':>10s}")
        print("-" * 110)
        for _, r in df.iterrows():
            vdb = r.get("vdb_file_bytes", float("nan"))
            ratio = r["blosc2_total_bytes"] / vdb if pd.notna(vdb) and vdb > 0 else float("nan")
            print(f"{r['grid_name']:>18s}  {r['threshold']:>6s}  {r['variant']:>10s}  "
                  f"{int(r['nodes']):8d}  {int(r['boxes']):8d}  "
                  f"{int(r['raw_total_bytes']):10d}  "
                  f"{int(r['blosc2_total_bytes']) if pd.notna(r['blosc2_total_bytes']) else 'N/A':>10}  "
                  f"{int(r['lz4_total_bytes']) if pd.notna(r['lz4_total_bytes']) else 'N/A':>10}  "
                  f"{int(vdb) if pd.notna(vdb) else 'N/A':>10}  "
                  f"{ratio:9.3f}x" if pd.notna(ratio) else "")


if __name__ == "__main__":
    main()
