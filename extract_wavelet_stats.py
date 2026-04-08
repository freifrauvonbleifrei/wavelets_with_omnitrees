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

    # Downsplit (pushdown)
    if "pushdown" in files or "downsplit" in files:
        try:
            disc = load_descriptor(files["pushdown"])
        except KeyError:
            disc = load_descriptor(files["downsplit"])
        s = descriptor_stats(disc)
        row["ds_nodes"] = s["nodes"]
        row["ds_boxes"] = s["boxes"]
        row["ds_topo_bits"] = s["topo_bits"]
        row["ds_ref_counts"] = s["ref_counts"]
    
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
        default="output",
        help="Directory containing the .bin and .vdb files (default: output/)",
    )
    parser.add_argument(
        "--csv",
        default=CSV_FILENAME,
        help=f"Output CSV file (default: {CSV_FILENAME})",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
