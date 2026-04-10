#!/usr/bin/env python3
"""Run compare_openvdb_vs_wavelet_omnitrees.py in parallel over thingi10k slices."""

import argparse
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

from icecream import ic

SCRIPT = os.path.join(os.path.dirname(__file__), "compare_openvdb_vs_wavelet_omnitrees.py")
SLICES_PER_WORKER = 32


def run_slice(slice_string: str, level: int, output_dir: str):
    ic(slice_string)
    cmd = [
        sys.executable, "-O", SCRIPT,
        "--thingy", "thingi10k",
        "--slice", slice_string,
        "--level", str(level),
        "--output-dir", output_dir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"slice {slice_string} FAILED:\n{proc.stderr}", flush=True)
    else:
        print(proc.stdout, end="", flush=True)
    return proc.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel driver for compare_openvdb_vs_wavelet_omnitrees.py"
    )
    parser.add_argument("--max-level", type=int, default=8)
    parser.add_argument(
        "--num-slices", type=int, default=None,
        help="number of slices to split the thingi10k dataset into "
        f"(default: {SLICES_PER_WORKER} * number of workers)",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    num_workers = max(1, os.cpu_count() - 9)
    num_slices = (
        args.num_slices if args.num_slices is not None
        else num_workers * SLICES_PER_WORKER
    )
    ic(num_workers, num_slices)

    tp = ThreadPool(num_workers)
    for level in range(2, args.max_level):
        for i in reversed(range(num_slices)):
            slice_str = f"{i}/{num_slices}"
            tp.apply_async(run_slice, (slice_str, level, args.output_dir))

    tp.close()
    tp.join()


