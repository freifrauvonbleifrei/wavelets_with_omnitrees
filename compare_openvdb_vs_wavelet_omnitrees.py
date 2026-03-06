#!/usr/bin/env python3
"""Compare OpenVDB vs omnitree canonical coarsening vs pushdown (mnl=F / mnl=T)."""

import argparse as arg
import copy
from icecream import ic
import json
import numpy as np
import openvdb as vdb
from pathlib import Path
import subprocess
from collections import defaultdict

import dyada
import dyada.descriptor
import dyada.discretization
import dyada.linearization

try:
    from wavelets_with_omnitrees.thingies_with_wavelets_and_omnitrees import (
        THINGIES,
        midpoint_occupancy_coefficients,
        binary_values_on_full_grid,
    )
    from wavelets_with_omnitrees.wavelets_with_omnitrees import (
        transform_to_all_wavelet_coefficients,
        fill_scaling_from_hierarchical_coefficients,
        compress_by_omnitree_coarsening,
        compress_by_pushdown_coarsening,
    )
except ModuleNotFoundError:
    from thingies_with_wavelets_and_omnitrees import (  # type: ignore
        THINGIES,
        midpoint_occupancy_coefficients,
        binary_values_on_full_grid,
    )
    from wavelets_with_omnitrees import (  # type: ignore
        transform_to_all_wavelet_coefficients,
        fill_scaling_from_hierarchical_coefficients,
        compress_by_omnitree_coarsening,
        compress_by_pushdown_coarsening,
    )


# this function courtesy of claude
def openvdb_topology_bits(vdb_grid, grid_name: str | None = None) -> dict:
    """
    Returns a dict keyed by grid name. Each value contains:
      {
        "levels": [
          {
            "level": int,
            "type": str,          # e.g. "Leaf_8", "Internal_16", "Internal_32"
            "nodes": int,
            "mask_bits_per_node": int,
            "total_bits": int,
          },
          ...
        ],
        "total_topology_bits":  int,
        "total_topology_bytes": int,
      }

    Mask bits counted:
      Leaf nodes:     value mask only            (8^3  = 512 bits)
      Internal nodes: child mask + value mask    (16^3 = 8192 bits, 32^3 = 65536 bits)
      Root node:      excluded (sparse hash map, implementation-specific)
    Pointer storage, padding, and value arrays are excluded throughout.
    """
    vdb_file = "./tmp.vdb"
    vdb.write(vdb_file, grids=[vdb_grid])
    bin = Path(__file__).parent / "vdb_topology_bits"
    cmd = [str(bin), vdb_file]
    if grid_name:
        cmd.append(grid_name)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vdb_topology_bits failed:\n{result.stderr}")

    grids: dict = defaultdict(lambda: {"levels": []})

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        name = obj["grid"]
        if "summary" in obj:
            grids[name]["total_topology_bits"] = obj["total_topology_bits"]
            grids[name]["total_topology_bytes"] = obj["total_topology_bytes"]
        elif "value_summary" in obj:
            grids[name]["bits_per_value"] = obj["bits_per_value"]
            grids[name]["active_leaf_voxels"] = obj["active_leaf_voxels"]
            grids[name]["active_tiles"] = obj["active_tiles"]
            grids[name]["dense_value_bits"] = obj["dense_value_bits"]
            grids[name]["dense_value_bytes"] = obj["dense_value_bytes"]
            grids[name]["active_value_bits"] = obj["active_value_bits"]
            grids[name]["active_value_bytes"] = obj["active_value_bytes"]
        else:
            grids[name]["levels"].append(
                {
                    "level": obj["level"],
                    "type": obj["type"],
                    "nodes": obj["nodes"],
                    "mask_bits_per_node": obj["mask_bits_per_node"],
                    "total_bits": obj["total_bits"],
                }
            )

    return dict(grids)


def midpoint_occupancy_openvdb(inside_fn, level: int) -> vdb.BoolGrid:
    voxel_size = 2.0 ** (-level)
    dims = 1 << level
    grid = vdb.BoolGrid(False)
    grid.name = "inside"
    grid.transform = vdb.createLinearTransform(voxel_size)
    acc = grid.getAccessor()
    for i in range(dims):
        for j in range(dims):
            for k in range(dims):
                midpoint = np.array(
                    [
                        (i + 0.5) * voxel_size,
                        (j + 0.5) * voxel_size,
                        (k + 0.5) * voxel_size,
                    ],
                    dtype=np.float64,
                )
                if inside_fn(midpoint.reshape(1, 3))[0]:
                    acc.setValueOn((i, j, k), True)
    grid.pruneInactive()
    return grid


def reconstruct_and_check(full_disc, disc, coeff, reference_binary):
    """Recover scalings, reconstruct binary grid, return (desc, boxes, exact)."""
    coeff = copy.deepcopy(coeff)
    fill_scaling_from_hierarchical_coefficients(disc, coeff)
    scaling = np.array(
        [coeff[i][0] for i in range(len(coeff)) if disc.descriptor.is_box(i)],
        dtype=np.float32,
    )
    recon = binary_values_on_full_grid(full_disc, disc, scaling)
    exact = bool(np.array_equal(recon, reference_binary))
    return len(disc.descriptor), len(disc), exact, recon


def run_one(thingy_name: str, inside_fn, level: int):
    full_discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(3, [level, level, level]),
    )
    full_occupancy = midpoint_occupancy_coefficients(inside_fn, level)
    reference_binary = full_occupancy >= 0.5

    coefficients = transform_to_all_wavelet_coefficients(
        full_discretization, full_occupancy
    )
    root_scaling = coefficients[0][0]
    for coeff in coefficients:
        coeff[0] = np.nan
    coefficients[0][0] = root_scaling

    results = {}
    reconstructions = {}

    # ── omnitree (canonical coarsening) ──────────────────────────────────────
    disc, coeff = compress_by_omnitree_coarsening(
        full_discretization,
        [list(c) for c in coefficients],
        coarsening_threshold=0.0,
    )
    nd, nb, exact, recon = reconstruct_and_check(
        full_discretization, disc, coeff, reference_binary
    )
    results["canonical"] = (nd, nb, exact)
    reconstructions["canonical"] = recon

    # ── pushdown mnl=False ───────────────────────────────────────────────────
    disc, coeff = compress_by_pushdown_coarsening(
        full_discretization,
        [list(c) for c in coefficients],
        coarsening_threshold=0.0,
        merge_non_leaf=False,
    )
    nd, nb, exact, recon = reconstruct_and_check(
        full_discretization, disc, coeff, reference_binary
    )
    results["push(mnl=F)"] = (nd, nb, exact)
    reconstructions["push(mnl=F)"] = recon

    # ── pushdown mnl=True ────────────────────────────────────────────────────
    disc, coeff = compress_by_pushdown_coarsening(
        full_discretization,
        [list(c) for c in coefficients],
        coarsening_threshold=0.0,
        merge_non_leaf=True,
    )
    nd, nb, exact, recon = reconstruct_and_check(
        full_discretization, disc, coeff, reference_binary
    )
    results["push(mnl=T)"] = (nd, nb, exact)
    reconstructions["push(mnl=T)"] = recon

    # ── OpenVDB ──────────────────────────────────────────────────────────────
    openvdb_grid = midpoint_occupancy_openvdb(inside_fn, level=level)

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\nthingy={thingy_name}, level={level}")
    print(
        "  openvdb: ",
        f"active_voxels={openvdb_grid.activeVoxelCount()}",
        f"leaf_count={openvdb_grid.leafCount()}",
        f"total_count={openvdb_grid.nonLeafCount()+openvdb_grid.leafCount()}",
        f"mem_usage={openvdb_grid.memUsage()}",
    )
    print(f"  {'method':18s}  {'desc':>6}  {'boxes':>6}  exact_match")
    print("  " + "-" * 50)
    for label, (nd, nb, ok) in results.items():
        print(f"  {label:18s}  {nd:6d}  {nb:6d}  {ok}")

    # ── Assertions ───────────────────────────────────────────────────────────
    for label, recon in reconstructions.items():
        assert np.array_equal(recon, reference_binary), (
            f"{thingy_name} level={level} {label}: "
            f"reconstruction does not match reference"
        )

    labels = list(reconstructions.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert np.array_equal(
                reconstructions[labels[i]], reconstructions[labels[j]]
            ), (
                f"{thingy_name} level={level}: "
                f"{labels[i]} and {labels[j]} reconstructions differ"
            )


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument(
        "--thingy",
        type=str,
        default="all",
        choices=["all", "sphere", "tetrahedron", "diagonal_rod"],
    )
    args = parser.parse_args()

    selected = THINGIES.items()
    if args.thingy != "all":
        selected = [(args.thingy, THINGIES[args.thingy])]

    for thingy_name, inside_fn in selected:
        run_one(thingy_name, inside_fn, level=args.level)
