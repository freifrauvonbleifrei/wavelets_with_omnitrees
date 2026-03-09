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
        THINGIES as _THINGIES_BY_NAME,
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
        THINGIES as _THINGIES_BY_NAME,
        midpoint_occupancy_coefficients,
        binary_values_on_full_grid,
    )
    from wavelets_with_omnitrees import (  # type: ignore
        transform_to_all_wavelet_coefficients,
        fill_scaling_from_hierarchical_coefficients,
        compress_by_omnitree_coarsening,
        compress_by_pushdown_coarsening,
    )

try:
    import thingi10k
    import trimesh

    HAS_THINGI10K = True
except ImportError:
    HAS_THINGI10K = False

# Fake thingi IDs for the analytic shapes (matching special_thingies.py).
ANALYTIC_THINGIES: dict[int, callable] = {
    0: _THINGIES_BY_NAME["tetrahedron"],
    1: _THINGIES_BY_NAME["sphere"],
    2: _THINGIES_BY_NAME["diagonal_rod"],
}


def mesh_to_unit_cube(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    mesh.apply_scale(1.0 / mesh.extents.max())
    bounds_mean = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-bounds_mean + 0.5)
    return mesh


def load_thingi10k_thingies(
    file_ids: list[int] | None = None,
    slice_str: str | None = None,
) -> list[tuple[int, callable]]:
    """Load thingi10k meshes as (file_id, inside_fn) pairs.

    file_ids: explicit list of thingi10k file IDs to use.
    slice_str: "i/n" to select slice i of n from the default subset.
    """
    if not HAS_THINGI10K:
        raise RuntimeError(
            "thingi10k and trimesh are required for thingi10k thingies. "
            "Install with: pip install thingi10k trimesh"
        )

    thingi10k.init()

    if file_ids is not None:
        subset = thingi10k.dataset(file_id=file_ids)
    else:
        subset = thingi10k.dataset(
            num_vertices=(None, 10000),
            closed=True,
            self_intersecting=False,
            solid=True,
        )
        if slice_str is not None:
            parts = slice_str.split("/")
            my_slice = int(parts[0])
            num_slices = int(parts[1])
            all_ids = subset["file_id"]
            chunk_size = len(all_ids) / num_slices
            start = round(my_slice * chunk_size)
            end = round((my_slice + 1) * chunk_size)
            subset = thingi10k.dataset(file_id=all_ids[start:end])

    thingies = []
    for thingi in subset:
        mesh_data = np.load(thingi["file_path"])
        mesh = trimesh.Trimesh(
            vertices=mesh_data["vertices"], faces=mesh_data["facets"]
        )
        try:
            if not mesh.is_watertight:
                print(f"  skipping thingi {thingi['file_id']}: not watertight")
                continue
        except IndexError:
            print(f"  skipping thingi {thingi['file_id']}: degenerate mesh")
            continue
        mesh = mesh_to_unit_cube(mesh)
        thingies.append((thingi["file_id"], lambda pts, m=mesh: m.contains(pts)))

    return thingies


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
    assert exact
    return len(disc.descriptor), len(disc), recon


def run_one(thingy_id: int, inside_fn, level: int, output_dir: Path):
    dimensionality = 3
    full_discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(dimensionality, [level, level, level]),
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
    descriptors = {}

    # ── omnitree (canonical coarsening) ──────────────────────────────────────
    disc_can, coeff_can = compress_by_omnitree_coarsening(
        full_discretization,
        [list(c) for c in coefficients],
        coarsening_threshold=0.0,
    )
    nd, nb, recon = reconstruct_and_check(
        full_discretization, disc_can, coeff_can, reference_binary
    )
    topology_bits = nd * dimensionality
    results["canonical"] = (topology_bits, nb)
    reconstructions["canonical"] = recon
    descriptors["canonical"] = disc_can.descriptor

    # ── pushdown (starting from canonical) ───────────────────────────────────
    disc_pd, coeff_pd = compress_by_pushdown_coarsening(
        disc_can,
        [list(c) for c in coeff_can],
        coarsening_threshold=0.0,
    )
    nd, nb, recon = reconstruct_and_check(
        full_discretization, disc_pd, coeff_pd, reference_binary
    )
    topology_bits = nd * dimensionality
    results["pushdown"] = (topology_bits, nb)
    reconstructions["pushdown"] = recon
    descriptors["pushdown"] = disc_pd.descriptor

    # ── OpenVDB ──────────────────────────────────────────────────────────────
    openvdb_grid = midpoint_occupancy_openvdb(inside_fn, level=level)
    openvdb_stats = openvdb_topology_bits(openvdb_grid)
    results["openvdb"] = (
        openvdb_stats["inside"]["total_topology_bits"],
        openvdb_stats["inside"]["dense_value_bytes"],
    )

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\nthingy={thingy_id}, level={level}")
    print(
        f"  {'method':<18s} {'topo bits':>10s} {'value bits':>11s} {'total bits':>11s}"
    )
    print(f"  {'-' * 52}")
    for label, (nt, nb) in results.items():
        print(f"  {label:<18s} {nt:10d} {nb:11d} {nt + nb:11d}")

    # ── Assertions ───────────────────────────────────────────────────────────
    for label, recon in reconstructions.items():
        assert np.array_equal(recon, reference_binary), (
            f"thingy {thingy_id} level={level} {label}: "
            f"reconstruction does not match reference"
        )

    labels = list(reconstructions.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert np.array_equal(
                reconstructions[labels[i]], reconstructions[labels[j]]
            ), (
                f"thingy {thingy_id} level={level}: "
                f"{labels[i]} and {labels[j]} reconstructions differ"
            )

    # ── Write descriptor and VDB files ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{thingy_id}_l{level}"
    for label, desc in descriptors.items():
        desc.to_file(str(output_dir / f"{prefix}_{label}"))
    vdb_path = str(output_dir / f"{prefix}_openvdb.vdb")
    vdb.write(vdb_path, grids=[openvdb_grid])
    print(f"  wrote descriptors and VDB to {output_dir}/")


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument(
        "--thingy",
        type=str,
        default="analytic",
        choices=["all", "analytic", "thingi10k"],
        help="which thingies to run: 'analytic' (default) for built-in shapes "
        "(IDs 0-2), 'thingi10k' for thingi10k meshes, 'all' for both",
    )
    parser.add_argument(
        "--thingy-id",
        type=int,
        nargs="+",
        default=None,
        help="specific thingy IDs to run; analytic IDs (0=tetrahedron, "
        "1=sphere, 2=diagonal_rod) are resolved locally, other IDs "
        "are looked up in thingi10k",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="which slice of the thingi10k dataset to process, e.g. '0/8' "
        "for slice 0 of 8 (for parallelization)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="directory for descriptor and VDB files",
    )
    args = parser.parse_args()

    selected: list[tuple[int, callable]] = []

    if args.thingy_id is not None:
        # ── explicit IDs: split into analytic vs thingi10k ───────────────
        thingi10k_ids = []
        for tid in args.thingy_id:
            if tid in ANALYTIC_THINGIES:
                selected.append((tid, ANALYTIC_THINGIES[tid]))
            else:
                thingi10k_ids.append(tid)
        if thingi10k_ids:
            selected.extend(load_thingi10k_thingies(file_ids=thingi10k_ids))
    else:
        # ── category-based selection ─────────────────────────────────────
        if args.thingy in ("all", "analytic"):
            selected.extend(ANALYTIC_THINGIES.items())
        if args.thingy in ("all", "thingi10k"):
            selected.extend(load_thingi10k_thingies(slice_str=args.slice))

    if not selected:
        parser.error("no thingies selected")

    output_dir = Path(args.output_dir)
    for thingy_id, inside_fn in selected:
        run_one(thingy_id, inside_fn, level=args.level, output_dir=output_dir)
