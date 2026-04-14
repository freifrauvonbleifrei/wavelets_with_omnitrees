#!/usr/bin/env python3
"""Compare OpenVDB vs omnitree canonical coarsening vs downsplit vs level-sweep."""

import argparse as arg
from icecream import ic
import json
import numpy as np
from pathlib import Path
import subprocess
from collections import defaultdict

try:
    import openvdb as vdb

    HAS_OPENVDB = True
except ImportError:
    HAS_OPENVDB = False


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
    from wavelets_with_omnitrees.thingies_with_wavelets_and_omnitrees import (
        THINGIES as _THINGIES_BY_NAME,
        midpoint_occupancy_coefficients,
        binary_values_on_full_grid,
    )
    from wavelets_with_omnitrees.wavelets_with_omnitrees import (
        transform_to_all_wavelet_coefficients,
        get_leaf_scalings,
        compress_by_omnitree_coarsening,
        compress_by_downsplit_coarsening,
        omnitree_from_vdb,
        read_vdb_grid,
        stream_inside_fn_to_float_vdb,
        fill_omnitree_into_vdb,
        #compress_by_level_sweep_coarsening,
    )
except ModuleNotFoundError:
    from thingies_with_wavelets_and_omnitrees import (  # type: ignore
        THINGIES as _THINGIES_BY_NAME,
        midpoint_occupancy_coefficients,
        binary_values_on_full_grid,
    )
    from wavelets_with_omnitrees import (  # type: ignore
        transform_to_all_wavelet_coefficients,
        get_leaf_scalings,
        compress_by_omnitree_coarsening,
        compress_by_downsplit_coarsening,
        omnitree_from_vdb,
        read_vdb_grid,
        stream_inside_fn_to_float_vdb,
        fill_omnitree_into_vdb,
        #compress_by_level_sweep_coarsening,
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
def openvdb_topology_bits(vdb_file: str, grid_name: str | None = None) -> dict:
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


def _write_inside_bool_vdb(disc_init, init_leaf_vals, level: int, vdb_path: str):
    """Write a sparse VDB BoolGrid for the binary inside-occupancy data.

    Uses fill_omnitree_into_vdb so the BoolGrid is built from the adaptive
    omnitree's leaves directly via grid.fill() — no dense intermediate.
    """
    voxel_size = 1.0 / (1 << level)
    grid = vdb.BoolGrid(False)
    grid.name = "inside"
    grid.transform = vdb.createLinearTransform(voxel_size)
    fill_omnitree_into_vdb(
        grid,
        disc_init,
        init_leaf_vals,
        [level] * 3,
        np.array([0, 0, 0], dtype=np.int64),
        value_transform=lambda v: bool(v > 0.5),
        skip_value=False,
    )
    vdb.write(vdb_path, grids=[grid])


def reconstruct_and_check(full_disc, disc, coeff, reference_binary):
    """Recover scalings, reconstruct binary grid, return (desc, boxes, exact)."""
    scaling = get_leaf_scalings(disc, coeff).astype(np.float32)
    recon = binary_values_on_full_grid(full_disc, disc, scaling)
    if not np.array_equal(recon, reference_binary):
        diff_idx = np.flatnonzero(recon != reference_binary)
        raise RuntimeError(
            f"reconstruction mismatch: {len(diff_idx)} / {recon.size} voxels differ "
            f"(first few morton indices: {diff_idx[:8].tolist()})"
        )
    return len(disc.descriptor), len(disc), recon


def output_files_for(thingy_id: int, level: int, output_dir: Path) -> list[Path]:
    """Return the list of output files that run_one produces for a given thingy."""
    prefix = f"{thingy_id}_l{level}"
    files = [
        output_dir / f"{prefix}_canonical_3d.bin",
        output_dir / f"{prefix}_downsplit_3d.bin",
    ]
    if HAS_OPENVDB:
        files.append(output_dir / f"{prefix}_openvdb.vdb")
    return files


def _sample_at_finest_midpoints(
    discretization: dyada.discretization.Discretization,
    inside_fn,
    level: int,
) -> np.ndarray:
    """Sample inside_fn at a finest-level midpoint within each leaf box.

    Fallback path used only when OpenVDB is not available; the VDB-backed
    pipeline reads leaf values from the on-disk grid via
    _read_leaf_values_from_vdb instead.
    """
    n_boxes = len(discretization)
    dim = discretization.descriptor.get_num_dimensions()
    half_voxel = 0.5 / (1 << level)
    midpoints = np.empty((n_boxes, dim), dtype=np.float64)
    for box_index in range(n_boxes):
        interval = dyada.discretization.coordinates_from_box_index(
            discretization, box_index
        )
        midpoints[box_index] = interval.lower_bound + half_voxel
    return inside_fn(midpoints).astype(np.float64)


def _read_leaf_values_from_vdb(
    discretization: dyada.discretization.Discretization,
    vdb_grid,
    level: int,
) -> np.ndarray:
    """Read one voxel value per leaf from a VDB grid.

    Looks up the lower-corner voxel of each leaf region. This is correct for
    trees built from uniform-region collapsing (e.g. via omnitree_from_vdb),
    where each leaf region holds a single value across all of its voxels.
    """
    n_boxes = len(discretization)
    finest_dims = 1 << level
    leaf_values = np.empty(n_boxes, dtype=np.float64)
    acc = vdb_grid.getConstAccessor()
    for box_index in range(n_boxes):
        interval = dyada.discretization.coordinates_from_box_index(
            discretization, box_index
        )
        ijk = (
            int(round(float(interval.lower_bound[0]) * finest_dims)),
            int(round(float(interval.lower_bound[1]) * finest_dims)),
            int(round(float(interval.lower_bound[2]) * finest_dims)),
        )
        leaf_values[box_index] = float(acc.getValue(ijk))
    return leaf_values


def _hierarchize_on_tree(
    discretization: dyada.discretization.Discretization,
    leaf_values: np.ndarray,
) -> list:
    """Hierarchize leaf values on an arbitrary tree, returning wavelet coefficients."""
    coefficients = transform_to_all_wavelet_coefficients(discretization, leaf_values)
    root_scaling = coefficients[0][0]
    for coeff in coefficients:
        coeff[0] = np.nan
    coefficients[0][0] = root_scaling
    return coefficients


def _load_discretization(
    descriptor_file: Path,
) -> dyada.discretization.Discretization:
    """Load a descriptor from file and build a Discretization."""
    desc = dyada.descriptor.RefinementDescriptor.from_file(str(descriptor_file))
    return dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(), desc
    )


def run_one(thingy_id: int, inside_fn, level: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    dimensionality = 3
    prefix = f"{thingy_id}_l{level}"

    canonical_file = output_dir / f"{prefix}_canonical_3d.bin"
    downsplit_file = output_dir / f"{prefix}_downsplit_3d.bin"
    vdb_file = output_dir / f"{prefix}_openvdb.vdb"

    need_canonical = not canonical_file.exists()
    need_downsplit = not downsplit_file.exists()
    need_openvdb = HAS_OPENVDB and not vdb_file.exists()

    if not (need_canonical or need_downsplit or need_openvdb):
        print(f"  skipping thingy {thingy_id}: all output files exist")
        return

    full_discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(dimensionality, [level] * dimensionality),
    )

    results = {}
    reconstructions = {}
    descriptors = {}

    # ── Decide how to reconstruct coefficients ───────────────────────────────
    # Walk the pipeline backwards to find the latest existing result we can
    # start from, then sample + hierarchize on that tree to reconstruct
    # wavelet coefficients cheaply (instead of resampling the full grid).

    disc_can = coeff_can = None
    disc_pd = coeff_pd = None
    reference_binary = None

    def _ensure_reference_binary():
        nonlocal reference_binary
        if reference_binary is not None:
            return
        if HAS_OPENVDB and (loaded_vdb_grid is not None or vdb_file.exists()):
            grid = loaded_vdb_grid if loaded_vdb_grid is not None \
                else read_vdb_grid(str(vdb_file), "inside")
            ref_disc, ref_vals = omnitree_from_vdb(
                grid,
                np.array([0, 0, 0], dtype=np.int64),
                np.array([level] * dimensionality),
            )
            reference_binary = binary_values_on_full_grid(
                full_discretization, ref_disc,
                ref_vals.astype(np.float32) >= 0.5,
            )
        else:
            full_occupancy = midpoint_occupancy_coefficients(inside_fn, level)
            reference_binary = full_occupancy >= 0.5

    # ── Build the adaptive omnitree from inside_fn (no full dense array),
    #    write the OpenVDB file from those leaves, then load it back from disk.
    # Any subsequent pipeline step (canonical / downsplit / level_sweep) reads
    # leaf values from this loaded grid instead of resampling inside_fn.
    disc_init = init_leaf_vals = None
    loaded_vdb_grid = None
    needs_any_vdb = HAS_OPENVDB and (
        need_openvdb or need_canonical or need_downsplit
    )

    if needs_any_vdb:
        # Mirroring compare_cloud's flow: stream inside_fn into a FloatGrid
        # slice-by-slice (no dims^3 dense array), then build the adaptive
        # octree via omnitree_from_vdb. The same omnitree feeds both the
        # OpenVDB output (a sparse BoolGrid via fill_omnitree_into_vdb) and
        # the canonical/downsplit/level_sweep pipeline below.
        if need_canonical or not vdb_file.exists():
            float_grid = stream_inside_fn_to_float_vdb(inside_fn, level)
            disc_init, init_leaf_vals = omnitree_from_vdb(
                float_grid,
                np.array([0, 0, 0], dtype=np.int64),
                np.array([level] * dimensionality),
            )
            del float_grid
        if not vdb_file.exists():
            _write_inside_bool_vdb(disc_init, init_leaf_vals, level, str(vdb_file))
        if need_openvdb:
            try:
                openvdb_stats = openvdb_topology_bits(str(vdb_file))
                results["openvdb"] = (
                    openvdb_stats["inside"]["total_topology_bits"],
                    openvdb_stats["inside"]["dense_value_bytes"],
                )
            except RuntimeError as e:
                results["openvdb"] = (-1,-1)
        loaded_vdb_grid = read_vdb_grid(str(vdb_file), "inside")

    if need_canonical:
        if HAS_OPENVDB:
            # Hierarchize on the adaptive (octree) tree, then canonical-coarsen
            # to recover the anisotropic refinements that compress_by_omnitree
            # would have produced from a full-grid start.
            init_coefficients = _hierarchize_on_tree(disc_init, init_leaf_vals)
            disc_can, coeff_can, _ = compress_by_omnitree_coarsening(
                disc_init,
                [list(c) for c in init_coefficients],
                coarsening_threshold=0.0,
            )
        else:
            # Fallback: full grid hierarchization + canonical coarsening
            full_occupancy = midpoint_occupancy_coefficients(inside_fn, level)
            reference_binary = full_occupancy >= 0.5
            coefficients = _hierarchize_on_tree(full_discretization, full_occupancy)
            disc_can, coeff_can, _ = compress_by_omnitree_coarsening(
                full_discretization,
                [list(c) for c in coefficients],
                coarsening_threshold=0.0,
            )
    elif need_downsplit: # or need_level_sweep:
        # Canonical exists — load and reconstruct coefficients
        disc_can = _load_discretization(canonical_file)
        if HAS_OPENVDB:
            leaf_values = _read_leaf_values_from_vdb(disc_can, loaded_vdb_grid, level)
        else:
            leaf_values = _sample_at_finest_midpoints(disc_can, inside_fn, level)
        coeff_can = _hierarchize_on_tree(disc_can, leaf_values)

    # free the initial-tree memory once canonical work is done
    disc_init = init_leaf_vals = None

    if need_canonical:
        _ensure_reference_binary()
        nd, nb, recon = reconstruct_and_check(
            full_discretization, disc_can, coeff_can, reference_binary
        )
        results["canonical"] = (nd * dimensionality, nb)
        reconstructions["canonical"] = recon
        descriptors["canonical"] = disc_can.descriptor

    # ── downsplit ────────────────────────────────────────────────────────────
    if need_downsplit:
        disc_pd, coeff_pd, _ = compress_by_downsplit_coarsening(
            disc_can,
            [list(c) for c in coeff_can],
            coarsening_threshold=0.0,
        )

    if need_downsplit:
        _ensure_reference_binary()
        nd, nb, recon = reconstruct_and_check(
            full_discretization, disc_pd, coeff_pd, reference_binary
        )
        results["downsplit"] = (nd * dimensionality, nb)
        reconstructions["downsplit"] = recon
        descriptors["downsplit"] = disc_pd.descriptor


    # ── Print ────────────────────────────────────────────────────────────────
    if results:
        print(f"\nthingy={thingy_id}, level={level}")
        print(
            f"  {'method':<18s} {'topo bits':>10s} {'value bits':>11s} {'total bits':>11s}"
        )
        print(f"  {'-' * 52}")
        for label, (nt, nb) in results.items():
            print(f"  {label:<18s} {nt:10d} {nb:11d} {nt + nb:11d}")

    labels = list(reconstructions.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert np.array_equal(
                reconstructions[labels[i]], reconstructions[labels[j]]
            ), (
                f"thingy {thingy_id} level={level}: "
                f"{labels[i]} and {labels[j]} reconstructions differ"
            )

    # ── Write descriptor files ───────────────────────────────────────────────
    for label, desc in descriptors.items():
        desc.to_file(str(output_dir / f"{prefix}_{label}"))
    if descriptors:
        print(f"  wrote descriptors to {output_dir}/")


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
