#!/usr/bin/env python3
import argparse as arg
import numpy as np
import openvdb as vdb

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
    )


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
                    [(i + 0.5) * voxel_size, (j + 0.5) * voxel_size, (k + 0.5) * voxel_size],
                    dtype=np.float64,
                )
                if inside_fn(midpoint.reshape(1, 3))[0]:
                    acc.setValueOn((i, j, k), True)
    grid.pruneInactive()
    return grid


def run_one(thingy_name: str, inside_fn, level: int):
    full_discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(3, [level, level, level]),
    )
    full_occupancy = midpoint_occupancy_coefficients(inside_fn, full_discretization)
    reference_binary = full_occupancy >= 0.5

    coefficients = transform_to_all_wavelet_coefficients(full_discretization, full_occupancy)
    root_scaling = coefficients[0][0]
    for coeff in coefficients:
        coeff[0] = np.nan
    coefficients[0][0] = root_scaling

    discretization, coefficients = compress_by_omnitree_coarsening(
        full_discretization,
        [list(c) for c in coefficients],
        coarsening_threshold=0.0,
    )
    fill_scaling_from_hierarchical_coefficients(discretization, coefficients)
    scaling_coefficients = np.array(
        [
            coefficients[i][0]
            for i in range(len(coefficients))
            if discretization.descriptor.is_box(i)
        ]
    )
    reconstructed_binary = binary_values_on_full_grid(
        full_discretization, discretization, scaling_coefficients
    )
    assert bool(np.array_equal(reconstructed_binary, reference_binary))

    openvdb_grid = midpoint_occupancy_openvdb(inside_fn, level=level)

    print(f"\nthingy={thingy_name}, level={level}")
    print(
        "  openvdb:",
        f"active_voxels={openvdb_grid.activeVoxelCount()}",
        f"leaf_count={openvdb_grid.leafCount()}",
        f"mem_usage={openvdb_grid.memUsage()}",
    )
    print(
        "  omnitree:",
        f"descriptor={len(discretization.descriptor)}",
        f"boxes={len(discretization)}",
        "exact_match=True",
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
