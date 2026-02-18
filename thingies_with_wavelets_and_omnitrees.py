#!/usr/bin/env python3
import argparse as arg
import numpy as np

from icecream import ic

import dyada
import dyada.descriptor
import dyada.discretization
import dyada.linearization

try:
    from wavelets_with_omnitrees.wavelets_with_omnitrees import (
        transform_to_all_wavelet_coefficients,
        fill_scaling_from_hierarchical_coefficients,
        compress_by_omnitree_coarsening,
    )
except ModuleNotFoundError:
    from wavelets_with_omnitrees import (
        transform_to_all_wavelet_coefficients,
        fill_scaling_from_hierarchical_coefficients,
        compress_by_omnitree_coarsening,
    )


def inside_sphere(points: np.ndarray) -> np.ndarray:
    center = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    radius = 0.35
    return np.sum((points - center) ** 2, axis=1) <= radius**2


def inside_tetrahedron(points: np.ndarray) -> np.ndarray:
    # Unit simplex in [0,1]^3
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return (x >= 0.0) & (y >= 0.0) & (z >= 0.0) & ((x + y + z) <= 1.0)


def inside_diagonal_rod(points: np.ndarray) -> np.ndarray:
    # Finite cylinder around segment p0->p1 in unit cube
    p0 = np.array([0.2, 0.2, 0.2], dtype=np.float64)
    p1 = np.array([0.8, 0.8, 0.8], dtype=np.float64)
    axis = p1 - p0
    axis_len2 = float(np.dot(axis, axis))
    radius = 0.08

    rel = points - p0
    t = np.clip(np.dot(rel, axis) / axis_len2, 0.0, 1.0)
    proj = p0 + np.outer(t, axis)
    dist2 = np.sum((points - proj) ** 2, axis=1)
    return dist2 <= radius**2


THINGIES: dict[str, callable] = {
    "sphere": inside_sphere,
    "tetrahedron": inside_tetrahedron,
    "diagonal_rod": inside_diagonal_rod,
}


def midpoint_occupancy_coefficients(
    inside_fn, discretization: dyada.discretization.Discretization
) -> np.ndarray:
    occupancy = np.zeros(len(discretization), dtype=np.float64)
    for box_index in range(len(discretization)):
        interval = dyada.discretization.coordinates_from_box_index(
            discretization, box_index
        )
        midpoint = 0.5 * (interval.lower_bound + interval.upper_bound)
        occupancy[box_index] = 1.0 if inside_fn(midpoint.reshape(1, 3))[0] else 0.0
    return occupancy


def binary_values_on_full_grid(
    full_discretization: dyada.discretization.Discretization,
    compressed_discretization: dyada.discretization.Discretization,
    compressed_scalings: np.ndarray,
) -> np.ndarray:
    full_grid_binary = np.zeros(len(full_discretization), dtype=bool)
    for box_index in range(len(full_discretization)):
        interval = dyada.discretization.coordinates_from_box_index(
            full_discretization, box_index
        )
        midpoint = 0.5 * (interval.lower_bound + interval.upper_bound)
        compressed_box_index = compressed_discretization.get_containing_box(midpoint)
        if isinstance(compressed_box_index, tuple):
            compressed_box_index = min(compressed_box_index)
        full_grid_binary[box_index] = compressed_scalings[compressed_box_index] >= 0.5
    return full_grid_binary


def run_for_thingy(thingy_name: str, inside_fn, max_level: int):
    full_discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(3, [max_level, max_level, max_level]),
    )
    fully_resolved_occupancy = midpoint_occupancy_coefficients(
        inside_fn, full_discretization
    )

    all_coefficients = transform_to_all_wavelet_coefficients(
        full_discretization, fully_resolved_occupancy
    )
    root_scaling = all_coefficients[0][0]
    for c in all_coefficients:
        c[0] = np.nan
    all_coefficients[0][0] = root_scaling

    discretization, coefficients = compress_by_omnitree_coarsening(
        full_discretization,
        [list(c) for c in all_coefficients],
        coarsening_threshold=0.0,
    )

    fill_scaling_from_hierarchical_coefficients(discretization, coefficients)
    scaling_coefficients = np.array(
        [
            coefficients[index][0]
            for index in range(len(coefficients))
            if discretization.descriptor.is_box(index)
        ]
    )
    assert not np.isnan(scaling_coefficients).any()

    reference_binary = fully_resolved_occupancy >= 0.5
    reconstructed_binary = binary_values_on_full_grid(
        full_discretization,
        discretization,
        scaling_coefficients,
    )
    exact_match = bool(np.array_equal(reconstructed_binary, reference_binary))
    assert exact_match

    print(f"\nthingy={thingy_name}")
    print(
        f"descriptor={len(discretization.descriptor)}",
        f"boxes={len(discretization)}",
        f"exact_match={exact_match}",
    )


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "--max_level",
        type=int,
        default=6,
        help="fully resolved isotropic level per dimension",
    )
    parser.add_argument(
        "--thingy",
        type=str,
        default="all",
        choices=["all", "sphere", "tetrahedron", "diagonal_rod"],
        help="which analytic thingy to run",
    )
    args = parser.parse_args()

    selected = THINGIES.items()
    if args.thingy != "all":
        selected = [(args.thingy, THINGIES[args.thingy])]

    for thingy_name, inside_fn in selected:
        run_for_thingy(
            thingy_name,
            inside_fn,
            max_level=args.max_level,
        )
