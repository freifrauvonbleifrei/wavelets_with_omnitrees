#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import bitarray.util
from functools import lru_cache
import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.typing as npt
from icecream import ic
from itertools import product
from PIL import Image
from libtiff import TIFF
from typing import Callable, Optional, Sequence

import dyada
import dyada.descriptor
import dyada.discretization
import dyada.drawing
import dyada.linearization
import dyada.refinement


def hadamard_haar_1d() -> npt.NDArray[np.int8]:
    one_d_coefficients: npt.NDArray[np.int8] = np.ones([2, 2], dtype=np.int8)
    one_d_coefficients[1, 1] = -1
    return one_d_coefficients


# generate the higher-order Hadamard-Haar matrices for transform:
@lru_cache(maxsize=7)
def coefficient_matrix(
    dimensionality: int, one_d_transform: Callable[[], npt.NDArray[np.int8]]
) -> npt.NDArray[np.int8]:
    coefficients = one_d_transform()
    if dimensionality == 0:
        return np.array([[1]], dtype=np.int8)
    elif dimensionality == 1:
        return coefficients
    return np.kron(
        coefficients, coefficient_matrix(dimensionality - 1, one_d_transform)
    )


@lru_cache(maxsize=7)
def hierarchization_matrix(num_refined_dimensions: int) -> npt.NDArray[np.float16]:
    # assumes uniform compact wavelet basis in all dimensions
    return 0.5**num_refined_dimensions * coefficient_matrix(
        num_refined_dimensions, hadamard_haar_1d
    )


@lru_cache(maxsize=7)
def nodalization_matrix(num_refined_dimensions: int) -> npt.NDArray[np.int8]:
    return coefficient_matrix(num_refined_dimensions, hadamard_haar_1d)


def hierarchize(
    nodal_coefficients: Sequence, num_refined_dimensions: int
) -> npt.NDArray[np.float64]:
    result = np.matmul(
        hierarchization_matrix(num_refined_dimensions),
        nodal_coefficients,
        dtype=np.float64,
    )
    # scope for asserting perfect reconstructability
    if True:
        assert all(
            nodal_coefficients
            == np.matmul(nodalization_matrix(num_refined_dimensions), result)
        )
    return result


def dehierarchize(
    hierarchical_coefficients: Sequence, num_refined_dimensions: int
) -> npt.NDArray[np.float64]:
    result = np.matmul(
        nodalization_matrix(num_refined_dimensions),
        hierarchical_coefficients,
        dtype=np.float64,
    )
    return result


def transform_to_all_wavelet_coefficients(
    discretization: dyada.discretization.Discretization,
    nodal_coefficients: Sequence[np.float32],
):
    assert len(nodal_coefficients) == len(discretization)
    # the coefficients vector stores a list of coefficients for every omnitree node:
    # leaves have a single (nodal/scaling) coefficient, parents have first a nodal/scaling coefficient s, and then the hierarchical increments / d
    coefficients: list[Optional[Sequence]] = [
        None for _ in range(len(discretization.descriptor))
    ]  # of np.floats with varying precision

    # iterate the refinement descriptor backwards
    backwards_nodal_coefficients = reversed(nodal_coefficients)
    computed_scaling_coefficients = []  # stack of previously obtained results
    descriptor_index = len(discretization.descriptor)  # reversed enumerate index
    for refinement in reversed(discretization.descriptor):
        descriptor_index -= 1
        if refinement == discretization.descriptor.d_zeros:
            coefficient = next(backwards_nodal_coefficients)
            # ic(coefficient)
            coefficients[descriptor_index] = [coefficient]
            computed_scaling_coefficients.append(coefficient)
        else:
            num_refinements = refinement.count()
            num_children = 1 << num_refinements
            children_coefficients = []
            for _ in range(num_children):
                children_coefficients.append(computed_scaling_coefficients.pop())
            # assert isinstance(
            #     discretization._linearization,
            #     dyada.linearization.MortonOrderLinearization,
            # ), (
            #     "assuming Z order; if you need something else, here is the place to implement it."
            # )
            coefficients[descriptor_index] = hierarchize(
                children_coefficients, num_refinements
            )
            computed_scaling_coefficients.append(coefficients[descriptor_index][0])  # type: ignore

    return coefficients


def fill_scaling_from_hierarchical_coefficients(
    discretization: dyada.discretization.Discretization,
    coefficients: Sequence[Optional[Sequence[np.float32]]],
):
    # fills in the scaling coefficients from the hierarchical coefficients
    # iterate from the top down
    for i_c, (branch, refinement) in enumerate(
        dyada.descriptor.branch_generator(discretization.descriptor)
    ):
        coeff_array = coefficients[i_c]
        assert coeff_array is not None
        assert coeff_array[0] is not None
        num_refined_dimensions = refinement.count()
        scaling_coefficients = dehierarchize(coeff_array, num_refined_dimensions)
        # assign the scaling coefficients to the children
        children_indices = discretization.descriptor.get_children(
            i_c, branch_to_parent=branch
        )
        for i_child, child_index in enumerate(children_indices):
            assert coefficients[child_index] is not None
            assert coefficients[child_index][0] is None or np.isnan(
                coefficients[child_index][0]
            )
            coefficients[child_index][0] = scaling_coefficients[i_child]


def read_img_file(filename: str) -> npt.NDArray[np.float16]:
    """
    Reads an image file and returns the pixel values as a numpy array.
    The image is converted to grayscale and normalized to [0, 1].
    """
    if filename.lower().endswith(".tiff") or filename.lower().endswith(".tif"):
        tiff = TIFF.open(filename, mode="r")
        img_array = tiff.read_image()
        if len(img_array.shape) == 3:
            # Convert to grayscale using luminosity method
            img_array = (
                0.2989 * img_array[:, :, 0]
                + 0.5870 * img_array[:, :, 1]
                + 0.1140 * img_array[:, :, 2]
            )
        img_array = img_array.astype(np.float16) / 255.0  # Normalize to [0, 1]
    else:
        img = Image.open(filename, mode="r").convert("L")  # Convert to grayscale
        img_array = np.array(img, dtype=np.float16) / 255.0  # Normalize to [0, 1]
    return img_array


def get_resampled_image(
    discretization: dyada.discretization.Discretization,
    coefficients: Sequence[np.float32],
    target_maximum_level: Optional[npt.NDArray[np.int64]] = None,
) -> npt.NDArray[np.float32]:
    if target_maximum_level is None:
        maximum_level = discretization.descriptor.get_maximum_level().astype(np.int64)
    else:
        maximum_level = target_maximum_level
    resampled_array: npt.NDArray = np.zeros(
        shape=(2 ** maximum_level[0], 2 ** maximum_level[1])
    )
    # fast(er) iteration over all boxes
    box_index = -1
    for branch, refinement in dyada.descriptor.branch_generator(
        discretization.descriptor
    ):
        if refinement.count() == 0:  # leaf node
            box_index += 1
            location_level, location_index = discretization.get_level_index_from_branch(
                branch
            )
            box_size = 2 ** (maximum_level - location_level)
            min_corner = location_index * box_size
            max_corner = min_corner + box_size
            resampled_array[
                min_corner[0] : max_corner[0], min_corner[1] : max_corner[1]
            ] = coefficients[box_index]
    return resampled_array


def plot_2d_image(
    resampled_array: npt.NDArray[np.float32],
) -> None:
    plt.imshow(resampled_array, cmap="Greys")  # , vmin=0.0, vmax=1.0)
    plt.show()


@lru_cache(maxsize=None)
def get_numbers_with_ith_bit_set(
    i_set_bit_index: int, bit_length: int, reverse: bool = False
) -> tuple[int, ...]:
    numbers = []
    for j in range(2 ** (bit_length)):
        j_as_bitarray = bitarray.util.int2ba(j, length=bit_length)
        one_d_refinement = get_one_d_refinement(
            i_set_bit_index, bit_length, reverse=reverse
        )
        if (one_d_refinement & j_as_bitarray).count() > 0:
            numbers.append(j)
    return tuple(numbers)


def get_set_bitarray_indices(bits: ba.bitarray) -> set[int]:
    return {i for i, bit in enumerate(reversed(bits)) if bit}


@lru_cache(maxsize=None)
def get_one_d_refinement(
    i_set_bit_index: int, bit_length: int, reverse=False
) -> ba.bitarray:
    one_d_refinement = ba.bitarray("0" * bit_length)
    one_d_refinement[i_set_bit_index] = 1
    if reverse:
        one_d_refinement.reverse()
    return one_d_refinement


def morton_to_multidim(
    morton_index: int, num_dimensions: int = 2, level: int = 6
) -> tuple[int, ...]:
    index_as_bitarray = bitarray.util.int2ba(
        morton_index, length=level * num_dimensions
    )
    return tuple(
        bitarray.util.ba2int(index_as_bitarray[mask_start::num_dimensions])
        for mask_start in range(num_dimensions)
    )


def get_refined_dimensions_desc(
    refinement: ba.frozenbitarray | ba.bitarray,
) -> list[int]:
    return list(reversed(dyada.linearization.bitmask_to_indices(refinement)))


def get_local_bit_index(
    global_dimension: int, refined_dimensions_desc: Sequence[int]
) -> int:
    """Map global dimension id to local bit index in coefficient ordering."""
    return refined_dimensions_desc.index(global_dimension)


def compress_by_omnitree_coarsening(
    discretization: dyada.discretization.Discretization,
    coefficients: list[Sequence[float]],
    coarsening_threshold: float = 0.0,
) -> tuple[dyada.discretization.Discretization, list[Sequence[float]]]:
    """Iteratively coarsen with staged coefficient-local bounds."""
    num_dimensions = discretization.descriptor.get_num_dimensions()

    def apply_single_phase_round(
        current_discretization: dyada.discretization.Discretization,
        current_coefficients: list[Sequence[float]],
        phase_threshold: float,
    ) -> tuple[
        dyada.discretization.Discretization,
        list[Sequence[float]],
        bool,
    ]:
        p = dyada.refinement.PlannedAdaptiveRefinement(current_discretization)
        planned_any = False

        for descriptor_index, (branch, current_refinement) in enumerate(
            dyada.descriptor.branch_generator(current_discretization.descriptor)
        ):
            num_refinements = current_refinement.count()
            if num_refinements < 1:
                continue

            # Coarsening only when all direct children are leaves
            children_indices = current_discretization.descriptor.get_children(
                descriptor_index, branch_to_parent=branch
            )
            if any(len(current_coefficients[c]) > 1 for c in children_indices):
                continue

            parent_coefficients = current_coefficients[descriptor_index]
            branch_level = dyada.descriptor.get_level_from_branch(branch).astype(
                np.int64
            )
            box_volume = float(np.prod(np.power(2.0, -branch_level)))
            local_bound = phase_threshold * box_volume

            if abs(parent_coefficients[-1]) > local_bound:
                continue

            if sum(abs(v) for v in parent_coefficients[1:]) <= local_bound:
                coarsen_mask = ba.bitarray(
                    current_discretization.descriptor[descriptor_index]
                )
                p.plan_coarsening(descriptor_index, coarsen_mask)
                planned_any = True
                continue

            refined_dimensions_desc = get_refined_dimensions_desc(current_refinement)
            for global_dimension in refined_dimensions_desc:
                local_bit_index = get_local_bit_index(
                    global_dimension, refined_dimensions_desc
                )
                hierarchical_indices = get_numbers_with_ith_bit_set(
                    local_bit_index, num_refinements
                )
                hierarchical_abs_values = [
                    abs(parent_coefficients[i_h]) for i_h in hierarchical_indices
                ]
                can_partially_coarsen = sum(hierarchical_abs_values) <= local_bound
                if can_partially_coarsen:
                    one_d_refinement = ba.bitarray("0" * num_dimensions)
                    one_d_refinement[global_dimension] = 1
                    p.plan_coarsening(descriptor_index, one_d_refinement)
                    planned_any = True

        if not planned_any:
            return current_discretization, current_coefficients, False

        new_discretization, mapping = p.apply_refinements(
            track_mapping="patches", sweep_mode="canonical"
        )
        final_markers = p._markers
        if __debug__:
            dyada.descriptor.validate_descriptor(new_discretization.descriptor)

        new_coefficients: list[Sequence[float]] = [
            [np.nan] for _ in range(len(new_discretization.descriptor))
        ]
        inverted_mapping: dict[int, set[int]] = {}
        for old_index, mapped_to in enumerate(mapping):
            for new_index in mapped_to:
                inverted_mapping.setdefault(new_index, set()).add(old_index)

        for new_index, mapped_from in inverted_mapping.items():
            expected_num_refinements = new_discretization.descriptor[new_index].count()
            expected_len = (
                1 if expected_num_refinements == 0 else (1 << expected_num_refinements)
            )
            mapped_from_sorted = sorted(mapped_from)
            matching_old_indices = [
                old_i
                for old_i in mapped_from_sorted
                if len(current_coefficients[old_i]) == expected_len
            ]
            first_found_old_index = (
                matching_old_indices[0]
                if matching_old_indices
                else mapped_from_sorted[0]
            )
            new_coefficients[new_index] = current_coefficients[first_found_old_index]
            if first_found_old_index not in final_markers:
                continue
            marker = final_markers[first_found_old_index]
            if not np.any(marker < 0):
                continue

            refined_dimensions_desc = get_refined_dimensions_desc(
                current_discretization.descriptor[first_found_old_index]
            )
            num_refinements = current_discretization.descriptor[
                first_found_old_index
            ].count()
            delete_indices: set[int] = set()
            for d_i in range(num_dimensions):
                if marker[d_i] >= 0:
                    continue
                local_bit_index = get_local_bit_index(d_i, refined_dimensions_desc)
                delete_indices.update(
                    get_numbers_with_ith_bit_set(local_bit_index, num_refinements)
                )
            new_coefficients[new_index] = np.delete(
                arr=new_coefficients[new_index],
                obj=sorted(delete_indices),
            )
            if len(new_coefficients[new_index]) != expected_len:
                raise ValueError(
                    f"Coefficient length mismatch at new index {new_index}: "
                    f"got {len(new_coefficients[new_index])}, expected {expected_len}"
                )

        return new_discretization, new_coefficients, True

    def run_phase(
        phase_discretization: dyada.discretization.Discretization,
        phase_coefficients: list[Sequence[float]],
        phase_threshold: float,
    ) -> tuple[dyada.discretization.Discretization, list[Sequence[float]]]:
        while True:
            (
                phase_discretization,
                phase_coefficients,
                changed,
            ) = apply_single_phase_round(
                phase_discretization,
                phase_coefficients,
                phase_threshold=phase_threshold,
            )
            if not changed:
                return phase_discretization, phase_coefficients

    discretization, coefficients = run_phase(
        discretization, coefficients, phase_threshold=0.0
    )
    discretization, coefficients = run_phase(
        discretization,
        coefficients,
        phase_threshold=coarsening_threshold,
    )

    return discretization, coefficients


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "--compression_ratio",
        type=float,
        help="desired compression ratio, > 0.0, can be 1 for lossless",
        default=1.0,
    )
    parser.add_argument(
        "file_to_compress",
        type=str,
        help="filename ; if file suffix is .png, will use 2d compression",
        # default="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/gray/102061.jpg",
    )
    parser.add_argument(
        "--max_level",
        type=int,
        default=None,
        help="optional cap for base dyadic level per dimension to reduce runtime",
    )
    args = parser.parse_args()

    # read the data tensor
    data = read_img_file(args.file_to_compress)

    # construct matching discretization
    input_shape = data.shape
    dimensionality = len(input_shape)
    base_resolution_level = [
        max(0, math.floor(math.log2(extent))) for extent in input_shape
    ]
    if args.max_level is not None:
        base_resolution_level = [
            min(level, args.max_level) for level in base_resolution_level
        ]
    ic(base_resolution_level)
    uniform_descriptor = dyada.descriptor.RefinementDescriptor(
        dimensionality, base_resolution_level
    )
    discretization = dyada.discretization.Discretization(
        linearization=dyada.linearization.MortonOrderLinearization(),
        descriptor=uniform_descriptor,
    )
    descriptor_entries_before = len(discretization.descriptor)
    boxes_before = len(discretization)

    # reorder input data to Z order
    ordered_input_coefficients: npt.NDArray[np.float32] = np.zeros(
        shape=(len(discretization)), dtype=np.float32
    )
    for morton_index in range(len(ordered_input_coefficients)):
        assert base_resolution_level[0] == base_resolution_level[1]
        multidim_index = morton_to_multidim(
            morton_index, level=base_resolution_level[0]
        )
        ordered_input_coefficients[morton_index] = data[
            multidim_index[1], multidim_index[0]
        ]

    ordered_input_coefficients_2: npt.NDArray[np.float32] = np.empty_like(
        ordered_input_coefficients
    )
    for box_index, (level, multidim_index) in enumerate(
        discretization.get_all_boxes_level_indices()
    ):
        # this loop would be too slow
        ordered_input_coefficients_2[box_index] = data[tuple(multidim_index)]
        assert (
            ordered_input_coefficients[box_index]
            == ordered_input_coefficients_2[box_index]
        )
        if box_index > 32:
            break

    assert None not in ordered_input_coefficients
    initial_length = len(ordered_input_coefficients)

    # show the original image
    target_maximum_level = discretization.descriptor.get_maximum_level().astype(
        np.int64
    )
    raster_before = get_resampled_image(
        discretization,
        ordered_input_coefficients,
        target_maximum_level=target_maximum_level,
    )
    plot_2d_image(raster_before)

    # # transform to Haar wavelet data
    ic("start transforming")
    coefficients = transform_to_all_wavelet_coefficients(
        discretization,
        ordered_input_coefficients,  # type: ignore
    )

    # drop the scaling coefficients
    base_scaling_coefficient = coefficients[0][0]
    for c in coefficients:
        c[0] = np.nan
    coefficients[0][0] = base_scaling_coefficient
    ic(len(coefficients))
    assert all(len(c) > 0 for c in coefficients)
    coarsening_threshold = 1e-7
    discretization, coefficients = compress_by_omnitree_coarsening(
        discretization,
        coefficients,
        coarsening_threshold=coarsening_threshold,
    )
    descriptor_entries_after = len(discretization.descriptor)
    boxes_after = len(discretization)

    # validate by showing the image again
    fill_scaling_from_hierarchical_coefficients(discretization, coefficients)
    scaling_coefficients = [
        coefficients[index][0]
        for index in range(len(coefficients))
        if discretization.descriptor.is_box(index)
    ]
    assert len(scaling_coefficients) == len(discretization)
    raster_after = get_resampled_image(
        discretization,
        scaling_coefficients,
        target_maximum_level=target_maximum_level,
    )
    plot_2d_image(raster_after)
    ic(initial_length, len(scaling_coefficients))

    assert raster_before.shape == raster_after.shape
    difference = raster_before - raster_after
    abs_difference = np.abs(difference)
    ic(np.max(abs_difference), np.mean(difference))
    print(
        "compression summary:",
        f"descriptor_entries {descriptor_entries_before}->{descriptor_entries_after}",
        f"(x{descriptor_entries_before / descriptor_entries_after:.3f})",
        f"boxes {boxes_before}->{boxes_after}",
        f"(x{boxes_before / boxes_after:.3f})",
    )
    assert np.isfinite(np.max(abs_difference))
    ic(str(discretization))
    # Compression decisions are coefficient-local, not direct pixel-error bounded.
    # coarsen leaves with low coefficients until compression ratio is reached #TODO
