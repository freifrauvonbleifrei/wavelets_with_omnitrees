#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import bitarray.util
from collections import Counter
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
import dyada.coordinates
import dyada.discretization
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
) -> npt.NDArray:
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
) -> npt.NDArray[np.float32]:
    result = np.matmul(
        hierarchization_matrix(num_refined_dimensions),
        nodal_coefficients,
        dtype=np.float32,
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
) -> npt.NDArray[np.float32]:
    result = np.matmul(
        nodalization_matrix(num_refined_dimensions),
        hierarchical_coefficients,
        dtype=np.float32,
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
            num_children = 2**num_refinements
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
    for i_c, coeff_array in enumerate(coefficients):
        assert coeff_array is not None
        assert coeff_array[0] is not None
        refinement = discretization.descriptor[i_c]
        num_refined_dimensions = refinement.count()
        scaling_coefficients = dehierarchize(coeff_array, num_refined_dimensions)
        # assign the scaling coefficients to the children
        children_indices = discretization.descriptor.get_children(i_c)
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
) -> npt.NDArray[np.float32]:
    maximum_level = discretization.descriptor.get_maximum_level()
    maximum_level = maximum_level.astype(np.int64)
    resampled_array: npt.NDArray = np.ndarray(
        shape=(2 ** maximum_level[0], 2 ** maximum_level[1])
    )
    # fast(er) iteration over all boxes
    box_index = -1
    for branch, refinement in dyada.descriptor.branch_generator(
        discretization.descriptor
    ):
        if refinement.count() == 0:  # leaf node
            box_index += 1
            location_level, location_index = (
                dyada.discretization.get_level_index_from_branch(
                    linearization=discretization._linearization, branch=branch
                )
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
) -> set[int]:
    numbers = set()
    for j in range(2 ** (bit_length)):
        j_as_bitarray = bitarray.util.int2ba(j, length=bit_length)
        one_d_refinement = get_one_d_refinement(
            i_set_bit_index, bit_length, reverse=reverse
        )
        if (one_d_refinement & j_as_bitarray).count() > 0:
            numbers.add(j)
    return numbers


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
    args = parser.parse_args()

    # read the data tensor
    data = read_img_file(args.file_to_compress)

    # construct matching discretization
    input_shape = data.shape
    dimensionality = len(input_shape)
    base_resolution_level = [5, 5]
    #     math.floor(math.log2(extent)) for extent in input_shape
    # ]  # TODO ceil # [2,2]  #
    ic(base_resolution_level)
    uniform_descriptor = dyada.descriptor.RefinementDescriptor(
        dimensionality, base_resolution_level
    )
    discretization = dyada.discretization.Discretization(
        linearization=dyada.linearization.MortonOrderLinearization(),
        descriptor=uniform_descriptor,
    )

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
    raster_before = get_resampled_image(discretization, ordered_input_coefficients)
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
        c[0] = None
    coefficients[0][0] = base_scaling_coefficient
    ic(len(coefficients))
    assert all(len(c) > 0 for c in coefficients)
    num_dimensions = discretization.descriptor.get_num_dimensions()

    # coarsen all zero-coefficient values -> lossless
    coarsened_index = 0
    while True:
        p = dyada.refinement.PlannedAdaptiveRefinement(discretization)
        current_length = len(discretization.descriptor)
        ic(current_length, len(discretization))
        for descriptor_index in range(coarsened_index, len(discretization.descriptor)):
            current_refinement = discretization.descriptor[descriptor_index]
            num_refinements = current_refinement.count()
            if num_refinements < 1:
                continue
            # plan coarsening only if: all children are leaf nodes and all hierarchical coefficients are zeros
            num_children = num_refinements**2
            children_indices = discretization.descriptor.get_children(
                descriptor_index
            )  # TODO speed up w/ branch to parent
            if any(len(coefficients[c]) > 1 for c in children_indices):
                continue
            if coefficients[descriptor_index][-1] != 0:
                # if the last coefficient is not 0, there's no chance than any dimension can be losslessly refined
                continue
            if all(d == 0.0 for d in coefficients[descriptor_index][1:]):
                p.plan_coarsening(
                    descriptor_index,
                    discretization.descriptor[descriptor_index],
                )
            else:
                refined_dimensions = get_set_bitarray_indices(current_refinement)
                for d_i in refined_dimensions:
                    # translate the hierarchical function index
                    # into which refinements are involved in the hierarchical coefficient
                    hierarchical_coefficent_indices = get_numbers_with_ith_bit_set(
                        d_i, num_dimensions
                    )
                    if all(
                        coefficients[descriptor_index][hierarchical_index] == 0.0
                        for hierarchical_index in hierarchical_coefficent_indices
                    ):
                        one_d_refinement = get_one_d_refinement(
                            d_i, num_dimensions, reverse=True
                        )
                        p.plan_coarsening(
                            descriptor_index,
                            one_d_refinement,
                        )
        if len(p._planned_refinements) == 0:
            # nothing more to compress
            break
        planned_refinements = p._planned_refinements
        new_discretization, mapping = p.apply_refinements(
            track_mapping="patches"
        )  # TODO allow combining independent refinements
        new_coefficients = [np.nan for _ in range(len(new_discretization.descriptor))]
        planned_refinements_as_dict = {
            planned_refinement[0]: planned_refinement[1]
            for planned_refinement in planned_refinements
        }
        new_index_counter = Counter(m for s in mapping for m in s)
        for new_index, count in new_index_counter.items():
            first_found_old_index = next(
                old_index
                for old_index, mapped_to in enumerate(mapping)
                if new_index in mapped_to
            )
            new_coefficients[new_index] = coefficients[first_found_old_index]
            if count > 1:
                if first_found_old_index not in planned_refinements_as_dict:
                    continue
                # accumulate negative entries
                negative_refined_at = (
                    d_i
                    for d_i in range(num_dimensions)
                    if planned_refinements_as_dict[first_found_old_index][d_i] < 0
                )
                hierarchical_coefficent_indices_to_delete: set[int] = set()
                for refined_dimension in negative_refined_at:
                    hierarchical_coefficent_indices_to_delete.update(
                        get_numbers_with_ith_bit_set(
                            refined_dimension, num_dimensions, reverse=True
                        )
                    )
                new_coefficients[new_index] = np.delete(
                    arr=new_coefficients[new_index],
                    obj=list(hierarchical_coefficent_indices_to_delete),
                )
        assert len(new_discretization.descriptor) == len(new_coefficients)
        discretization = new_discretization
        coefficients = new_coefficients

    # validate by showing the image again
    fill_scaling_from_hierarchical_coefficients(discretization, coefficients)
    scaling_coefficients = [
        coefficients[index][0]
        for index in range(len(coefficients))
        if discretization.descriptor.is_box(index)
    ]
    assert len(scaling_coefficients) == len(discretization)
    raster_after = get_resampled_image(discretization, scaling_coefficients)
    plot_2d_image(raster_after)
    ic(initial_length, len(scaling_coefficients))

    assert raster_before.shape == raster_after.shape
    difference = np.abs(raster_before - raster_after)
    ic(np.max(difference), np.mean(difference))
    assert np.max(difference) < 0.00000001
    # coarsen leaves with low coefficients until compression ratio is reached #TODO
