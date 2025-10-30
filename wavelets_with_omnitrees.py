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
    if dimensionality == 1:
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
) -> npt.NDArray[np.float128]:
    result = np.matmul(
        hierarchization_matrix(num_refined_dimensions),
        nodal_coefficients,
        dtype=np.float128,
    )
    # scope for asserting perfect reconstructability
    if True:
        assert all(
            nodal_coefficients
            == np.matmul(nodalization_matrix(num_refined_dimensions), result)
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


def read_img_file(filename: str) -> npt.NDArray[np.float16]:
    """
    Reads an image file and returns the pixel values as a numpy array.
    The image is converted to grayscale and normalized to [0, 1].
    """
    img = Image.open(filename).convert("L")  # Convert to grayscale
    img_array = np.array(img, dtype=np.float16) / 255.0  # Normalize to [0, 1]
    return img_array


def plot_2d_image(
    discretization: dyada.discretization.Discretization,
    coefficients: Sequence[np.float32],
):
    maximum_level = discretization.descriptor.get_maximum_level()
    resampled_array: npt.NDArray = np.ndarray(shape=2**maximum_level)
    very_small_number = 1e-12
    for x, y in product(
        range(0, 2 ** maximum_level[0]), range(0, 2 ** maximum_level[1])
    ):
        x_coord = x / 2.0 ** maximum_level[0] + very_small_number
        y_coord = y / 2.0 ** maximum_level[1] + very_small_number
        dyada_coordinate = dyada.coordinates.coordinate_from_sequence(
            [x_coord, y_coord]
        )
        box_index = discretization.get_containing_box(dyada_coordinate)
        resampled_array[x, y] = coefficients[box_index]

    plt.imshow(
        resampled_array,
        cmap="Greys",  # vmin=0., vmax=1.0
    )
    plt.show()


@lru_cache(maxsize=None)
def get_numbers_with_ith_bit_set(i_set_bit_index: int, bit_length: int) -> set[int]:
    numbers = set()
    for j in range(2 ** (bit_length)):
        j_as_bitarray = bitarray.util.int2ba(j, length=bit_length)
        if (
            get_one_d_refinement(i_set_bit_index, bit_length) & j_as_bitarray
        ).count() > 0:
            numbers.add(j)
    return numbers


def get_set_bitarray_indices(bits: ba.bitarray) -> set[int]:
    return {i for i, bit in enumerate(reversed(bits)) if bit}


@lru_cache(maxsize=None)
def get_one_d_refinement(i_set_bit_index: int, bit_length: int) -> ba.bitarray:
    one_d_refinement = ba.bitarray("0" * bit_length)
    one_d_refinement[bit_length - 1 - i_set_bit_index] = 1
    return one_d_refinement


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
    if args.file_to_compress.endswith(".png") or args.file_to_compress.endswith(".jpg"):
        data = read_img_file(args.file_to_compress)
    else:
        raise NotImplementedError(
            "File extension of " + args.file_to_compress + " not supported"
        )

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

    ordered_input_coefficients_2: list[float] = [None] * len(discretization)  # type: ignore
    for box_index, (level, multidim_index) in enumerate(
        discretization.get_all_boxes_level_indices()
    ):
        # TODO this loop is toooo sloooow
        ordered_input_coefficients_2[box_index] = data[tuple(multidim_index)]

    assert None not in ordered_input_coefficients_2
    initial_length = len(ordered_input_coefficients_2)

    # show the original image
    plot_2d_image(discretization, ordered_input_coefficients_2)
    ordered_input_coefficients = ordered_input_coefficients_2

    # # transform to Haar wavelet data
    ic("start transforming")
    coefficients = transform_to_all_wavelet_coefficients(
        discretization,
        ordered_input_coefficients,  # type: ignore
    )
    ic(len(coefficients))
    assert all(len(c) > 0 for c in coefficients)
    num_dimensions = discretization.descriptor.get_num_dimensions()

    # coarsen all zero-coefficient values -> lossless
    while True:
        p = dyada.refinement.PlannedAdaptiveRefinement(discretization)
        current_length = len(discretization.descriptor)
        ic(current_length, len(discretization))
        for descriptor_index in range(len(discretization.descriptor)):
            num_refinements = discretization.descriptor[descriptor_index].count()
            if num_refinements < 1:
                continue
            # plan coarsening only if: all children are leaf nodes and all hierarchical coefficients are zeros
            num_children = num_refinements**2
            children_indices = discretization.descriptor.get_children(descriptor_index)
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
                break
            else:
                # TODO needs dyada.linearization if not Z order
                refined_dimensions = get_set_bitarray_indices(
                    discretization.descriptor[descriptor_index]
                )
                for d_i in refined_dimensions:
                    # translate the hierarchical function index
                    # into which refinements are involved in the hierarchical coefficient
                    hierarchical_coefficent_indices = get_numbers_with_ith_bit_set(
                        d_i, num_dimensions
                    )
                    if all(
                        coefficients[descriptor_index][hierarchical_index] == 0.0
                        for hierarchical_index in hierarchical_coefficent_indices
                    ) and (len(p._planned_refinements) == 0):
                        one_d_refinement = get_one_d_refinement(d_i, num_dimensions)
                        p.plan_coarsening(
                            descriptor_index,
                            one_d_refinement,
                        )
                        break
        if len(p._planned_refinements) == 0:
            # nothing more to compress
            break
        planned_refinements = p._planned_refinements
        new_descriptor, mapping = p.apply_refinements(
            track_mapping="patches"
        )  # TODO allow combining independent refinements

        dyada.descriptor.validate_descriptor(new_descriptor)
        new_descriptor, normalization_mapping, num_rounds = (
            dyada.refinement.normalize_discretization(
                dyada.discretization.Discretization(
                    discretization._linearization, new_descriptor
                ),
                track_mapping="patches",
            )
        )
        dyada.descriptor.validate_descriptor(new_descriptor)
        new_discretization = dyada.discretization.Discretization(
            discretization._linearization, new_descriptor
        )
        assert num_rounds < 2
        try:
            mapping = dyada.refinement.merge_mappings(mapping, normalization_mapping)
        except Exception as e:
            ic(mapping, normalization_mapping)
            raise e
        new_coefficients = {}
        for key, value in sorted(mapping.items(), reverse=True):
            assert len(value) == 1
            new_coefficients[value[0]] = coefficients[key]
            # shorten the arrays if necessary
            if key in planned_refinements and min(planned_refinements[key]) < 0:
                # accumulate negative entries
                negative_refined_at = (
                    d_i
                    for d_i in range(num_dimensions)
                    if planned_refinements[key][d_i] < 0
                )
                hierarchical_coefficent_indices_to_delete: set[int] = set()
                for refined_dimension in negative_refined_at:
                    hierarchical_coefficent_indices_to_delete.update(
                        get_numbers_with_ith_bit_set(
                            num_dimensions - 1 - refined_dimension, num_dimensions
                        )
                    )
                new_coefficients[value[0]] = np.delete(
                    arr=new_coefficients[value[0]],
                    obj=list(hierarchical_coefficent_indices_to_delete),
                )
        dyada.descriptor.validate_descriptor(new_descriptor)  # TODO remove
        discretization = dyada.discretization.Discretization(
            discretization._linearization, new_descriptor
        )
        assert len(discretization.descriptor) == len(new_coefficients)
        coefficients = new_coefficients

    # validate by showing the image again
    scaling_coefficients = [
        coefficients[index][0]
        for index in range(len(coefficients))
        if discretization.descriptor.is_box(index)
    ]
    assert len(scaling_coefficients) == len(discretization)
    plot_2d_image(discretization, scaling_coefficients)
    ic(initial_length, len(scaling_coefficients))

    # coarsen leaves with low coefficients until compression ratio is reached #TODO
