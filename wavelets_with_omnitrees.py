#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import bitarray.util
import cProfile
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
import dyada.descriptor
import dyada.discretization
import dyada.linearization
import dyada.refinement

try:
    from libtiff import TIFF
except ImportError:
    TIFF = None


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
        if TIFF is None:
            raise ImportError("libtiff is required for reading .tif/.tiff files")
        tiff = TIFF.open(filename, mode="r")
        img_array = tiff.read_image()
        if len(img_array.shape) == 3:
            # Convert to grayscale using luminosity method
            img_array = (
                0.2989 * img_array[:, :, 0]
                + 0.5870 * img_array[:, :, 1]
                + 0.1140 * img_array[:, :, 2]
            )
        ic(img_array.dtype, img_array.shape, np.min(img_array), np.max(img_array))
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


@lru_cache(maxsize=None)
def _one_d_coeff_index(d_local: int, k: int) -> int:
    """Coefficient index for the pure 1D detail of local dim d_local in a k-dim node.

    Returns the integer index j whose k-bit representation has only bit d_local set,
    i.e. the Haar coefficient that encodes variation in exactly one dimension.
    For k=3: d_local 0→4, 1→2, 2→1  (the "1, 2, 4" the user refers to).
    """
    bits = ba.bitarray(k)
    bits.setall(0)
    bits[d_local] = 1
    return bitarray.util.ba2int(bits)


def pushdown_single_node_coefficients(
    old_coeffs: Sequence[float],
    old_ref: ba.bitarray,
    pushed_dim: int,
) -> tuple[list[float], list[list[float]]]:
    """Compute coefficient arrays for the new parent and intermediates after a pushdown.

    Splits the k-dim Haar coefficient vector using the tensor structure:
      C[k_rem, k_d] = old_coeffs[j]   where j encodes (k_rem, k_d) via bit d_local.

    Returns:
        new_parent_coeffs  : (k-1)-dim coefficient list (the k_d=0 slice of C).
        intermediate_coeffs: list of 2^(k-1) pairs [scaling, detail_d], one per
                             new intermediate child in remaining-dim Morton order.
    """
    k = old_ref.count()
    assert k >= 2
    refined_dims = get_refined_dimensions_desc(old_ref)
    d_local = get_local_bit_index(pushed_dim, refined_dims)
    n_rem = k - 1
    n_rem_slots = 1 << n_rem

    # Build the (2^(k-1) × 2) tensor C[k_rem, k_d].
    C = np.zeros((n_rem_slots, 2), dtype=np.float64)
    for j in range(1 << k):
        j_bits = bitarray.util.int2ba(j, length=k)
        k_d = int(j_bits[d_local])
        rem_bits = j_bits.copy()
        del rem_bits[d_local]
        k_rem = bitarray.util.ba2int(rem_bits) if n_rem > 0 else 0
        C[k_rem, k_d] = old_coeffs[j]

    new_parent_coeffs = list(C[:, 0])
    # Apply the inverse remaining-dim Haar to both columns to get per-intermediate values.
    scalings = dehierarchize(C[:, 0], n_rem)
    details = dehierarchize(C[:, 1], n_rem)
    intermediate_coeffs = [
        [float(scalings[r]), float(details[r])] for r in range(n_rem_slots)
    ]
    return new_parent_coeffs, intermediate_coeffs


def _merged_node_coefficients(
    childA_coeffs: Sequence[float],
    childB_coeffs: Sequence[float],
    intermediate_detail: float,
    pushed_dim: int,
    merged_ref: ba.bitarray,
) -> list[float]:
    """Compute wavelet coefficients for a merged non-leaf intermediate.

    When two non-leaf siblings (childA at pushed_dim=0, childB at pushed_dim=1)
    share the same refinement and are absorbed into a single merged node, the
    merged node's coefficients follow from the tensor product structure
    (cf. haar_on_pushdown.tex eq. 3):

        w[j]_merged = 0.5 * (childA[j_child] + (-1)^{j_push} * childB[j_child])

    For j_child=0 (children's scaling, which is NaN): the j_push=0 entry is NaN
    (merged scaling, convention), and the j_push=1 entry equals the intermediate's
    pushed-dim detail (computed by the pushdown formula, always real).
    """
    k_merged = merged_ref.count()
    merged_dims = get_refined_dimensions_desc(merged_ref)
    d_local_merged = get_local_bit_index(pushed_dim, merged_dims)

    merged_coeffs = [np.nan] * (1 << k_merged)
    for j in range(1 << k_merged):
        j_bits = bitarray.util.int2ba(j, length=k_merged)
        j_push = int(j_bits[d_local_merged])
        j_child_bits = j_bits.copy()
        del j_child_bits[d_local_merged]
        j_child = bitarray.util.ba2int(j_child_bits) if len(j_child_bits) > 0 else 0

        if j_child == 0 and j_push == 0:
            merged_coeffs[j] = np.nan  # scaling — convention
        elif j_child == 0 and j_push == 1:
            merged_coeffs[j] = intermediate_detail  # from pushdown formula
        else:
            sign = 1 if j_push == 0 else -1
            merged_coeffs[j] = 0.5 * (
                float(childA_coeffs[j_child]) + sign * float(childB_coeffs[j_child])
            )
    return merged_coeffs


def compress_by_pushdown_coarsening(
    discretization: dyada.discretization.Discretization,
    coefficients: list[Sequence[float]],
    coarsening_threshold: float = 0.0,
    merge_non_leaf: bool = True,
) -> tuple[dyada.discretization.Discretization, list[Sequence[float]]]:
    """Compress via progressive bottom-up pushdown guided by 1D detail coefficients.

    Each round:
      1. Pushdown: for every multi-dim node, push down the dimension whose pure 1D
         detail coefficient (indices 1, 2, 4, … in the Haar vector) is smallest.
      2. Coarsen: for every single-dim intermediate whose two children are all leaves,
         coarsen it if its lone detail coefficient is within the local error bound.
    Stop when no coarsenings occur in a round.

    merge_non_leaf: passed to apply_planned_pushdowns.  When True (default), non-leaf
        sibling pairs that share the same ref are merged into a single multi-dim
        intermediate.  When False, a 1-D intermediate node is inserted instead.

    Merged non-leaf intermediates are computed directly in the wavelet domain.
    """
    from dyada.pushdown import apply_planned_pushdowns

    num_dimensions = discretization.descriptor.get_num_dimensions()

    while True:
        # ── Pushdown round ──────────────────────────────────────────────────
        # Push the deepest multi-dim nodes first (by tree depth).  This avoids
        # ancestor–descendant pairs in the same batch while being less
        # restrictive than the old frontier check (which required all children
        # to have count() <= 1).
        candidates: list[tuple[int, ba.bitarray, int]] = (
            []
        )  # (desc_index, branch, depth)
        for desc_index, (branch, current_ref) in enumerate(
            dyada.descriptor.branch_generator(discretization.descriptor)
        ):
            if current_ref.count() < 2:
                continue
            depth = int(np.sum(dyada.descriptor.get_level_from_branch(branch)))
            candidates.append((desc_index, branch, depth))
        max_depth = max((d for _, _, d in candidates), default=-1)

        planned_pushdowns: list[tuple[int, ba.bitarray]] = []
        for desc_index, branch, depth in candidates:
            if depth < max_depth:
                continue
            current_ref = ba.bitarray(discretization.descriptor[desc_index])
            # Push the dim with the smallest pure 1D detail coefficient.
            k = current_ref.count()
            refined_dims = get_refined_dimensions_desc(current_ref)
            best_dim, best_score = refined_dims[0], float("inf")
            for global_dim in refined_dims:
                d_local = get_local_bit_index(global_dim, refined_dims)
                score = abs(coefficients[desc_index][_one_d_coeff_index(d_local, k)])
                if score < best_score:
                    best_score, best_dim = score, global_dim
            dims_ba = ba.bitarray(discretization.descriptor.d_zeros)
            dims_ba[best_dim] = 1
            planned_pushdowns.append((desc_index, dims_ba))

        if planned_pushdowns:
            old_disc = discretization
            discretization, pd_mapping = apply_planned_pushdowns(
                discretization,
                planned_pushdowns,
                track_mapping="patches",
                merge_non_leaf=merge_non_leaf,
            )

            # Update coefficients: copy unchanged nodes, compute new parent + intermediates.
            pushdown_dims: dict[int, int] = {
                old_i: next(d for d in range(num_dimensions) if dims_ba[d])
                for old_i, dims_ba in planned_pushdowns
            }
            # Build old child index pairs for each pushed parent: {old_i: [(childA, childB), ...]}
            old_child_pairs: dict[int, list[tuple[int, int]]] = {}
            linearization = old_disc._linearization
            for old_i, pushed_dim in pushdown_dims.items():
                old_ref = ba.bitarray(old_disc.descriptor[old_i])
                remaining_ref = old_ref.copy()
                remaining_ref[pushed_dim] = 0
                n_rem = remaining_ref.count()
                n_rem_slots = 1 << n_rem if n_rem > 0 else 1
                old_children = old_disc.descriptor.get_children(old_i)
                pairs: list[tuple[int, int]] = [(-1, -1)] * n_rem_slots
                for child_pos, child_idx in enumerate(old_children):
                    child_bits = linearization.get_binary_position_from_index(
                        [child_pos], [old_ref]
                    )
                    pushed_bit = int(child_bits[pushed_dim])
                    rem_bits = child_bits.copy()
                    rem_bits[pushed_dim] = 0
                    rem_pos = (
                        linearization.get_index_from_binary_position(
                            rem_bits, [], [remaining_ref]
                        )
                        if n_rem > 0
                        else 0
                    )
                    a, b = pairs[rem_pos]
                    if pushed_bit == 0:
                        pairs[rem_pos] = (child_idx, b)
                    else:
                        pairs[rem_pos] = (a, child_idx)
                old_child_pairs[old_i] = pairs

            new_coefficients: list = [
                [np.nan] for _ in range(len(discretization.descriptor))
            ]
            # Track new indices that receive computed merged-node coefficients,
            # so we don't overwrite them when copying non-pushed old nodes.
            merged_new_indices: set[int] = set()
            # First pass: handle pushed parents (compute new parent + child coefficients).
            for old_i, new_i_set in enumerate(pd_mapping):
                if old_i not in pushdown_dims:
                    continue

                pushed_dim = pushdown_dims[old_i]
                old_ref = ba.bitarray(old_disc.descriptor[old_i])
                remaining_ref = old_ref.copy()
                remaining_ref[pushed_dim] = 0

                new_parent_coeffs, interm_coeffs = pushdown_single_node_coefficients(
                    coefficients[old_i], old_ref, pushed_dim
                )

                # The new parent is the unique node in new_i_set with ref == remaining_ref.
                new_parent_i = next(
                    ni
                    for ni in new_i_set
                    if ba.bitarray(discretization.descriptor[ni]) == remaining_ref
                )
                new_coefficients[new_parent_i] = new_parent_coeffs
                merged_new_indices.add(new_parent_i)
                # Intermediates are the children of the new parent, in Morton order.
                for r, (child_start, _) in enumerate(
                    discretization.descriptor.get_child_ranges(new_parent_i)
                ):
                    child_ref = ba.bitarray(discretization.descriptor[child_start])
                    k_child = child_ref.count()
                    if k_child <= 1:
                        # Regular (non-merged) intermediate or leaf: use pushdown result.
                        new_coefficients[child_start] = interm_coeffs[r]
                        merged_new_indices.add(child_start)
                    else:
                        # Merged non-leaf: compute directly in wavelet space.
                        # The two absorbed children are childA (pushed_dim=0) and
                        # childB (pushed_dim=1) at remaining-dim position r.
                        childA_idx, childB_idx = old_child_pairs[old_i][r]
                        merged_ref = child_ref
                        child_orig_ref = merged_ref.copy()
                        child_orig_ref[pushed_dim] = 0
                        new_coefficients[child_start] = _merged_node_coefficients(
                            coefficients[childA_idx],
                            coefficients[childB_idx],
                            interm_coeffs[r][1],  # pushed-dim detail
                            pushed_dim,
                            merged_ref,
                        )
                        merged_new_indices.add(child_start)

            # Second pass: copy coefficients for non-pushed old nodes,
            # skipping new indices already assigned by merged-node computation.
            for old_i, new_i_set in enumerate(pd_mapping):
                if old_i in pushdown_dims:
                    continue
                new_i = next(iter(new_i_set))
                if new_i not in merged_new_indices:
                    new_coefficients[new_i] = coefficients[old_i]

            coefficients = new_coefficients

        # ── Coarsening round ────────────────────────────────────────────────
        # Handle k≥1 nodes: with merge_non_leaf=True, pushdown can create k≥2
        # merged intermediates whose all-leaf children should be coarsenable.
        p = dyada.refinement.PlannedAdaptiveRefinement(discretization)
        planned_any = False
        for desc_index, (branch, current_ref) in enumerate(
            dyada.descriptor.branch_generator(discretization.descriptor)
        ):
            k = current_ref.count()
            if k < 1:
                continue
            # Only coarsen when ALL children are leaves.
            children_indices = discretization.descriptor.get_children(
                desc_index, branch_to_parent=branch
            )
            if any(len(coefficients[c]) > 1 for c in children_indices):
                continue
            branch_level = dyada.descriptor.get_level_from_branch(branch).astype(
                np.int64
            )
            local_bound = coarsening_threshold * float(
                np.prod(np.power(2.0, -branch_level))
            )
            # For k=1: one detail; for k≥2: all 2^k-1 details must be within bound.
            if all(
                abs(coefficients[desc_index][j]) <= local_bound
                for j in range(1, 1 << k)
            ):
                p.plan_coarsening(desc_index, ba.bitarray(current_ref))
                planned_any = True

        if not planned_any:
            break

        old_disc = discretization
        discretization, coarsen_mapping = p.apply_refinements(
            track_mapping="patches", sweep_mode="canonical"
        )
        final_markers = p._markers
        new_coefficients = [[np.nan] for _ in range(len(discretization.descriptor))]
        inverted_mapping: dict[int, set[int]] = {}
        for old_index, mapped_to in enumerate(coarsen_mapping):
            for new_index in mapped_to:
                inverted_mapping.setdefault(new_index, set()).add(old_index)
        for new_index, mapped_from in inverted_mapping.items():
            expected_num_ref = discretization.descriptor[new_index].count()
            expected_len = 1 if expected_num_ref == 0 else (1 << expected_num_ref)
            mapped_from_sorted = sorted(mapped_from)
            matching = [
                oi for oi in mapped_from_sorted if len(coefficients[oi]) == expected_len
            ]
            first_found = matching[0] if matching else mapped_from_sorted[0]
            if first_found not in final_markers:
                new_coefficients[new_index] = coefficients[first_found]
                continue
            marker = final_markers[first_found]
            if not np.any(marker < 0):
                new_coefficients[new_index] = coefficients[first_found]
                continue
            # Coarsened node: drop the detail coefficients for coarsened dims
            # directly from the parent's existing wavelet coefficients.
            # No intermediate point values needed (haar_on_pushdown.tex eq. 3).
            num_ref = old_disc.descriptor[first_found].count()
            refined_dimensions_desc = get_refined_dimensions_desc(
                old_disc.descriptor[first_found]
            )
            delete_indices: set[int] = set()
            for d_i in range(num_dimensions):
                if marker[d_i] >= 0:
                    continue
                local_bit_index = get_local_bit_index(d_i, refined_dimensions_desc)
                delete_indices.update(
                    get_numbers_with_ith_bit_set(local_bit_index, num_ref)
                )
            new_coefficients[new_index] = list(
                np.delete(arr=coefficients[first_found], obj=sorted(delete_indices))
            )
        coefficients = new_coefficients

    # ── Normalize: pull all splits up as far as possible ─────────────────────
    # After the pushdown-coarsening loop the tree may contain 1-D intermediate
    # nodes whose siblings all share the same refinement dimension.  Normalizing
    # merges those back up so every split lives as high as possible in the tree
    # (the canonical omnitree form).  The wavelet coefficients are recomputed
    # from scratch by filling leaf scalings and re-hierarchising on the
    # normalized descriptor.

    # Extract leaf scalings from wavelet coefficients via top-down dehierarchization.
    # Leaf scalings (index 0) may be NaN from the wavelet convention, so we
    # compute them by propagating the root scaling through the tree.
    old_leaf_scalings = np.full(len(discretization), np.nan, dtype=np.float32)
    for desc_i, (branch, ref) in enumerate(
        dyada.descriptor.branch_generator(discretization.descriptor)
    ):
        if ref.count() > 0:
            coeff_arr = list(coefficients[desc_i])
            # The scaling (index 0) may be NaN; dehierarchize needs all values.
            # If scaling is NaN, the result will be NaN. This happens when the
            # parent hasn't propagated a scaling yet—skip and let the parent do it.
            if not np.isnan(coeff_arr[0]):
                scaling_vals = dehierarchize(coeff_arr, ref.count())
                children = discretization.descriptor.get_children(
                    desc_i, branch_to_parent=branch
                )
                for child_pos, child_idx in enumerate(children):
                    coefficients[child_idx][0] = scaling_vals[child_pos]
        if discretization.descriptor.is_box(desc_i):
            box_i = discretization.descriptor.to_box_index(desc_i)
            old_leaf_scalings[box_i] = coefficients[desc_i][0]

    # Normalize the descriptor (pull shared splits upward).
    discretization, norm_box_mapping, _ = dyada.refinement.normalize_discretization(
        discretization, track_mapping="boxes"
    )

    # Map old leaf scalings to new box order.
    if norm_box_mapping:
        new_leaf_scalings = np.full(len(discretization), np.nan, dtype=np.float32)
        for old_box_i, new_box_set in enumerate(norm_box_mapping):
            for new_box_j in new_box_set:
                new_leaf_scalings[new_box_j] = old_leaf_scalings[old_box_i]
    else:
        # No normalization needed — leaf scalings are unchanged.
        new_leaf_scalings = old_leaf_scalings

    # Recompute all wavelet coefficients on the normalized descriptor.
    coefficients = transform_to_all_wavelet_coefficients(
        discretization, new_leaf_scalings
    )

    # nan-scaling (scaling stored only at root).
    base_scaling = coefficients[0][0]
    for c in coefficients:
        c[0] = np.nan
    coefficients[0][0] = base_scaling

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
    # plot_2d_image(raster_before)

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
