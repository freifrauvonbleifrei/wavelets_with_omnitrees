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
from PIL import Image
from typing import Callable, Iterable, Optional, Sequence

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
        assert np.allclose(
            nodal_coefficients,
            np.matmul(nodalization_matrix(num_refined_dimensions), result),
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
    # iterate forward, using a stack instead of branch_generator + get_children
    descriptor = discretization.descriptor
    # Stack entries: [scaling_coefficients, child_counter, num_children]
    parent_stack: list = []

    for desc_i in range(len(descriptor)):
        # If this node has a parent on the stack, assign its scaling coefficient
        if parent_stack:
            p = parent_stack[-1]
            assert coefficients[desc_i] is not None
            assert coefficients[desc_i][0] is None or np.isnan(coefficients[desc_i][0])
            coefficients[desc_i][0] = p[0][p[1]]
            p[1] += 1

        ref = descriptor[desc_i]
        num_ref = ref.count()
        if num_ref > 0:
            coeff_array = coefficients[desc_i]
            assert coeff_array is not None
            assert coeff_array[0] is not None
            scaling = dehierarchize(coeff_array, num_ref)
            parent_stack.append([scaling, 0, 1 << num_ref])
        else:
            # Leaf: pop completed parents
            while parent_stack and parent_stack[-1][1] == parent_stack[-1][2]:
                parent_stack.pop()


def get_leaf_scalings(
    discretization: dyada.discretization.Discretization,
    coefficients: list[Sequence[float]],
) -> npt.NDArray[np.float64]:
    """Extract per-leaf scaling values from hierarchical coefficients.

    Makes a deep copy of the coefficients, fills in scaling values via
    dehierarchization, and returns an array of leaf scaling values.
    """
    import copy

    coeff = copy.deepcopy(coefficients)
    fill_scaling_from_hierarchical_coefficients(discretization, coeff)
    return np.array(
        [coeff[i][0] for i in range(len(coeff)) if discretization.descriptor.is_box(i)],
        dtype=np.float64,
    )


def _precompute_location_codes(
    descriptor: dyada.descriptor.RefinementDescriptor,
) -> list[list]:
    """Precompute per-node location codes in a single forward DFS pass.

    Returns loc_codes[i] = list of num_dimensions bitarrays, encoding the
    left/right path from root to node i in each dimension.
    Equivalent to branch_to_location_code(get_branch(i)[0], linearization)
    for Morton order, but O(n) total.
    """
    nd = descriptor.get_num_dimensions()
    n = len(descriptor)
    loc_codes: list[list] = [None] * n  # type: ignore
    # Stack: (parent_loc, parent_refined_dims, remaining_children, next_child_idx)
    stack: list = []
    current_loc = [ba.bitarray() for _ in range(nd)]

    for i in range(n):
        loc_codes[i] = [ba.bitarray(d) for d in current_loc]

        ref = descriptor[i]
        if ref.count() > 0:
            refined_dims = [d for d in range(nd) if ref[d]]
            stack.append([
                [ba.bitarray(d) for d in current_loc],
                refined_dims,
                1 << ref.count(),
                0,
            ])
            # First child: all position bits are 0
            for d in refined_dims:
                current_loc[d].append(0)
        else:
            while stack:
                parent_loc, refined_dims, remaining, child_idx = stack[-1]
                remaining -= 1
                child_idx += 1
                if remaining > 0:
                    stack[-1] = [parent_loc, refined_dims, remaining, child_idx]
                    current_loc = [ba.bitarray(d) for d in parent_loc]
                    for local_bit, d in enumerate(refined_dims):
                        current_loc[d].append((child_idx >> local_bit) & 1)
                    break
                stack.pop()

    return loc_codes


def _compute_node_levels(
    descriptor: dyada.descriptor.RefinementDescriptor,
) -> list[npt.NDArray[np.int8]]:
    """Pre-compute per-dimension level for every descriptor node in a single forward pass.

    Returns a list where levels[i] is the per-dimension level array for node i.
    Avoids branch_generator and get_level_from_branch entirely.
    """
    num_dims = descriptor.get_num_dimensions()
    n = len(descriptor)
    levels: list[npt.NDArray[np.int8]] = [None] * n  # type: ignore
    current_level = np.zeros(num_dims, dtype=np.int8)
    # Stack entries: [remaining_children, saved_parent_level]
    stack: list = []

    for i in range(n):
        ref = descriptor[i]
        levels[i] = current_level

        if ref.count() > 0:
            # Parent: save current level, compute child level
            child_level = current_level + np.array(
                [int(ref[d]) for d in range(num_dims)], dtype=np.int8
            )
            stack.append([1 << ref.count(), current_level])
            current_level = child_level
        else:
            # Leaf: advance — pop completed parents
            while stack:
                stack[-1][0] -= 1
                if stack[-1][0] > 0:
                    break
                current_level = stack[-1][1]
                stack.pop()

    return levels


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
        descriptor = current_discretization.descriptor
        node_levels = _compute_node_levels(descriptor)

        # Backwards iteration: children are visited before parents,
        # so the stack naturally provides children indices at each parent.
        stack: list[int] = []  # stack of descriptor indices
        desc_i = len(descriptor) - 1
        for current_refinement in reversed(descriptor):
            num_refinements = current_refinement.count()
            if num_refinements == 0:
                stack.append(desc_i)
                desc_i -= 1
                continue

            num_children = 1 << num_refinements
            children_indices = [stack.pop() for _ in range(num_children)]

            # Coarsening only when all direct children are leaves
            if not any(len(current_coefficients[c]) > 1 for c in children_indices):
                parent_coefficients = current_coefficients[desc_i]
                branch_level = node_levels[desc_i].astype(np.int64)
                box_volume = float(np.prod(np.power(2.0, -branch_level)))
                local_bound = phase_threshold * box_volume

                if abs(parent_coefficients[-1]) <= local_bound:
                    if sum(abs(v) for v in parent_coefficients[1:]) <= local_bound:
                        coarsen_mask = ba.bitarray(descriptor[desc_i])
                        p.plan_coarsening(desc_i, coarsen_mask)
                        planned_any = True
                    else:
                        refined_dimensions_desc = get_refined_dimensions_desc(
                            current_refinement
                        )
                        for global_dimension in refined_dimensions_desc:
                            local_bit_index = get_local_bit_index(
                                global_dimension, refined_dimensions_desc
                            )
                            hierarchical_indices = get_numbers_with_ith_bit_set(
                                local_bit_index, num_refinements
                            )
                            hierarchical_abs_values = [
                                abs(parent_coefficients[i_h])
                                for i_h in hierarchical_indices
                            ]
                            if sum(hierarchical_abs_values) <= local_bound:
                                one_d_refinement = ba.bitarray("0" * num_dimensions)
                                one_d_refinement[global_dimension] = 1
                                p.plan_coarsening(desc_i, one_d_refinement)
                                planned_any = True

            stack.append(desc_i)
            desc_i -= 1

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
            new_coefficients[new_index] = list(
                current_coefficients[first_found_old_index]
            )
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


def downsplit_single_node_coefficients(
    old_coeffs: Sequence[float],
    old_ref: ba.bitarray,
    down_dim: int,
) -> tuple[list[float], list[list[float]]]:
    """Compute coefficient arrays for the new parent and intermediates after a downsplit.

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
    d_local = get_local_bit_index(down_dim, refined_dims)
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


def _subset_index_in_ref(global_dims_set: ba.bitarray, ref: ba.bitarray) -> int:
    """Compute the coefficient array index for a subset of global dims within ref.

    Given a subset of global dimensions (as a bitarray) and a refinement bitarray,
    return the integer index into the coefficient array (of length 2^|ref|) that
    corresponds to that subset.
    """
    ref_dims = get_refined_dimensions_desc(ref)  # local ordering of dims in ref
    k = ref.count()
    bits = ba.bitarray(k)
    bits.setall(0)
    for local_i, global_dim in enumerate(ref_dims):
        if global_dims_set[global_dim]:
            bits[local_i] = 1
    return bitarray.util.ba2int(bits)


def reverse_downsplit_coefficients(
    parent_coeffs: Sequence[float],
    parent_ref: ba.bitarray,
    children_coeffs: list[Sequence[float]],
    children_refs: ba.bitarray | list[ba.bitarray],
    absorbed_dims: ba.bitarray,
) -> tuple[list[float], list[list[list[float]]]]:
    """Wavelet-space normalization: absorb dims from children into parent.

    Given parent with ref R_p and 2^|R_p| children, where absorbed_dims D is
    the set of dims shared by ALL children but not in R_p, produce:
    - Merged parent coefficients for ref R_p ∪ D (length 2^|R_p ∪ D|).
    - For each original child j: a list of 2^|D| sub-child coefficient arrays,
      each of length 2^|R_c_j \\ D|, indexed by D-position in Morton order.

    children_refs can be a single bitarray (all children have the same ref) or
    a list of bitarrays (one per child, allowing heterogeneous refs).  All
    children must have the absorbed_dims bits set.

    No scaling coefficients are ever used; this is purely wavelet-space arithmetic.
    """
    k_parent = parent_ref.count()
    n_parent_children = 1 << k_parent if k_parent > 0 else 1
    merged_ref = parent_ref | absorbed_dims
    k_merged = merged_ref.count()
    k_absorbed = absorbed_dims.count()
    n_absorbed_slots = 1 << k_absorbed
    ndim = len(parent_ref)

    # Normalize children_refs to a list
    if isinstance(children_refs, ba.bitarray):
        per_child_ref = [children_refs] * n_parent_children
    else:
        per_child_ref = children_refs

    assert len(children_coeffs) == n_parent_children
    absorbed_dims_list = get_refined_dimensions_desc(absorbed_dims)

    # ── Merged parent coefficients ──────────────────────────────────────────
    merged_coeffs = [np.nan] * (1 << k_merged)
    parent_dims_list = get_refined_dimensions_desc(parent_ref)

    for t2_int in range(n_absorbed_slots):
        # Build T₂ as a global-dim bitarray
        t2_bits_local = bitarray.util.int2ba(t2_int, length=k_absorbed)
        t2_global = ba.bitarray(ndim)
        t2_global.setall(0)
        for local_i, global_dim in enumerate(absorbed_dims_list):
            if t2_bits_local[local_i]:
                t2_global[global_dim] = 1

        if t2_int == 0:
            # T₂ = ∅: fiber is the parent's existing coefficients
            fiber = list(parent_coeffs)
        else:
            # Collect w_{T₂} from each child, using that child's own ref
            raw_fiber = []
            for j in range(n_parent_children):
                child_t2_index = _subset_index_in_ref(t2_global, per_child_ref[j])
                raw_fiber.append(float(children_coeffs[j][child_t2_index]))
            # Forward Haar (hierarchize)
            fiber = list(
                np.matmul(hierarchization_matrix(k_parent), raw_fiber, dtype=np.float64)
            )

        # Place fiber entries into merged_coeffs
        for t1_int in range(1 << k_parent if k_parent > 0 else 1):
            t1_bits_local = (
                bitarray.util.int2ba(t1_int, length=k_parent)
                if k_parent > 0
                else ba.bitarray()
            )
            t1_global = ba.bitarray(ndim)
            t1_global.setall(0)
            for local_i, global_dim in enumerate(parent_dims_list):
                if t1_bits_local[local_i]:
                    t1_global[global_dim] = 1

            # T = T₁ ∪ T₂ in merged ref
            t_global = t1_global | t2_global
            merged_index = _subset_index_in_ref(t_global, merged_ref)
            merged_coeffs[merged_index] = fiber[t1_int]

    # ── Split child coefficient arrays ──────────────────────────────────────
    # For each child mⱼ, split into 2^|D| sub-children by applying |D|-dim
    # inverse Haar per remaining-dim fiber (per child's own ref).
    nod_matrix = nodalization_matrix(k_absorbed)  # 2^|D| × 2^|D|

    split_children: list[list[list[float]]] = []
    for j in range(n_parent_children):
        child_c = children_coeffs[j]
        child_ref_j = per_child_ref[j]
        remaining_child_ref = child_ref_j.copy()
        for d in range(ndim):
            if absorbed_dims[d]:
                remaining_child_ref[d] = 0
        k_remaining = remaining_child_ref.count()
        n_remaining_slots = 1 << k_remaining if k_remaining > 0 else 1
        remaining_dims_list = get_refined_dimensions_desc(remaining_child_ref)

        # Build a (2^|D| × n_remaining_slots) tensor
        tensor = np.zeros((n_absorbed_slots, n_remaining_slots), dtype=np.float64)
        for td_int in range(n_absorbed_slots):
            td_bits_local = bitarray.util.int2ba(td_int, length=k_absorbed)
            td_global = ba.bitarray(ndim)
            td_global.setall(0)
            for local_i, global_dim in enumerate(absorbed_dims_list):
                if td_bits_local[local_i]:
                    td_global[global_dim] = 1

            for tr_int in range(n_remaining_slots):
                tr_bits_local = (
                    bitarray.util.int2ba(tr_int, length=k_remaining)
                    if k_remaining > 0
                    else ba.bitarray()
                )
                tr_global = ba.bitarray(ndim)
                tr_global.setall(0)
                for local_i, global_dim in enumerate(remaining_dims_list):
                    if tr_bits_local[local_i]:
                        tr_global[global_dim] = 1

                t_global = td_global | tr_global
                child_index = _subset_index_in_ref(t_global, child_ref_j)
                tensor[td_int, tr_int] = float(child_c[child_index])

        # Apply inverse Haar in D-dims
        sub_values = np.matmul(nod_matrix, tensor, dtype=np.float64)

        sub_children_for_j: list[list[float]] = []
        for d_int in range(n_absorbed_slots):
            sub_coeffs = [np.nan] * n_remaining_slots
            for tr_int in range(n_remaining_slots):
                sub_coeffs[tr_int] = float(sub_values[d_int, tr_int])
            # Restore NaN at scaling slot
            sub_coeffs[0] = np.nan
            sub_children_for_j.append(sub_coeffs)

        split_children.append(sub_children_for_j)

    return merged_coeffs, split_children


def _merged_node_coefficients(
    childA_coeffs: Sequence[float],
    childB_coeffs: Sequence[float],
    intermediate_detail: float,
    down_dim: int,
    merged_ref: ba.bitarray,
) -> list[float]:
    """Compute wavelet coefficients for a merged non-leaf intermediate.

    When two non-leaf siblings (childA at down_dim=0, childB at down_dim=1)
    share the same refinement and are absorbed into a single merged node, the
    merged node's coefficients follow from the tensor product structure
    (cf. haar_on_downsplit.tex eq. 3):

        w[j]_merged = 0.5 * (childA[j_child] + (-1)^{j_push} * childB[j_child])

    For j_child=0 (children's scaling, which is NaN): the j_push=0 entry is NaN
    (merged scaling, convention), and the j_push=1 entry equals the intermediate's
    split-dim detail (computed by the downsplit formula, always real).
    """
    k_merged = merged_ref.count()
    merged_dims = get_refined_dimensions_desc(merged_ref)
    d_local_merged = get_local_bit_index(down_dim, merged_dims)

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
            merged_coeffs[j] = intermediate_detail  # from downsplit formula
        else:
            sign = 1 if j_push == 0 else -1
            merged_coeffs[j] = 0.5 * (
                float(childA_coeffs[j_child]) + sign * float(childB_coeffs[j_child])
            )
    return merged_coeffs


def compress_by_downsplit_coarsening(
    discretization: dyada.discretization.Discretization,
    coefficients: list[Sequence[float]],
    coarsening_threshold: float = 0.0,
    *,
    _depth_schedule: Iterable[int] | None = None,
) -> tuple[dyada.discretization.Discretization, list[Sequence[float]]]:
    """Compress via progressive bottom-up downsplit guided by 1D detail coefficients.

    Each round:
      1. Downsplit: for every multi-dim node, split down the dimension whose pure 1D
         detail coefficient (indices 1, 2, 4, … in the Haar vector) is smallest.
      2. Coarsen: for every single-dim intermediate whose two children are all leaves,
         coarsen it if its lone detail coefficient is within the local error bound.
    Stop when no coarsenings occur in a round.

    Merged non-leaf intermediates are computed directly in the wavelet domain.

    If _depth_schedule is provided, downsplit is applied at each specified depth
    in order, instead of always targeting the current maximum depth.
    """
    from dyada.downsplit import apply_planned_downsplits

    num_dimensions = discretization.descriptor.get_num_dimensions()
    _depth_iter = iter(_depth_schedule) if _depth_schedule is not None else None

    while True:
        # ── Downsplit round ─────────────────────────────────────────────────
        # Split the deepest multi-dim nodes first (by tree depth).  This avoids
        # ancestor–descendant pairs in the same batch while being less
        # restrictive than the old frontier check (which required all children
        # to have count() <= 1).
        descriptor = discretization.descriptor
        node_levels = _compute_node_levels(descriptor)
        candidates: list[tuple[int, int]] = []  # (desc_index, depth)
        for desc_index in range(len(descriptor)):
            if descriptor[desc_index].count() < 2:
                continue
            depth = int(np.sum(node_levels[desc_index]))
            candidates.append((desc_index, depth))

        if _depth_iter is not None:
            try:
                target_depth = next(_depth_iter)
            except StopIteration:
                break
        else:
            target_depth = max((d for _, d in candidates), default=-1)

        planned_downsplits: list[tuple[int, ba.bitarray]] = []
        for desc_index, depth in candidates:
            if depth != target_depth:
                continue
            current_ref = ba.bitarray(discretization.descriptor[desc_index])
            # Split the dim with the smallest pure 1D detail coefficient.
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
            planned_downsplits.append((desc_index, dims_ba))

        if planned_downsplits:
            old_disc = discretization
            discretization, pd_mapping = apply_planned_downsplits(
                discretization,
                planned_downsplits,
                track_mapping="patches",
            )

            # Update coefficients: copy unchanged nodes, compute new parent + intermediates.
            downdown_dims: dict[int, int] = {
                old_i: next(d for d in range(num_dimensions) if dims_ba[d])
                for old_i, dims_ba in planned_downsplits
            }
            # Build old child index pairs for each split parent: {old_i: [(childA, childB), ...]}
            # Precompute subtree_end for O(1) child index lookups (avoids O(n) get_children)
            old_desc = old_disc.descriptor
            _n_old = len(old_desc)
            _subtree_end = np.empty(_n_old, dtype=np.int64)
            _ii = _n_old - 1
            while _ii >= 0:
                _ref = old_desc[_ii]
                if _ref.count() == 0:
                    _subtree_end[_ii] = _ii + 1
                else:
                    _cs = _ii + 1
                    for _ in range(1 << _ref.count()):
                        _cs = int(_subtree_end[_cs])
                    _subtree_end[_ii] = _cs
                _ii -= 1

            old_child_pairs: dict[int, list[tuple[int, int]]] = {}
            linearization = old_disc._linearization
            for old_i, down_dim in downdown_dims.items():
                old_ref = ba.bitarray(old_desc[old_i])
                remaining_ref = old_ref.copy()
                remaining_ref[down_dim] = 0
                n_rem = remaining_ref.count()
                n_rem_slots = 1 << n_rem if n_rem > 0 else 1
                # Compute children via subtree_end instead of get_children
                num_ch = 1 << old_ref.count()
                old_children = []
                _ci = old_i + 1
                for _ in range(num_ch):
                    old_children.append(_ci)
                    _ci = int(_subtree_end[_ci])
                pairs: list[tuple[int, int]] = [(-1, -1)] * n_rem_slots
                for child_pos, child_idx in enumerate(old_children):
                    child_bits = linearization.get_binary_position_from_index(
                        [child_pos], [old_ref]
                    )
                    down_bit = int(child_bits[down_dim])
                    rem_bits = child_bits.copy()
                    rem_bits[down_dim] = 0
                    rem_pos = (
                        linearization.get_index_from_binary_position(
                            rem_bits, [], [remaining_ref]
                        )
                        if n_rem > 0
                        else 0
                    )
                    a, b = pairs[rem_pos]
                    if down_bit == 0:
                        pairs[rem_pos] = (child_idx, b)
                    else:
                        pairs[rem_pos] = (a, child_idx)
                old_child_pairs[old_i] = pairs

            new_coefficients: list = [
                [np.nan] for _ in range(len(discretization.descriptor))
            ]
            # Track new indices that receive computed merged-node coefficients,
            # so we don't overwrite them when copying non-split old nodes.
            merged_new_indices: set[int] = set()
            # First pass: handle split parents (compute new parent + child coefficients).
            for old_i, new_i_set in enumerate(pd_mapping):
                if old_i not in downdown_dims:
                    continue

                down_dim = downdown_dims[old_i]
                old_ref = ba.bitarray(old_disc.descriptor[old_i])
                remaining_ref = old_ref.copy()
                remaining_ref[down_dim] = 0

                new_parent_coeffs, interm_coeffs = downsplit_single_node_coefficients(
                    coefficients[old_i], old_ref, down_dim
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
                        # Regular (non-merged) intermediate or leaf: use downsplit result.
                        # Restore NaN convention for the scaling slot (index 0).
                        ic = list(interm_coeffs[r])
                        ic[0] = np.nan
                        new_coefficients[child_start] = ic
                        merged_new_indices.add(child_start)
                    else:
                        # Merged non-leaf: compute directly in wavelet space.
                        # The two absorbed children are childA (down_dim=0) and
                        # childB (down_dim=1) at remaining-dim position r.
                        childA_idx, childB_idx = old_child_pairs[old_i][r]
                        merged_ref = child_ref
                        child_orig_ref = merged_ref.copy()
                        child_orig_ref[down_dim] = 0
                        new_coefficients[child_start] = _merged_node_coefficients(
                            coefficients[childA_idx],
                            coefficients[childB_idx],
                            interm_coeffs[r][1],  # split-dim detail
                            down_dim,
                            merged_ref,
                        )
                        merged_new_indices.add(child_start)

            # Second pass: copy coefficients for non-split old nodes,
            # skipping new indices already assigned by merged-node computation.
            for old_i, new_i_set in enumerate(pd_mapping):
                if old_i in downdown_dims:
                    continue
                new_i = next(iter(new_i_set))
                if new_i not in merged_new_indices:
                    new_coefficients[new_i] = coefficients[old_i]

            coefficients = new_coefficients

        # ── Coarsening round ────────────────────────────────────────────────
        p = dyada.refinement.PlannedAdaptiveRefinement(discretization)
        planned_any = False
        # Recompute levels after downsplit may have changed the descriptor.
        node_levels = _compute_node_levels(discretization.descriptor)

        # Backwards iteration: children are visited before parents,
        # so the stack naturally provides children indices at each parent.
        coarsen_stack: list[int] = []
        coarsen_desc_i = len(discretization.descriptor) - 1
        for current_ref in reversed(discretization.descriptor):
            k = current_ref.count()
            if k == 0:
                coarsen_stack.append(coarsen_desc_i)
                coarsen_desc_i -= 1
                continue

            num_children = 1 << k
            children_indices = [coarsen_stack.pop() for _ in range(num_children)]

            # Coarsen only when all direct children are leaves and the
            # parent has a complete coefficient vector (normalization may
            # leave newly created multi-dim nodes with placeholder [nan]).
            if len(coefficients[coarsen_desc_i]) == (1 << k) and not any(
                len(coefficients[c]) > 1 for c in children_indices
            ):
                branch_level = node_levels[coarsen_desc_i].astype(np.int64)
                local_bound = coarsening_threshold * float(
                    np.prod(np.power(2.0, -branch_level))
                )
                if all(
                    abs(coefficients[coarsen_desc_i][j]) <= local_bound
                    for j in range(1, 1 << k)
                ):
                    p.plan_coarsening(coarsen_desc_i, ba.bitarray(current_ref))
                    planned_any = True

            coarsen_stack.append(coarsen_desc_i)
            coarsen_desc_i -= 1

        if planned_any:
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
                    oi
                    for oi in mapped_from_sorted
                    if len(coefficients[oi]) == expected_len
                ]
                first_found = matching[0] if matching else mapped_from_sorted[0]
                if first_found not in final_markers:
                    new_coefficients[new_index] = list(coefficients[first_found])
                    continue
                marker = final_markers[first_found]
                if not np.any(marker < 0):
                    new_coefficients[new_index] = list(coefficients[first_found])
                    continue
                # Coarsened node: drop the detail coefficients for coarsened dims
                # directly from the parent's existing wavelet coefficients.
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

        # ── Normalization round ──────────────────────────────────────────
        # After downsplit + coarsening the tree may contain uniqueness
        # violations (1-D intermediates whose children all share a dimension
        # the parent doesn't have).  Fix them by applying the inverse of the
        # downsplit formula purely in wavelet space, then restructure with
        # marker-based normalization.
        #
        # Nested violations (where inner violation parents are children of
        # outer violation parents) are handled by processing one level at a
        # time: resolve inner-most violations first, place their
        # coefficients, then re-detect and resolve the next level.  This
        # ensures the refs produced by structural normalization always match
        # the coefficient transform expectations.
        while True:
            violations = dyada.descriptor.find_uniqueness_violations(
                discretization.descriptor
            )
            if not violations:
                break

            # Build violation parent→children map.
            violation_map: dict[int, list[int]] = {}
            for violation in violations:
                sorted_v = sorted(violation)
                violation_map[sorted_v[0]] = sorted_v[1:]

            # Find leaf-level violations (inner-most: none of their children
            # are themselves violation parents).
            leaf_violations = {
                phi: ch
                for phi, ch in violation_map.items()
                if all(ci not in violation_map for ci in ch)
            }

            # Compute coefficient transforms for leaf-level violations.
            level_transforms: dict = {}
            for parent_hi, children_hi in leaf_violations.items():
                parent_ref = ba.bitarray(discretization.descriptor[parent_hi])
                child_refs = [
                    ba.bitarray(discretization.descriptor[ci]) for ci in children_hi
                ]
                common_child_ref = child_refs[0].copy()
                for cr in child_refs[1:]:
                    common_child_ref &= cr
                absorbed_dims = common_child_ref & ~parent_ref
                merged_ref = parent_ref | absorbed_dims

                merged_coeffs, split_children = reverse_downsplit_coefficients(
                    coefficients[parent_hi],
                    parent_ref,
                    [coefficients[ci] for ci in children_hi],
                    child_refs,
                    absorbed_dims,
                )
                child_map = {}
                for idx, ci in enumerate(children_hi):
                    remaining = child_refs[idx].copy()
                    for d in range(num_dimensions):
                        if absorbed_dims[d]:
                            remaining[d] = 0
                    child_map[ci] = (split_children[idx], absorbed_dims, remaining)

                level_transforms[parent_hi] = (
                    merged_coeffs,
                    merged_ref,
                    child_map,
                )

            # Apply structural normalization for this level's violations
            # only, using markers directly.
            old_disc = discretization
            p_norm = dyada.refinement.PlannedAdaptiveRefinement(discretization)
            for parent_hi, children_hi in leaf_violations.items():
                dimensions_to_shift = ~ba.bitarray(discretization.descriptor[parent_hi])
                for ci in children_hi:
                    dimensions_to_shift &= discretization.descriptor[ci]
                dim_shift_array = dyada.descriptor.int8_ndarray_from_iterable(
                    dimensions_to_shift,
                )
                p_norm._markers[parent_hi] += dim_shift_array
                for ci in children_hi:
                    p_norm._markers[ci] -= dim_shift_array

            discretization, norm_mapping = p_norm.create_new_discretization(
                track_mapping="patches"
            )
            linearization = discretization._linearization

            # Precompute location codes for both old and new descriptors (O(n) each)
            old_loc_codes = _precompute_location_codes(old_disc.descriptor)
            new_loc_codes = _precompute_location_codes(discretization.descriptor)

            # Build inverse mapping: new_index → set of old_indices.
            inv_map: dict[int, set[int]] = {}
            for old_i, new_set in enumerate(norm_mapping):
                for ni in new_set:
                    inv_map.setdefault(ni, set()).add(old_i)

            # Collect old indices involved in this level's violations.
            violation_involved: set[int] = set()
            for phi, children_hi in leaf_violations.items():
                violation_involved.add(phi)
                violation_involved.update(children_hi)

            # Place coefficients at new positions.
            new_coefficients: list = [
                [np.nan] for _ in range(len(discretization.descriptor))
            ]
            assigned: set[int] = set()

            # Pass 1: merged parents.
            for parent_hi, (
                merged_coeffs,
                merged_ref,
                child_map,
            ) in level_transforms.items():
                for ni in norm_mapping[parent_hi]:
                    new_ref = ba.bitarray(discretization.descriptor[ni])
                    if new_ref == merged_ref:
                        new_coefficients[ni] = list(merged_coeffs)
                        assigned.add(ni)

            # Pass 2: split children → sub-child coefficient arrays.
            for _, (_, _, child_map) in level_transforms.items():
                for child_hi, (
                    sub_list,
                    abs_dims,
                    remaining_child_ref,
                ) in child_map.items():
                    candidate_new = [
                        ni for ni in norm_mapping[child_hi] if ni not in assigned
                    ]
                    if not candidate_new:
                        continue

                    # If children dissolve (remaining ref = 0), sub-children
                    # are leaves with [nan] — the default is correct.
                    if remaining_child_ref.count() == 0:
                        continue

                    # Match sub-children to new positions by spatial location
                    # in the absorbed dims.
                    k_abs = abs_dims.count()
                    abs_dims_list = get_refined_dimensions_desc(abs_dims)

                    old_loc = old_loc_codes[child_hi]

                    for ni in candidate_new:
                        new_ref = ba.bitarray(discretization.descriptor[ni])
                        if new_ref != remaining_child_ref:
                            continue
                        new_loc = new_loc_codes[ni]
                        d_position = ba.bitarray(k_abs)
                        d_position.setall(0)
                        for local_i, gdim in enumerate(abs_dims_list):
                            new_bits = new_loc[gdim]
                            old_bits = old_loc[gdim]
                            if len(new_bits) > len(old_bits):
                                d_position[local_i] = new_bits[-1]
                        d_int = bitarray.util.ba2int(d_position)
                        new_coefficients[ni] = list(sub_list[d_int])
                        assigned.add(ni)

            # Pass 3: uninvolved nodes (1:1 copy from old).
            for ni in range(len(discretization.descriptor)):
                if ni in assigned:
                    continue
                old_sources = inv_map.get(ni, set())
                if not old_sources:
                    continue
                uninvolved_sources = sorted(
                    oi for oi in old_sources if oi not in violation_involved
                )
                new_ref = ba.bitarray(discretization.descriptor[ni])
                new_len = 1 if new_ref.count() == 0 else (1 << new_ref.count())
                for oi in uninvolved_sources:
                    if len(coefficients[oi]) == new_len:
                        new_coefficients[ni] = list(coefficients[oi])
                        break

            coefficients = new_coefficients

        if _depth_iter is None and not planned_any:
            break

    return discretization, coefficients


def compress_by_level_sweep_coarsening(
    discretization: dyada.discretization.Discretization,
    coefficients: list[Sequence[float]],
    coarsening_threshold: float = 0.0,
) -> tuple[dyada.discretization.Discretization, list[Sequence[float]]]:
    """Continue compression from a downsplit result by sweeping lower depths.

    Expects the input to already be the result of compress_by_downsplit_coarsening
    (i.e., canonical coarsening + default downsplit to convergence).
    Sweeps remaining multi-dimensional depths (max down to 0), then re-converges
    with default downsplit.  Repeats until no further improvement.

    This guarantees at least as many boxes are removed as default downsplit alone,
    since the input already is the downsplit result and this only adds work.
    """
    # Sweep remaining multi-dim depths and re-converge
    while True:
        descriptor = discretization.descriptor
        node_levels = _compute_node_levels(descriptor)
        multi_dim_depths: list[int] = sorted(
            {
                int(np.sum(node_levels[i]))
                for i in range(len(descriptor))
                if descriptor[i].count() >= 2
            },
            reverse=True,
        )
        if not multi_dim_depths:
            break

        old_count = len(discretization)
        # Downsplit at each remaining multi-dim depth (max down to lowest)
        discretization, coefficients = compress_by_downsplit_coarsening(
            discretization,
            [list(c) for c in coefficients],
            coarsening_threshold,
            _depth_schedule=multi_dim_depths,
        )
        # Re-converge with default downsplit to exploit any new opportunities
        discretization, coefficients = compress_by_downsplit_coarsening(
            discretization,
            [list(c) for c in coefficients],
            coarsening_threshold,
        )
        if len(discretization) >= old_count:
            break

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
    scaling_coefficients = get_leaf_scalings(discretization, coefficients)
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
