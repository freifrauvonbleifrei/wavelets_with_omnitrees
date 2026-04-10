#!/usr/bin/env python
"""Compare compress_by_omnitree_coarsening vs compress_by_pushdown_coarsening."""

import math
import copy
import numpy as np

import dyada.descriptor
import dyada.discretization
import dyada.linearization

try:
    from wavelets_with_omnitrees.dyada_cache_patch import install as _install_cache
except ModuleNotFoundError:
    from dyada_cache_patch import install as _install_cache  # type: ignore
_install_cache()
from wavelets_with_omnitrees import (
    transform_to_all_wavelet_coefficients,
    get_leaf_scalings,
    get_resampled_image,
    compress_by_omnitree_coarsening,
    compress_by_pushdown_coarsening,
    morton_to_multidim,
)

IMAGE = "standard_test_images/lena_gray_256.tif"
MAX_LEVEL = 4
THRESHOLDS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]


def build_initial_state(image_path: str):
    from PIL import Image

    data = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    input_shape = data.shape
    dimensionality = len(input_shape)
    base_resolution_level = [
        min(MAX_LEVEL, max(0, math.floor(math.log2(extent)))) for extent in input_shape
    ]
    uniform_descriptor = dyada.descriptor.RefinementDescriptor(
        dimensionality, base_resolution_level
    )
    discretization = dyada.discretization.Discretization(
        linearization=dyada.linearization.MortonOrderLinearization(),
        descriptor=uniform_descriptor,
    )
    ordered_input: np.ndarray = np.zeros(len(discretization), dtype=np.float32)
    for morton_index in range(len(ordered_input)):
        multidim_index = morton_to_multidim(
            morton_index, level=base_resolution_level[0]
        )
        ordered_input[morton_index] = data[multidim_index[1], multidim_index[0]]

    coefficients = transform_to_all_wavelet_coefficients(discretization, ordered_input)
    base_scaling = coefficients[0][0]
    for c in coefficients:
        c[0] = np.nan
    coefficients[0][0] = base_scaling

    target_level = discretization.descriptor.get_maximum_level().astype(np.int64)
    raster_orig = get_resampled_image(discretization, ordered_input, target_level)

    return discretization, coefficients, raster_orig, target_level


def evaluate(discretization, coefficients, raster_orig, target_level, label):
    scaling = get_leaf_scalings(discretization, coefficients)
    raster = get_resampled_image(discretization, scaling, target_level)
    max_err = float(np.max(np.abs(raster_orig - raster)))
    n_desc = len(discretization.descriptor)
    n_boxes = len(discretization)
    print(f"  {label:16s}  desc={n_desc:6d}  boxes={n_boxes:6d}  max_err={max_err:.2e}")
    return n_desc, n_boxes, max_err


print(f"Image: {IMAGE}\n")
print(f"{'threshold':>12}  {'method':16s}  {'desc':>6}  {'boxes':>6}  max_err")
print("-" * 68)

discretization0, coefficients0, raster_orig, target_level = build_initial_state(IMAGE)
n_desc0 = len(discretization0.descriptor)
n_boxes0 = len(discretization0)
print(f"  {'original':16s}  desc={n_desc0:6d}  boxes={n_boxes0:6d}")
print()

for threshold in THRESHOLDS:
    print(f"threshold={threshold:.0e}")

    disc_a, coeff_a, _ = compress_by_omnitree_coarsening(
        discretization0,
        copy.deepcopy(coefficients0),
        coarsening_threshold=threshold,
    )
    evaluate(disc_a, coeff_a, raster_orig, target_level, "canonical")

    disc_b, coeff_b = compress_by_pushdown_coarsening(
        discretization0,
        copy.deepcopy(coefficients0),
        coarsening_threshold=threshold,
    )
    evaluate(disc_b, coeff_b, raster_orig, target_level, "push(mnl=F)")
    print()
