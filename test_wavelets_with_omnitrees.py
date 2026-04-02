import os
import tempfile
from pathlib import Path

import numpy as np

import dyada
import dyada.descriptor
import dyada.discretization
import dyada.linearization

import bitarray as ba

from wavelets_with_omnitrees import (
    get_numbers_with_ith_bit_set,
    transform_to_all_wavelet_coefficients,
    get_leaf_scalings,
    compress_by_omnitree_coarsening,
    compress_by_downsplit_coarsening,
    compress_by_level_sweep_coarsening,
)
from compare_openvdb_vs_wavelet_omnitrees import (
    _sample_at_finest_midpoints,
    _hierarchize_on_tree,
    _load_discretization,
)


def test_get_numbers_with_ith_bit_set():
    assert list(get_numbers_with_ith_bit_set(0, 3)) == [4, 5, 6, 7]
    assert list(get_numbers_with_ith_bit_set(1, 3)) == [2, 3, 6, 7]
    assert list(get_numbers_with_ith_bit_set(2, 3)) == [1, 3, 5, 7]


def _build_full_grid(dim, level):
    return dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(dim, [level] * dim),
    )


def _sphere_fn(points):
    """Binary sphere occupancy: 1 if inside unit sphere centered at (0.5,…)."""
    center = 0.5
    radius = 0.4
    dist2 = np.sum((points - center) ** 2, axis=1)
    return (dist2 <= radius**2).astype(np.float64)


def _full_pipeline(dim, level, inside_fn):
    """Run the full pipeline: sample → hierarchize → canonical → downsplit → level_sweep."""
    disc = _build_full_grid(dim, level)
    # Sample at finest-level midpoints (the canonical way)
    grid_res = 1 << level
    n = len(disc)
    nodal = np.empty(n, dtype=np.float64)
    for box_index in range(n):
        interval = dyada.discretization.coordinates_from_box_index(disc, box_index)
        midpoint = (interval.lower_bound + interval.upper_bound) / 2.0
        nodal[box_index] = inside_fn(midpoint.reshape(1, -1))[0]

    coefficients = transform_to_all_wavelet_coefficients(disc, nodal)
    root_scaling = coefficients[0][0]
    for c in coefficients:
        c[0] = np.nan
    coefficients[0][0] = root_scaling

    disc_can, coeff_can = compress_by_omnitree_coarsening(
        disc,
        [list(c) for c in coefficients],
        coarsening_threshold=0.0,
    )
    disc_pd, coeff_pd = compress_by_downsplit_coarsening(
        disc_can,
        [list(c) for c in coeff_can],
        coarsening_threshold=0.0,
    )
    disc_ls, coeff_ls = compress_by_level_sweep_coarsening(
        disc_pd,
        [list(c) for c in coeff_pd],
        coarsening_threshold=0.0,
    )
    return disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls


def _build_disc_from_descriptor_string(dim, descriptor_string):
    """Build a Discretization from a space-separated binary descriptor string."""
    bits = descriptor_string.replace(" ", "")
    desc = dyada.descriptor.RefinementDescriptor.from_binary(
        num_dimensions=dim, binary=ba.bitarray(bits)
    )
    return dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(), desc
    )


def _plot_tikz(disc, labels, name):
    dyada.plot_all_boxes_2d(
        disc,
        labels=labels,
        backend="tikz",
        filename=name,
    )
    # reformulate labels to be empty where there is no leaf
    descriptor = disc.descriptor
    all_labels = [""] * len(descriptor)
    leaf_index = 0
    for index in range(len(descriptor)):
        if descriptor.is_box(index):
            all_labels[index] = labels[leaf_index]
            leaf_index += 1
    dyada.plot_tree_tikz(disc.descriptor, labels=all_labels, filename=name + "_tree")


def _compress_pipeline(disc, nodal, coarsening_threshold=0.0):
    """Hierarchize and run the full compression pipeline."""
    coefficients = transform_to_all_wavelet_coefficients(disc, nodal)
    root_scaling = coefficients[0][0]
    for c in coefficients:
        c[0] = np.nan
    coefficients[0][0] = root_scaling
    disc_can, coeff_can = compress_by_omnitree_coarsening(
        disc,
        [list(c) for c in coefficients],
        coarsening_threshold=coarsening_threshold,
    )
    disc_pd, coeff_pd = compress_by_downsplit_coarsening(
        disc_can,
        [list(c) for c in coeff_can],
        coarsening_threshold=coarsening_threshold,
    )
    disc_ls, coeff_ls = compress_by_level_sweep_coarsening(
        disc_pd,
        [list(c) for c in coeff_pd],
        coarsening_threshold=coarsening_threshold,
    )
    return disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls


def test_2d_cascading_downsplit():
    """Test that cascading downsplit compresses a tree that single-level downsplit cannot.

    The grid has matching sub-patterns across y at certain x positions,
    which cascading downsplit detects by splitting dim 1 first (merge-preferring
    heuristic), then further splitting the merged multi-dim child, enabling
    coarsening where the y-detail is zero.
    """
    disc = _build_disc_from_descriptor_string(2, "11 00 10 10 00 00 00 00 10 00 00")
    nodal = np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.float64)
    assert len(disc) == 7
    assert (
        dyada.discretization_to_2d_ascii(disc)
        == """\
_________________
|_______|___|___|
|_______|_|_|___|"""
    )

    labels = [str(int(v)) for v in nodal]
    _plot_tikz(disc, labels, "before")

    disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls = _compress_pipeline(
        disc, nodal
    )

    # Canonical coarsening: no change (no zero details at threshold=0)
    assert len(disc_can) == 7
    # Cascading downsplit: merges the top-right cell (both y-values are 1) → 7→6
    assert len(disc_pd) == 6
    assert len(disc_ls) == 6
    assert (
        dyada.discretization_to_2d_ascii(disc_pd)
        == """\
_________________
|_______|___|   |
|_______|_|_|___|"""
    )

    # Lossless: reconstructed scalings match original at the raster level
    scalings = get_leaf_scalings(disc_ls, coeff_ls)
    from wavelets_with_omnitrees import get_resampled_image

    target_level = disc.descriptor.get_maximum_level().astype(np.int64)
    raster_orig = get_resampled_image(disc, nodal, target_level)
    raster_new = get_resampled_image(disc_ls, scalings, target_level)
    np.testing.assert_array_equal(raster_orig, raster_new)


def test_2d_downsplit_compresses():
    """Test compression on a 2D descriptor where downsplit enables coarsening"""
    disc = _build_disc_from_descriptor_string(2, "11 10 00 00 00 00 00")
    nodal = np.array([0, 1, 1, 0, 1], dtype=np.float64)
    assert len(disc) == 5
    assert (
        dyada.discretization_to_2d_ascii(disc)
        == """\
_________
|___|___|
|_|_|___|"""
    )

    labels_before = [str(int(v)) for v in nodal]
    _plot_tikz(disc, labels_before, "before_pd")

    disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls = _compress_pipeline(
        disc, nodal
    )

    # Canonical coarsening does not change anything
    assert len(disc_can) == 5
    # Downsplit merges the right column: 5 → 4 boxes
    assert len(disc_pd) == 4
    assert len(disc_ls) == 4
    assert (
        dyada.discretization_to_2d_ascii(disc_pd)
        == """\
_________
|___|   |
|_|_|___|"""
    )

    # Verify leaf scaling values are losslessly preserved
    scalings = get_leaf_scalings(disc_ls, coeff_ls)
    np.testing.assert_array_equal(scalings, np.array([0, 1, 0, 1]))


def test_3d_level_sweep_compresses_beyond_downsplit():
    """Test that level sweep compresses further than downsplit in 3D.

    In 3D, downsplitting a 3-way refined node leaves 2-way refined children.
    The default downsplit (which targets only the deepest multi-dim nodes) may
    stop before processing those shallower 2-way nodes.  Level sweep explicitly
    schedules all remaining multi-dim depths and achieves further compression.
    """
    disc = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(3, [1, 1, 2]),
    )
    nodal = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    assert len(disc) == 16

    disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls = _compress_pipeline(
        disc, nodal
    )

    # Canonical coarsening reduces from 16 to 13
    assert len(disc_can) == 13

    # Downsplit reduces to 12 but leaves multi-dim nodes at shallower depths
    assert len(disc_pd) == 12

    # Level sweep processes those shallower multi-dim nodes: 12 → 8
    assert len(disc_ls) == 8
    scalings_ls = get_leaf_scalings(disc_ls, coeff_ls)

    # Verify lossless reconstruction
    from wavelets_with_omnitrees import get_resampled_image

    target_level = disc.descriptor.get_maximum_level().astype(np.int64)
    raster_orig = get_resampled_image(disc, nodal, target_level)
    raster_new = get_resampled_image(disc_ls, scalings_ls, target_level)
    np.testing.assert_array_equal(raster_orig, raster_new)


class TestReconstructFromFile:
    """Verify that loading a descriptor and reconstructing coefficients via
    _sample_at_finest_midpoints gives the same result as the full pipeline."""

    def test_level_sweep_from_loaded_downsplit_2d(self):
        dim, level = 2, 3
        disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls = _full_pipeline(
            dim, level, _sphere_fn
        )

        with tempfile.TemporaryDirectory() as tmp:
            # Save downsplit descriptor
            disc_pd.descriptor.to_file(str(Path(tmp) / "pd"))
            pd_file = Path(tmp) / f"pd_{dim}d.bin"
            assert pd_file.exists()

            # Load and reconstruct
            loaded_disc = _load_discretization(pd_file)
            leaf_values = _sample_at_finest_midpoints(loaded_disc, _sphere_fn, level)
            loaded_coeff = _hierarchize_on_tree(loaded_disc, leaf_values)

            # Run level_sweep from loaded coefficients
            disc_ls2, coeff_ls2 = compress_by_level_sweep_coarsening(
                loaded_disc,
                [list(c) for c in loaded_coeff],
                coarsening_threshold=0.0,
            )

        # Same number of boxes
        assert len(disc_ls2) == len(disc_ls)
        # Same leaf scaling values
        np.testing.assert_array_equal(
            get_leaf_scalings(disc_ls2, coeff_ls2),
            get_leaf_scalings(disc_ls, coeff_ls),
        )

    def test_level_sweep_from_loaded_downsplit_3d(self):
        dim, level = 3, 3
        disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls = _full_pipeline(
            dim, level, _sphere_fn
        )

        with tempfile.TemporaryDirectory() as tmp:
            disc_pd.descriptor.to_file(str(Path(tmp) / "pd"))
            pd_file = Path(tmp) / f"pd_{dim}d.bin"

            loaded_disc = _load_discretization(pd_file)
            leaf_values = _sample_at_finest_midpoints(loaded_disc, _sphere_fn, level)
            loaded_coeff = _hierarchize_on_tree(loaded_disc, leaf_values)

            disc_ls2, coeff_ls2 = compress_by_level_sweep_coarsening(
                loaded_disc,
                [list(c) for c in loaded_coeff],
                coarsening_threshold=0.0,
            )

        assert len(disc_ls2) == len(disc_ls)
        np.testing.assert_array_equal(
            get_leaf_scalings(disc_ls2, coeff_ls2),
            get_leaf_scalings(disc_ls, coeff_ls),
        )

    def test_downsplit_from_loaded_canonical_2d(self):
        dim, level = 2, 3
        disc_can, coeff_can, disc_pd, coeff_pd, disc_ls, coeff_ls = _full_pipeline(
            dim, level, _sphere_fn
        )

        with tempfile.TemporaryDirectory() as tmp:
            disc_can.descriptor.to_file(str(Path(tmp) / "can"))
            can_file = Path(tmp) / f"can_{dim}d.bin"

            loaded_disc = _load_discretization(can_file)
            leaf_values = _sample_at_finest_midpoints(loaded_disc, _sphere_fn, level)
            loaded_coeff = _hierarchize_on_tree(loaded_disc, leaf_values)

            disc_pd2, coeff_pd2 = compress_by_downsplit_coarsening(
                loaded_disc,
                [list(c) for c in loaded_coeff],
                coarsening_threshold=0.0,
            )
            disc_ls2, coeff_ls2 = compress_by_level_sweep_coarsening(
                disc_pd2,
                [list(c) for c in coeff_pd2],
                coarsening_threshold=0.0,
            )

        assert len(disc_pd2) == len(disc_pd)
        np.testing.assert_array_equal(
            get_leaf_scalings(disc_pd2, coeff_pd2),
            get_leaf_scalings(disc_pd, coeff_pd),
        )
        assert len(disc_ls2) == len(disc_ls)
        np.testing.assert_array_equal(
            get_leaf_scalings(disc_ls2, coeff_ls2),
            get_leaf_scalings(disc_ls, coeff_ls),
        )
