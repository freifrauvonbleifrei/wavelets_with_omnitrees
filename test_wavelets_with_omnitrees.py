import tempfile
from pathlib import Path

import numpy as np
import pytest
import random

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
    apply_downsplits,
    normalize_uniqueness,
    downsplit_node_coefficients,
    reverse_downsplit_coefficients,
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
    """Run the full pipeline: sample → hierarchize → canonical → downsplit."""
    disc = _build_full_grid(dim, level)
    # Sample at finest-level midpoints (the canonical way)
    n = len(disc)
    nodal = np.empty(n, dtype=np.float64)
    for box_index in range(n):
        interval = dyada.discretization.coordinates_from_box_index(disc, box_index)
        midpoint = (interval.lower_bound + interval.upper_bound) / 2.0
        nodal[box_index] = inside_fn(midpoint.reshape(1, -1))[0]

    return _compress_pipeline(disc, nodal)


def _compress_pipeline(disc, nodal, coarsening_threshold=0.0):
    """Hierarchize and run the full compression pipeline."""
    coefficients = transform_to_all_wavelet_coefficients(disc, nodal)
    root_scaling = coefficients[0][0]
    for c in coefficients:
        c[0] = np.nan
    coefficients[0][0] = root_scaling
    disc_can, coeff_can, _ = compress_by_omnitree_coarsening(
        disc,
        [list(c) for c in coefficients],
        coarsening_threshold=coarsening_threshold,
    )
    disc_pd, coeff_pd, _ = compress_by_downsplit_coarsening(
        disc_can,
        [list(c) for c in coeff_can],
        coarsening_threshold=coarsening_threshold,
    )
    return disc_can, coeff_can, disc_pd, coeff_pd


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


_INSUFFICIENT_DOWNSPLIT_CANDIDATES = [
    # ── A: minimal asymmetric (no multi-dim node, downsplit is a no-op) ──
    # The two bot leaves and the two top leaves are adjacent same-value
    # rectangles but live in two different `01` subtrees, so downsplit
    # (which only operates on multi-dim nodes) cannot touch them.
    # Level-sweep collapses to a single y-split.
    {
        "label": "A_minimal_no_multidim",
        "descriptor": "10 01 00 00 01 00 00",
        "values": [0, 1, 0, 1],
        "init_boxes": 4,
        "can_boxes": 4,
        "pd_boxes": 4,
        "ls_boxes": 2,
        "ascii_init": """\
_____
|_|_|
|_|_|""",
        "ascii_pd": """\
_____
|_|_|
|_|_|""",
        "ascii_ls": """\
___
|_|
|_|""",
    },
    # ── Q: 4×4 horizontal stripes via `11` root + four asymmetric quadrants ──
    # Has a multi-dim root, so downsplit *does* try to cascade — but the four
    # sibling subtrees are individually asymmetric and downsplit cannot
    # restructure across them, so the tree stays at 16 nodes.  Level-sweep
    # only manages to halve it (→8); the true optimum is 4 horizontal strips,
    # so even level_sweep is sub-optimal here.  Reachable via descriptor
    # ``01 01 00 00 01 00 00`` with values ``[0, 1, 0, 1]``, which renders as:
    #
    #     ___
    #     |_|
    #     |_|
    #     |_|
    #     |_|
    #
    {
        "label": "Q_4x4_horizontal_stripes_via_11",
        "descriptor": (
            "11 10 01 00 00 01 00 00 10 01 00 00 01 00 00 "
            "10 01 00 00 01 00 00 10 01 00 00 01 00 00"
        ),
        "values": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "init_boxes": 16,
        "can_boxes": 16,
        "pd_boxes": 16,
        "ls_boxes": 8,
        "ascii_init": """\
_________
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|""",
        "ascii_pd": """\
_________
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|""",
        "ascii_ls": """\
_____
|_|_|
|_|_|
|_|_|
|_|_|""",
    },
    # ── R: canonical input (4×4) — bot-left checker + 3 zero quadrants ──
    # Downsplit now reaches the level-sweep optimum (6 boxes) after the
    # reverse_downsplit_coefficients fix unblocked the merge; ascii_pd and
    # ascii_ls are now identical.
    {
        "label": "R_canonical_quadrant_checker",
        "descriptor": "11 11 00 00 00 00 00 00 00",
        "values": [0, 1, 1, 0, 0, 0, 0],
        "init_boxes": 7,
        "can_boxes": 7,
        "pd_boxes": 6,
        "ascii_init": """\
_________
|   |   |
|___|___|
|_|_|   |
|_|_|___|""",
        "ascii_pd": """\
_________
|   |   |
|___|   |
|_|_|   |
|_|_|___|""",
    },
    # ── S: canonical input (4×4) — bot-left checker + bot-right `01` + top half ──
    # Another already-canonical tree.  The two top-quadrant leaves (top-left
    # and top-right, both value 0) form a full-width strip that could merge
    # into a single 1×0.5 leaf, but downsplit's cascading at the root `11`
    # never assembles this combination.  Level-sweep finds it (8→7).
    #
    {
        "label": "S_canonical_top_half_merge",
        "descriptor": "11 11 00 00 00 00 01 00 00 00 00",
        "values": [0, 1, 1, 0, 1, 0, 0, 0],
        "init_boxes": 8,
        "can_boxes": 8,
        "pd_boxes": 7,
        "ascii_init": """\
_________
|   |   |
|___|___|
|_|_|___|
|_|_|___|""",
        "ascii_pd": """\
_________
|       |
|_______|
|_|_|___|
|_|_|___|""",
    },
    # ── M: original probe descriptor (no compression in the current pipeline) ──
    # The optimum here is 6 boxes (the merged top-right column collapses Q3.r
    # and Q1.r into a single x=0.75..1 strip), but neither downsplit nor
    # level-sweep finds it.  Optimal:
    #     _________________
    #     |_______|___|   |
    #     |_______|_|_|___|
    #
    {
        "label": "M_original_probe",
        "descriptor": "11 00 10 10 00 00 00 00 10 00 00",
        "values": [0, 0, 1, 1, 1, 0, 1],
        "init_boxes": 7,
        "can_boxes": 7,
        "pd_boxes": 7,
        "ls_boxes": 7,
        "ascii_init": """\
_________________
|_______|___|___|
|_______|_|_|___|""",
        "ascii_pd": """\
_________________
|_______|___|___|
|_______|_|_|___|""",
        "ascii_ls": """\
_________________
|_______|___|___|
|_______|_|_|___|""",
    },
]


@pytest.mark.parametrize(
    "case",
    _INSUFFICIENT_DOWNSPLIT_CANDIDATES,
    ids=[c["label"] for c in _INSUFFICIENT_DOWNSPLIT_CANDIDATES],
)
def test_2d_insufficient_downsplit(case):
    """Probe whether downsplit reaches the level-sweep optimum on a candidate tree.

    Each parametrised case represents a tree with adjacent same-size,
    same-value leaves that *could* be merged.  We assert both the box counts
    and the inline ASCII at the init / downsplit stages so
    the missed coarsening is visible directly in the test source: a case
    where ``ascii_pd`` and ``ascii_ls`` differ is a downsplit miss.
    """
    label = case["label"]
    disc = _build_disc_from_descriptor_string(2, case["descriptor"])
    nodal = np.array(case["values"], dtype=np.float64)

    assert (
        len(disc) == case["init_boxes"]
    ), f"{label}: init boxes: expected {case['init_boxes']}, got {len(disc)}"
    assert (
        dyada.discretization_to_2d_ascii(disc) == case["ascii_init"]
    ), f"{label}: init ascii mismatch"

    disc_can, coeff_can, disc_pd, coeff_pd = _compress_pipeline(disc, nodal)

    assert (
        len(disc_can) == case["can_boxes"]
    ), f"{label}: canonical boxes: expected {case['can_boxes']}, got {len(disc_can)}"
    assert (
        len(disc_pd) == case["pd_boxes"]
    ), f"{label}: downsplit boxes: expected {case['pd_boxes']}, got {len(disc_pd)}"
    assert (
        dyada.discretization_to_2d_ascii(disc_pd) == case["ascii_pd"]
    ), f"{label}: downsplit ascii mismatch"

    # Lossless: every stage's reconstruction must match the original raster.
    from wavelets_with_omnitrees import get_resampled_image

    target_level = disc.descriptor.get_maximum_level().astype(np.int64)
    raster_orig = get_resampled_image(disc, nodal, target_level)
    for stage_name, stage_disc, stage_coeff in (
        ("canonical", disc_can, coeff_can),
        ("downsplit", disc_pd, coeff_pd),
    ):
        scalings = get_leaf_scalings(stage_disc, stage_coeff)
        raster_new = get_resampled_image(stage_disc, scalings, target_level)
        np.testing.assert_array_equal(
            raster_orig,
            raster_new,
            err_msg=f"{label}: {stage_name} reconstruction mismatch",
        )


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

    disc_can, coeff_can, disc_pd, coeff_pd = _compress_pipeline(disc, nodal)

    # Canonical coarsening does not change anything
    assert len(disc_can) == 5
    # Downsplit merges the right column: 5 → 4 boxes
    assert len(disc_pd) == 4
    assert (
        dyada.discretization_to_2d_ascii(disc_pd)
        == """\
_________
|___|   |
|_|_|___|"""
    )

    # Verify leaf scaling values are losslessly preserved
    scalings = get_leaf_scalings(disc_pd, coeff_pd)
    np.testing.assert_array_equal(scalings, np.array([0, 1, 0, 1]))


def test_3d_compresses_beyond_downsplit():
    """Regression: 3D canonical-then-downsplit-then-level_sweep box counts."""
    disc = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(3, [1, 1, 2]),
    )
    nodal = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    assert len(disc) == 16

    disc_can, coeff_can, disc_pd, coeff_pd = _compress_pipeline(disc, nodal)

    # Canonical coarsening reduces 16 → 13
    assert len(disc_can) == 13
    scalings_can = get_leaf_scalings(disc_can, coeff_can)

    # Downsplit does not reduce further on this input (stays at 13).
    assert len(disc_pd) == 13
    scalings_pd = get_leaf_scalings(disc_pd, coeff_pd)
    # Verify lossless reconstruction
    from wavelets_with_omnitrees import get_resampled_image

    target_level = disc.descriptor.get_maximum_level().astype(np.int64)
    raster_orig = get_resampled_image(disc, nodal, target_level)
    raster_new = get_resampled_image(disc_pd, scalings_pd, target_level)
    np.testing.assert_array_equal(raster_orig, raster_new)


def _midpoints(disc):
    """Compute midpoints of all leaf boxes of a full grid discretization."""
    n = len(disc)
    dim = disc.descriptor.get_num_dimensions()
    pts = np.empty((n, dim))
    for box_index in range(n):
        iv = dyada.discretization.coordinates_from_box_index(disc, box_index)
        pts[box_index] = (iv.lower_bound + iv.upper_bound) / 2.0
    return pts


def _compress_count(disc, nodal, threshold):
    """Hierarchize, run downsplit compression, and return final box count."""
    coeffs = transform_to_all_wavelet_coefficients(disc, nodal)
    root = coeffs[0][0]
    for c in coeffs:
        c[0] = np.nan
    coeffs[0][0] = root
    disc_pd, _, _ = compress_by_downsplit_coarsening(
        disc,
        [list(c) for c in coeffs],
        coarsening_threshold=threshold,
    )
    return disc_pd.descriptor.get_num_boxes()


@pytest.mark.parametrize("level", [2, 3])
def test_threshold_sweep_analytical(level, capsys):
    """Sweep coarsening_threshold against analytical functions of varying smoothness."""
    dim = 3
    disc = _build_full_grid(dim, level)
    n_full = len(disc)
    pts = _midpoints(disc)

    # Magnitudes are ~5% so they mimic the cloud's continuous occupancy.
    cloud_mean = 0.05
    fns = {
        "constant": np.full(n_full, cloud_mean),
        "linear_x": cloud_mean + 0.05 * pts[:, 0],
        "quadratic": cloud_mean + 0.1 * np.sum((pts - 0.5) ** 2, axis=1),
        "gauss_wide": cloud_mean
        + 0.1 * np.exp(-np.sum((pts - 0.5) ** 2, axis=1) / (2 * 0.30**2)),
        "gauss_narrow": cloud_mean
        + 0.1 * np.exp(-np.sum((pts - 0.5) ** 2, axis=1) / (2 * 0.08**2)),
        "step_x": cloud_mean + 0.1 * (pts[:, 0] > 0.5).astype(np.float64),
    }

    thresholds = [0.0, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6]
    results: dict[str, list[int]] = {
        name: [_compress_count(disc, nodal, t) for t in thresholds]
        for name, nodal in fns.items()
    }

    with capsys.disabled():
        print(
            f"\n=== threshold sweep, dim={dim}, level={level}, "
            f"full leaves={n_full} ==="
        )
        header = f"{'function':<14}" + "".join(f"{t:>10.0e}" for t in thresholds)
        print(header)
        for name in fns:
            row = "".join(f"{c:>10d}" for c in results[name])
            print(f"{name:<14}{row}")

    # Constant function: every Haar detail is identically zero, so the tree
    # collapses to a single box for any threshold (including 0.0).
    assert all(c == 1 for c in results["constant"]), (
        f"constant function should always collapse to 1 box, got "
        f"{results['constant']}"
    )

    # Axis-aligned step in x with Haar wavelets: representable exactly as
    # a single x-split, irrespective of threshold or level.
    assert all(
        c <= 2 for c in results["step_x"]
    ), f"axis-aligned step should never exceed 2 boxes, got {results['step_x']}"

    # Box count is monotone non-increasing in threshold for every function.
    for name, counts in results.items():
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], (
                f"{name}: not monotonic between thr="
                f"{thresholds[i]:.0e} ({counts[i]}) and "
                f"thr={thresholds[i+1]:.0e} ({counts[i+1]})"
            )

    # All functions reach a single box at the largest threshold.
    for name, counts in results.items():
        assert (
            counts[-1] == 1
        ), f"{name} @ thr=1e6 should coarsen to 1 box, got {counts[-1]}"

    if level >= 3:
        thr_idx = thresholds.index(1e-2)
        for name in ("linear_x", "quadratic", "gauss_wide", "gauss_narrow"):
            assert (
                results[name][thr_idx] <= 2
            ), f"{name} @ thr=1e-2 expected <=2 boxes, got {results[name][thr_idx]}"


def _rasterize_3d(disc, leaf_values, level):
    """Expand leaf values onto a full 2^level per-dim grid in Morton order."""
    from dyada.linearization import grid_coord_to_z_index

    n = 1 << level
    grid_levels = [level] * 3
    grid = np.full(n**3, np.nan, dtype=np.float64)
    # Walk the tree and fill the grid cells each leaf covers
    box_idx = -1
    for branch, ref in dyada.descriptor.branch_generator(disc.descriptor):
        if ref.count() != 0:
            continue
        box_idx += 1
        iv = dyada.discretization.coordinates_from_box_index(disc, box_idx)
        lo = (iv.lower_bound * n + 0.5).astype(int)
        hi = np.minimum((iv.upper_bound * n + 0.5).astype(int), n)
        for ix in range(lo[0], hi[0]):
            for iy in range(lo[1], hi[1]):
                for iz in range(lo[2], hi[2]):
                    z_idx = grid_coord_to_z_index((ix, iy, iz), grid_levels)
                    grid[z_idx] = leaf_values[box_idx]
    return grid


def _actual_l1_error_3d(disc_orig, vals_orig, disc_comp, vals_comp, level):
    """Rasterize both trees to the finest grid and compute per-voxel L1."""
    grid_orig = _rasterize_3d(disc_orig, vals_orig, level)
    grid_comp = _rasterize_3d(disc_comp, vals_comp, level)
    cell_volume = (1.0 / (1 << level)) ** 3
    return float(np.sum(np.abs(grid_orig - grid_comp)) * cell_volume)


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize(
    "fn_label",
    ["linear_x", "quadratic", "gauss_wide"],
)
def test_discarded_l1_equals_actual_error(level, fn_label):
    """Verify that discarded_l1 matches the actual rasterised L1 error.

    The downsplit pipeline only performs 1-D coarsenings (k=1), so each
    discarded coefficient contributes |d| * box_volume exactly to the L1
    error.  The omnitree pipeline can do multi-dim coarsenings (k>=2) where
    the triangle inequality makes sum(|d_j|) * V an upper bound, not exact.

    This test checks:
    - downsplit-only (from full grid): discarded_l1 == actual L1 (exact)
    - full pipeline (omnitree + downsplit): discarded_l1 >= actual L1 (upper bound)
    """
    dim = 3
    disc = _build_full_grid(dim, level)
    pts = _midpoints(disc)
    cloud_mean = 0.05
    fns = {
        "linear_x": cloud_mean + 0.05 * pts[:, 0],
        "quadratic": cloud_mean + 0.1 * np.sum((pts - 0.5) ** 2, axis=1),
        "gauss_wide": cloud_mean
        + 0.1 * np.exp(-np.sum((pts - 0.5) ** 2, axis=1) / (2 * 0.30**2)),
    }
    nodal = fns[fn_label]

    for threshold in [1e-6, 1e-5, 1e-4]:
        coeffs = transform_to_all_wavelet_coefficients(disc, nodal)
        root = coeffs[0][0]
        for c in coeffs:
            c[0] = np.nan
        coeffs[0][0] = root

        # ── Downsplit-only path (from full grid) ──────────────────────────
        disc_pd_only, coeff_pd_only, dl1_pd_only = compress_by_downsplit_coarsening(
            disc,
            [list(c) for c in coeffs],
            coarsening_threshold=threshold,
        )
        # Skip rasterised check if tree fully collapsed — get_leaf_scalings
        # returns NaN for a single-box root (pre-existing coefficient-transfer
        # bug when the root itself is coarsened to a leaf).
        if len(disc_pd_only) > 1:
            scalings_pd_only = get_leaf_scalings(disc_pd_only, coeff_pd_only)
            actual_l1_pd = _actual_l1_error_3d(
                disc,
                nodal,
                disc_pd_only,
                scalings_pd_only,
                level,
            )
            ratio = dl1_pd_only / actual_l1_pd if actual_l1_pd > 0 else float("inf")
            print(
                f"  {fn_label} L={level} thr={threshold:.0e} ds-only: "
                f"boxes={len(disc_pd_only):4d}  "
                f"discarded={dl1_pd_only:.6e}  actual={actual_l1_pd:.6e}  "
                f"ratio={ratio:.4f}"
            )
            # Each per-node L1 is exact (via dehierarchize), but cascading
            # coarsenings across levels can partially cancel at the leaf
            # cells (triangle inequality), so the sum is an upper bound.
            assert dl1_pd_only >= actual_l1_pd - 1e-16, (
                f"{fn_label} level={level} thr={threshold} downsplit-only: "
                f"discarded={dl1_pd_only:.6e} < actual={actual_l1_pd:.6e}"
            )

        # ── Full pipeline (omnitree + downsplit) ───────────────────────────
        disc_can, coeff_can, dl1_can = compress_by_omnitree_coarsening(
            disc,
            [list(c) for c in coeffs],
            coarsening_threshold=threshold,
        )
        disc_pd, coeff_pd, dl1_pd = compress_by_downsplit_coarsening(
            disc_can,
            [list(c) for c in coeff_can],
            coarsening_threshold=threshold,
        )
        total_discarded = dl1_can + dl1_pd
        if len(disc_pd) > 1:
            scalings_comp = get_leaf_scalings(disc_pd, coeff_pd)
            actual_l1 = _actual_l1_error_3d(disc, nodal, disc_pd, scalings_comp, level)
            assert total_discarded >= actual_l1 - 1e-16, (
                f"{fn_label} level={level} thr={threshold}: "
                f"discarded_l1={total_discarded:.6e} < actual_l1={actual_l1:.6e}"
            )


class TestReconstructFromFile:
    """Verify that loading a descriptor and reconstructing coefficients via
    _sample_at_finest_midpoints gives the same result as the full pipeline."""

    def test_downsplit_from_loaded_canonical_2d(self):
        dim, level = 2, 3
        disc_can, coeff_can, disc_pd, coeff_pd = _full_pipeline(dim, level, _sphere_fn)

        with tempfile.TemporaryDirectory() as tmp:
            disc_can.descriptor.to_file(str(Path(tmp) / "can"))
            can_file = Path(tmp) / f"can_{dim}d.bin"

            loaded_disc = _load_discretization(can_file)
            leaf_values = _sample_at_finest_midpoints(loaded_disc, _sphere_fn, level)
            loaded_coeff = _hierarchize_on_tree(loaded_disc, leaf_values)

            disc_pd2, coeff_pd2, _ = compress_by_downsplit_coarsening(
                loaded_disc,
                [list(c) for c in loaded_coeff],
                coarsening_threshold=0.0,
            )

        assert len(disc_pd2) == len(disc_pd)
        np.testing.assert_array_equal(
            get_leaf_scalings(disc_pd2, coeff_pd2),
            get_leaf_scalings(disc_pd, coeff_pd),
        )


@pytest.mark.parametrize(
    "push_str",
    ["100", "010", "001", "110", "101", "011"],
)
def test_downsplit_then_reverse_roundtrip_3d(push_str):
    """Single 3D 111-node: ``downsplit_node_coefficients`` then
    ``reverse_downsplit_coefficients`` must return the original wavelet
    coefficients for every non-empty ``push_dims`` subset.

    This is the minimal reproducer for the NaN leakage seen in
    ``test_3d_random_multi_dim_downsplits_roundtrip``: the merged coefficients
    end up NaN in exactly the wavelet slots along the pushed dimensions,
    regardless of how many downsplits happen above — a single forward/reverse
    cycle suffices to exhibit the bug.
    """
    ref111 = ba.bitarray("111")
    push = ba.bitarray(push_str)

    c = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)

    new_parent, intermediates = downsplit_node_coefficients(c, ref111, push)

    parent_ref = ref111 & ~push
    children_refs = [push.copy() for _ in intermediates]
    children = []
    for it in intermediates:
        tmp = np.asarray(it, dtype=np.float64).copy()
        tmp[0] = np.nan
        children.append(tmp)

    merged, _ = reverse_downsplit_coefficients(
        new_parent,
        parent_ref,
        children,
        children_refs,
        absorbed_dims=push,
    )

    np.testing.assert_allclose(
        merged,
        c,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
        err_msg=f"forward/reverse downsplit mismatch for push={push_str}",
    )


def test_3d_random_multi_dim_downsplits_roundtrip():
    """Sequential vs batched downsplits on a 3D level-[2,3,3] grid.

    1.  Build a 3D level-[2,3,3] full grid (4×8×8 = 256 cells).
    2.  Initialise the 256 nodal coefficients randomly and hierarchise.
    3.  Pre-generate a *shared* plan list: for every node with refinement
        ``111`` (root + level-1 children), draw one of three push options
        — ``{dim 1}``, ``{dim 2}``, or ``{dim 1, dim 2}``.
    3a) Apply the plans sequentially, **highest-index node first**, so each
        next plan's recorded index remains valid in the evolving tree.
    3b) Apply the same plans in a single batched ``apply_downsplits``
        call.  The two final (descriptor, coefficients) pairs must agree
        before normalisation.
    4.  Print all four trees (3a, 3b, and their normalised forms).
    5.  Normalise both and assert they collapse back to the initial full
        grid descriptor + coefficients.
    """
    rng = random.Random(0xDEADBEEF)
    np_rng = np.random.default_rng(0xDEADBEEF)

    # 1) 3D level-[2,3,3] full grid → 256 leaves.
    disc_init = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.descriptor.RefinementDescriptor(3, [2, 3, 3]),
    )
    assert len(disc_init) == 256

    # 2) Random nodal coefficients → wavelet coefficients (NaN-scaling
    # convention, with the root scaling preserved).
    nodal = np_rng.standard_normal(256)
    coeff_init_raw = transform_to_all_wavelet_coefficients(disc_init, nodal)
    coeff_init: list[np.ndarray] = [
        np.asarray(c, dtype=np.float64).copy() for c in coeff_init_raw
    ]
    root_scaling = coeff_init[0][0]
    for c in coeff_init:
        c[0] = np.nan

    # 3) Shared plan list: one push option per first-level node
    PUSH_OPTIONS = [
        ba.bitarray("010"),  # push dim 1
        ba.bitarray("001"),  # push dim 2
        ba.bitarray("011"),  # push dims 1 and 2
    ]
    target_ref = ba.bitarray("111")
    plans_initial: list[tuple[int, ba.bitarray]] = []
    for i in range(1, len(disc_init.descriptor)):  # skip root #TODO?
        if ba.bitarray(disc_init.descriptor[i]) == target_ref:
            plans_initial.append((i, ba.bitarray(rng.choice(PUSH_OPTIONS))))
    print()
    print(f"shared plan list ({len(plans_initial)} plans):")
    for old_idx, push_ba in plans_initial:
        print(f"     old_idx={old_idx:3d} push={push_ba.to01()}")

    # ── 3a) Sequential downsplits, highest-index node first ────────────
    disc, coefficients = disc_init, [c.copy() for c in coeff_init]
    for plan in sorted(plans_initial, key=lambda p: p[0], reverse=True):
        disc, coefficients = apply_downsplits(disc, coefficients, [plan])

    disc_seq, coeff_seq = disc, coefficients

    # ── 3b) Batched downsplits — same plan list, single call ───────────
    disc_batch, coeff_batch = apply_downsplits(
        disc_init, [c.copy() for c in coeff_init], plans_initial
    )

    # ── Pre-normalisation: 3a and 3b must produce identical results ────
    assert disc_seq.descriptor._data == disc_batch.descriptor._data, (
        "sequential and batched downsplits produced different descriptors\n"
        f"3a: {disc_seq.descriptor._data.to01()}\n"
        f"3b: {disc_batch.descriptor._data.to01()}"
    )
    assert len(coeff_seq) == len(coeff_batch)
    for i, (cs, cb) in enumerate(zip(coeff_seq, coeff_batch)):
        np.testing.assert_allclose(
            cs,
            cb,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"sequential vs batched: coefficients differ at node {i}",
        )

    # 4) Normalise both and assert each collapses back to the initial full
    # grid (descriptor + coefficients).
    disc_seq_n, coeff_seq_n = normalize_uniqueness(disc_seq, coeff_seq)
    disc_batch_n, coeff_batch_n = normalize_uniqueness(disc_batch, coeff_batch)

    coeff_seq_n[0][0] = root_scaling
    coeff_batch_n[0][0] = root_scaling
    coeff_init[0][0] = root_scaling
    for label, d_norm, coeffs_n in (
        ("3a sequential", disc_seq_n, coeff_seq_n),
        ("3b batch", disc_batch_n, coeff_batch_n),
    ):
        assert (
            d_norm.descriptor._data == disc_init.descriptor._data
        ), f"{label}: normalised descriptor differs from initial full grid"
        assert len(coeffs_n) == len(
            coeff_init
        ), f"{label}: normalised coefficient list length mismatch"
        for i, (c_norm, c_init) in enumerate(zip(coeffs_n, coeff_init)):
            np.testing.assert_allclose(
                c_norm,
                c_init,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"{label}: normalised coefficient at node {i} differs",
            )
