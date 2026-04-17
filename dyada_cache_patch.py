"""Monkey-patch dyada's RefinementDescriptor with branch and ancestry caches.

Usage::

    import dyada_cache_patch
    dyada_cache_patch.install()

    # ... use dyada as normal; get_branch and get_ancestry_by_index are now cached ...

    dyada_cache_patch.uninstall()  # optional: restore originals
"""

import bisect
import numpy as np

from dyada.descriptor import RefinementDescriptor, Branch, LevelCounter
from dyada.linearization import location_code_from_branch
import dyada.ancestrybranch as _ab_mod


# ── helpers ──────────────────────────────────────────────────────────────────


def _deep_copy_branch(branch: Branch) -> Branch:
    """Deep-copy a Branch (LevelCounter.count_to_go_up is mutable)."""
    bc = Branch(branch)
    for k in range(len(bc)):
        lc = bc[k]
        bc[k] = LevelCounter(lc.level_increment, lc.count_to_go_up)
    return bc


# ── parent cache (O(n) build, O(depth) ancestry lookup) ─────────────────────


def _ensure_parent_cache(self: RefinementDescriptor) -> None:
    """Build ``_parent_cache[i]`` = direct parent index (-1 for root)."""
    if hasattr(self, "_parent_cache"):
        return

    n = len(self)
    parent_cache = [-1] * n
    parent_stack: list[int] = []
    remaining: list[int] = []

    for i, ref in enumerate(self):
        parent_cache[i] = parent_stack[-1] if parent_stack else -1
        if ref != self.d_zeros:
            remaining.append(1 << ref.count())
            parent_stack.append(i)
        else:
            while remaining:
                remaining[-1] -= 1
                if remaining[-1] > 0:
                    break
                remaining.pop()
                parent_stack.pop()

    self._parent_cache = parent_cache


def _get_ancestry_by_index(
    self: RefinementDescriptor, hierarchical_index: int
) -> list[int]:
    """Sorted ancestry via parent chain — O(depth)."""
    _ensure_parent_cache(self)
    ancestry: list[int] = []
    idx = self._parent_cache[hierarchical_index]
    while idx != -1:
        ancestry.append(idx)
        idx = self._parent_cache[idx]
    ancestry.reverse()
    return ancestry


# ── children cache (derived from parent cache) ──────────────────────────────

_original_get_children = RefinementDescriptor.get_children


def _cached_get_children(
    self: RefinementDescriptor, parent_index, branch_to_parent=None
):
    """Return direct children of ``parent_index`` by scanning ``_parent_cache``"""
    if self.is_box(parent_index):
        return []
    _ensure_parent_cache(self)
    parent_cache = self._parent_cache
    num_children = 1 << self[parent_index].count()
    children = [parent_index + 1]
    i = parent_index + 1
    while len(children) < num_children:
        i += 1
        if parent_cache[i] == parent_index:
            children.append(i)
    return children


# ── branch cache (on-demand, nearest-hint) ───────────────────────────────────

_original_get_branch = RefinementDescriptor.get_branch


def _traverse_branch(self, index, hint_previous_branch):
    """Core traversal for hierarchical (non-box) index — no cache logic."""
    d_zeros = self.d_zeros
    if hint_previous_branch is None:
        current_branch = Branch(self._num_dimensions)
        i = 0
        current_iterator = iter(self)
    else:
        i, current_branch = hint_previous_branch
        current_branch = current_branch.copy()
        current_iterator = self.__iter__(start=i)
    while i < index:
        current_refinement = next(current_iterator)
        if current_refinement == d_zeros:
            current_branch.advance_branch()
        else:
            current_branch.grow_branch(current_refinement)
        i += 1
    return current_branch, current_iterator


def _cached_get_branch(self, index, is_box_index=True, hint_previous_branch=None):
    if index < 0 or index >= len(self):
        raise IndexError("Index out of range")

    if not is_box_index:
        cache: dict[int, Branch] = self.__dict__.setdefault("_branch_cache", {})
        keys: list[int] = self.__dict__.setdefault("_branch_cache_keys", [])

        if index in cache:
            return _deep_copy_branch(cache[index]), self.__iter__(start=index)

        # Use nearest smaller cached key if closer than caller's hint
        if keys:
            pos = bisect.bisect_right(keys, index) - 1
            if pos >= 0:
                best = keys[pos]
                hint_idx = (
                    hint_previous_branch[0] if hint_previous_branch is not None else -1
                )
                if best > hint_idx:
                    hint_previous_branch = (best, _deep_copy_branch(cache[best]))

        # Compute, cache, return
        branch, it = _traverse_branch(self, index, hint_previous_branch)
        cache[index] = _deep_copy_branch(branch)
        bisect.insort(keys, index)
        return branch, it

    return _original_get_branch(self, index, is_box_index, hint_previous_branch)


# ── patched _is_old_index... using get_ancestry_by_index ─────────────────────

_original_is_old = _ab_mod._is_old_index_now_at_or_containing_location_code


def _patched_is_old(
    discretization,
    markers,
    desired_dimensionwise_positions,
    parent_of_next_refinement,
    parent_branch,
    old_index,
):
    descriptor = discretization.descriptor
    old_index_branch, _ = descriptor.get_branch(
        old_index,
        is_box_index=False,
        hint_previous_branch=(parent_of_next_refinement, parent_branch),
    )
    old_index_dimensionwise_positions = location_code_from_branch(
        old_index_branch, discretization._linearization
    )

    # Use cached ancestry instead of O(n) get_ancestry
    old_index_ancestors = descriptor.get_ancestry_by_index(old_index)
    old_index_ancestry_accumulated_markers = np.sum(
        [markers[ancestor] for ancestor in old_index_ancestors],
        axis=0,
    )
    shortened_parent_positions = [
        desired_dimensionwise_positions[d][
            : len(desired_dimensionwise_positions[d])
            - old_index_ancestry_accumulated_markers[d]
        ]
        for d in range(descriptor.get_num_dimensions())
    ]
    part_of_history = all(
        _ab_mod.bitarray_startswith(
            old_index_dimensionwise_positions[d],
            shortened_parent_positions[d],
        )
        for d in range(descriptor.get_num_dimensions())
    )
    return part_of_history, old_index_dimensionwise_positions


# ── patched _AncestryMappingState.__init__ ────────────────────────────────────

_AncestryBranch = _ab_mod.AncestryBranch
_AncestryMappingState = _AncestryBranch._AncestryMappingState
_original_mapping_state_init = _AncestryMappingState.__init__


def _patched_mapping_state_init(
    self, discretization, ancestry_branch_state, starting_index=None
):
    if starting_index is not None:
        # Fast path: use cached parent chain instead of O(n) get_ancestry
        self.ancestry = discretization.descriptor.get_ancestry_by_index(starting_index)
    else:
        self.ancestry = discretization.descriptor.get_ancestry(
            ancestry_branch_state.current_modified_branch
        )
    from collections import defaultdict
    from dyada.linearization import TrackToken

    self.old_indices_map_track_tokens = defaultdict(list)
    self.current_track_token = TrackToken(-1)
    self.missed_mappings = defaultdict(set)
    self.track_info_mapping = {}


_original_ab_init = _AncestryBranch.__init__


def _patched_ab_init(self, discretization, starting_index, markers):
    self.markers = markers
    self._discretization = discretization
    self._branch_state = _AncestryBranch._AncestryBranchState(
        discretization, starting_index
    )
    self._mapping_state = _AncestryMappingState(
        discretization, self._branch_state, starting_index=starting_index
    )
    assert (
        len(self._mapping_state.ancestry) == self._branch_state.initial_branch_depth - 1
    )


# ── install / uninstall ─────────────────────────────────────────────────────

_installed = False


def install():
    """Monkey-patch dyada with branch + ancestry caches."""
    global _installed
    if _installed:
        return
    RefinementDescriptor.get_branch = _cached_get_branch
    RefinementDescriptor.get_ancestry_by_index = _get_ancestry_by_index
    RefinementDescriptor.get_children = _cached_get_children
    _ab_mod._is_old_index_now_at_or_containing_location_code = _patched_is_old
    _AncestryMappingState.__init__ = _patched_mapping_state_init
    _AncestryBranch.__init__ = _patched_ab_init
    _installed = True


def uninstall():
    """Restore original dyada methods."""
    global _installed
    if not _installed:
        return
    RefinementDescriptor.get_branch = _original_get_branch
    RefinementDescriptor.get_children = _original_get_children
    if hasattr(RefinementDescriptor, "get_ancestry_by_index"):
        delattr(RefinementDescriptor, "get_ancestry_by_index")
    _ab_mod._is_old_index_now_at_or_containing_location_code = _original_is_old
    _AncestryMappingState.__init__ = _original_mapping_state_init
    _AncestryBranch.__init__ = _original_ab_init
    _installed = False
