# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduce completed contact buffers using ordered UInt32 atomic phases.

Candidate keys are immutable before a table slot publishes their row index.
The high-word maximum completes before the low-word maximum starts. These
phases implement the existing packed-value ordering without exposing a partial
64-bit atomic transaction. Both value passes use the same compiled kernel with
a runtime Boolean argument, so immutable inputs produce the same score bits in
both passes. Splitting their floating-point arithmetic into distinct pipelines
could change rounding and invalidate the tied-high-word test in the low pass.
Speculative and inline SDF reduction are separate algorithms and are not
supported by this companion.
"""

from __future__ import annotations

import warp as wp

from .contact_reduction import (
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_voxel_index,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .contact_reduction_global import (
    BETA_THRESHOLD,
    VALUES_PER_KEY,
    GlobalContactReducer,
    GlobalContactReducerData,
    decode_oct,
    make_contact_key,
    make_contact_value,
)
from .hashtable import _hashtable_hash


@wp.func
def _find_or_insert_key_row(
    row: int,
    candidate_keys: wp.array[wp.uint64],
    slot_rows: wp.array[int],
    published_keys: wp.array[wp.uint64],
    active_slots: wp.array[int],
) -> int:
    """Publish a row reference whose key was completed in an earlier dispatch."""
    capacity = slot_rows.shape[0]
    mask = capacity - 1
    key = candidate_keys[row]
    slot = _hashtable_hash(key, mask)
    for _attempt in range(capacity):
        previous = wp.atomic_cas(slot_rows, slot, -1, row)
        if previous == -1:
            # No insertion-phase reader uses published_keys or active_slots.
            # Their only writer owns this slot; later dispatches see both words.
            published_keys[slot] = key
            ticket = wp.atomic_add(active_slots, capacity, 1)
            active_slots[ticket] = slot
            return slot
        if candidate_keys[previous] == key:
            return slot
        slot = (slot + 1) & mask
    return -1


@wp.func
def _reduce_packed_word(
    index: int,
    value: wp.uint64,
    high: wp.array[wp.uint32],
    low: wp.array[wp.uint32],
    high_pass: bool,
):
    """Reduce one limb while retaining the complete UInt64 comparison order."""
    value_high = wp.uint32(value >> wp.uint64(32))
    if high_pass:
        wp.atomic_max(high, index, value_high)
    elif value_high == high[index]:
        wp.atomic_max(low, index, wp.uint32(value))


@wp.kernel(enable_backward=False)
def _build_candidate_keys(
    data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
    voxel_resolution: wp.array[wp.vec3i],
    candidate_keys: wp.array[wp.uint64],
    voxel_slots: wp.array[int],
):
    index = wp.tid()
    count = wp.min(data.contact_count[0], data.capacity)
    if index >= count:
        return
    contact_id = index + 1
    pair = data.shape_pairs[contact_id]
    normal_bin = get_slot(decode_oct(data.normal[contact_id]))
    candidate_keys[2 * index] = make_contact_key(pair[0], pair[1], normal_bin)
    pd = data.position_depth[contact_id]
    position = wp.vec3(pd[0], pd[1], pd[2])
    local_position = wp.transform_point(wp.transform_inverse(shape_transform[pair[0]]), position)
    voxel = compute_voxel_index(local_position, aabb_lower[pair[0]], aabb_upper[pair[0]], voxel_resolution[pair[0]])
    voxel = wp.clamp(voxel, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))
    voxel_slots[index] = voxel % wp.static(VALUES_PER_KEY)
    voxel_bin = voxel // wp.static(VALUES_PER_KEY)
    candidate_keys[2 * index + 1] = make_contact_key(pair[0], pair[1], wp.static(NUM_NORMAL_BINS) + voxel_bin)


@wp.kernel(enable_backward=False)
def _insert_candidate_keys(
    data: GlobalContactReducerData,
    candidate_keys: wp.array[wp.uint64],
    slot_rows: wp.array[int],
    entry_rows: wp.array[int],
):
    row = wp.tid()
    count = wp.min(data.contact_count[0], data.capacity)
    if row >= 2 * count:
        return
    entry = _find_or_insert_key_row(row, candidate_keys, slot_rows, data.ht_keys, data.ht_active_slots)
    entry_rows[row] = entry
    if entry < 0:
        wp.atomic_add(data.ht_insert_failures, 0, 1)


@wp.kernel(enable_backward=False)
def _reduce_candidate_words(
    data: GlobalContactReducerData,
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
    entry_rows: wp.array[int],
    voxel_slots: wp.array[int],
    high: wp.array[wp.uint32],
    low: wp.array[wp.uint32],
    high_pass: bool,
):
    index = wp.tid()
    count = wp.min(data.contact_count[0], data.capacity)
    if index >= count:
        return
    contact_id = index + 1
    pd = data.position_depth[contact_id]
    depth = pd[3]
    fingerprint = data.contact_fingerprints[contact_id]
    normal_entry = entry_rows[2 * index]
    if normal_entry >= 0:
        shape_a = data.shape_pairs[contact_id][0]
        use_beta = depth < wp.static(BETA_THRESHOLD) * wp.length(aabb_upper[shape_a] - aabb_lower[shape_a])
        if use_beta:
            bin_id = get_slot(decode_oct(data.normal[contact_id]))
            projected = project_point_to_plane(bin_id, wp.vec3(pd[0], pd[1], pd[2]))
            for direction in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                score = wp.dot(projected, get_spatial_direction_2d(direction))
                value = make_contact_value(score, fingerprint, contact_id, data.deterministic)
                _reduce_packed_word(direction * data.ht_capacity + normal_entry, value, high, low, high_pass)
        value = make_contact_value(-depth, fingerprint, contact_id, data.deterministic)
        _reduce_packed_word(
            wp.static(NUM_SPATIAL_DIRECTIONS) * data.ht_capacity + normal_entry, value, high, low, high_pass
        )
    voxel_entry = entry_rows[2 * index + 1]
    if voxel_entry >= 0:
        value = make_contact_value(-depth, fingerprint, contact_id, data.deterministic)
        _reduce_packed_word(voxel_slots[index] * data.ht_capacity + voxel_entry, value, high, low, high_pass)


@wp.kernel(enable_backward=False)
def reject_inline_reduction_routes(pair_count: wp.array[int]):
    """Report an unsupported inline SDF route without copying counters to the host."""
    if pair_count[0] > 0:
        wp.printf("Error: WebGPU buffered contact reduction does not support inline mesh-SDF contacts\n")


class BufferedContactReducer:
    """Reduce a freshly cleared, completed non-speculative contact buffer.

    Args:
        reducer: Existing reducer whose contact buffer and output table are reused.

    Call ``clear()`` alongside the reducer's ordinary clear, before producing
    contacts. ``reduce()`` consumes the immutable buffer once; it preserves its
    contact IDs, capacity, overflow counters, key/value packing and export layout.
    """

    def __init__(self, reducer: GlobalContactReducer):
        if not reducer.enable_reduction:
            raise ValueError("Buffered contact reduction requires an enabled reduction table")
        if reducer.store_hydroelastic_data or reducer.enable_contact_reclamation:
            raise NotImplementedError(
                "Buffered contact reduction does not support hydroelastic or provisional contacts"
            )
        if reducer.capacity > 0x3FFFFFFF:
            raise ValueError("Buffered contact reduction candidate row indices must fit Int32")
        self.reducer = reducer
        self.device = reducer.position_depth.device
        self.candidate_keys = wp.empty(2 * reducer.capacity, dtype=wp.uint64, device=self.device)
        self.entry_rows = wp.empty(2 * reducer.capacity, dtype=int, device=self.device)
        self.voxel_slots = wp.empty(reducer.capacity, dtype=int, device=self.device)
        self.slot_rows = wp.full(reducer.hashtable.capacity, -1, dtype=int, device=self.device)
        # Native UInt64 storage and its UInt32 limbs share the same allocations.
        # The view chain retains its owner; each completed phase writes only one
        # limb, and ordinary UInt64 reads begin after both phases have finished.
        self.words = reducer.ht_values.view(wp.vec2ui).view(wp.uint32)
        self.low = self.words[:, 0]
        self.high = self.words[:, 1]

    def clear(self):
        """Clear only row references without resetting any reducer counters."""
        self.slot_rows.fill_(-1)

    def reduce(self, shape_transform, aabb_lower, aabb_upper, voxel_resolution):
        """Launch ordered key publication and exact high/low maximum passes."""
        data = self.reducer.get_data_struct()
        wp.launch(
            _build_candidate_keys,
            dim=self.reducer.capacity,
            inputs=[
                data,
                shape_transform,
                aabb_lower,
                aabb_upper,
                voxel_resolution,
                self.candidate_keys,
                self.voxel_slots,
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _insert_candidate_keys,
            dim=2 * self.reducer.capacity,
            inputs=[data, self.candidate_keys, self.slot_rows, self.entry_rows],
            device=self.device,
            record_tape=False,
        )
        # Keep one pipeline and a runtime flag: independently compiled score
        # arithmetic must not choose different high words between these passes.
        for high_pass in (True, False):
            wp.launch(
                _reduce_candidate_words,
                dim=self.reducer.capacity,
                inputs=[
                    data,
                    aabb_lower,
                    aabb_upper,
                    self.entry_rows,
                    self.voxel_slots,
                    self.high,
                    self.low,
                    high_pass,
                ],
                device=self.device,
                record_tape=False,
            )
