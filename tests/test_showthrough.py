"""Unit tests for the exact show-through feasibility analysis.

The tables are cross-checked against a direct brute-force enumeration of the
integer model on small cases, so the tests validate the math rather than just
restating the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.showthrough import (
    COEFF_MAX,
    GRAY_UNCERTAIN,
    NO_SOLUTION,
    review,
    tables_for,
)


def forward(kf: int, kb: int, f: int, b: int) -> tuple[int, int]:
    return min(255, f + (kf * b) // 10), min(255, b + (kb * f) // 10)


def brute_envelope(obs_f: int, obs_b: int, mode: int) -> tuple[int, int]:
    """Min/max feasible front depth over every coefficient pair (brute force)."""
    lo, hi = 256, -1
    for kf in range(COEFF_MAX + 1):
        for kb in range(COEFF_MAX + 1):
            pair_lo, pair_hi = 256, -1
            for f in range(256):
                if mode & 1 and f != 0:
                    continue
                for b in range(256):
                    if mode & 2 and b != 0:
                        continue
                    if forward(kf, kb, f, b) == (obs_f, obs_b):
                        pair_lo = min(pair_lo, f)
                        pair_hi = max(pair_hi, f)
            if pair_hi >= 0:
                lo = min(lo, pair_lo)
                hi = max(hi, pair_hi)
    return lo, hi


def synthetic_pair(
    seed: int = 0,
    size: tuple[int, int] = (40, 50),
    kf: int = 2,
    kb: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate observations from a known random ground truth."""
    rng = np.random.default_rng(seed)
    h, w = size
    f = rng.integers(0, 256, size=(h, w)).astype(np.int64)
    b = rng.integers(0, 256, size=(h, w)).astype(np.int64)
    front_zero = np.zeros(size, dtype=bool)
    back_zero = np.zeros(size, dtype=bool)
    front_zero[0, 0] = True
    back_zero[0, 1] = True
    f[0, 0] = 0
    b[0, 1] = 0
    obs_f = np.minimum(255, f + (kf * b) // 10).astype(np.uint8)
    obs_b = np.minimum(255, b + (kb * f) // 10).astype(np.uint8)
    front_gray = (255 - obs_f).astype(np.uint8)
    back_gray = (255 - obs_b).astype(np.uint8)
    return f, b, obs_f, obs_b, front_gray, back_gray, front_zero, back_zero


def test_tables_match_forward_model_for_one_coefficient_pair():
    kf, kb = 2, 3
    tables = tables_for(kf, kb)
    # Enumerate the full 256x256 true-depth grid and check every table entry.
    f = np.arange(256, dtype=np.int16)
    b = np.arange(256, dtype=np.int16)
    ff, bb = np.meshgrid(f, b, indexing="ij")
    of = np.minimum(255, ff + (kf * bb) // 10)
    ob = np.minimum(255, bb + (kb * ff) // 10)
    assert tables.feasible[0, of, ob].all()
    # Every recorded extremum must itself be an admissible front depth.
    seen = set()
    for i, j in zip(of.ravel().tolist(), ob.ravel().tolist()):
        seen.add((i, j))
    assert len(seen) == int(tables.feasible[0].sum())
    for ofv, obv in sorted(seen):
        members = ff[(of == ofv) & (ob == obv)]
        assert tables.f_min[0, ofv, obv] == members.min()
        assert tables.f_max[0, ofv, obv] == members.max()


def test_zero_coefficients_make_observations_equal_truth():
    tables = tables_for(0, 0)
    assert tables.feasible[0, 123, 45]
    assert tables.f_min[0, 123, 45] == tables.f_max[0, 123, 45] == 123
    # Pinned f=0 but the front observation carries positive depth: impossible.
    assert not tables.feasible[1, 10, 5]
    assert tables.f_min[1, 10, 5] == tables.f_max[1, 10, 5] == NO_SOLUTION


def test_pinned_modes_restrict_the_preimage():
    kf, kb = 4, 2
    tables = tables_for(kf, kb)
    ofv, obv = forward(kf, kb, 0, 80)
    assert tables.feasible[1, ofv, obv]
    assert tables.f_min[1, ofv, obv] == tables.f_max[1, ofv, obv] == 0
    ofv, obv = forward(kf, kb, 150, 0)
    assert tables.feasible[2, ofv, obv]
    # With b pinned to zero, f equals the front observation exactly.
    assert tables.f_min[2, ofv, obv] == tables.f_max[2, ofv, obv] == ofv
    # Both pinned: only the all-zero observation survives.
    assert tables.feasible[3, 0, 0]
    assert not tables.feasible[3, 1, 0]


def test_saturated_observations_remain_feasible():
    tables = tables_for(5, 5)
    ofv, obv = forward(5, 5, 250, 200)
    assert (ofv, obv) == (255, 255)
    assert tables.feasible[0, 255, 255]
    assert tables.f_min[0, 255, 255] <= 250 <= tables.f_max[0, 255, 255]


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_table_extrema_match_brute_force_on_sample(mode):
    rng = np.random.default_rng(42)
    for _ in range(24):
        ofv = int(rng.integers(0, 256))
        obv = int(rng.integers(0, 256))
        lo, hi = 256, -1
        for kf in range(COEFF_MAX + 1):
            for kb in range(COEFF_MAX + 1):
                tables = tables_for(kf, kb)
                if tables.feasible[mode, ofv, obv]:
                    lo = min(lo, int(tables.f_min[mode, ofv, obv]))
                    hi = max(hi, int(tables.f_max[mode, ofv, obv]))
        brute_lo, brute_hi = brute_envelope(ofv, obv, mode)
        assert (lo, hi) == (brute_lo, brute_hi)


def test_ground_truth_coefficients_are_feasible_and_envelope_covers_truth():
    f, b, obs_f, obs_b, front_gray, back_gray, fz, bz = synthetic_pair()
    result = review(front_gray, back_gray, fz, bz, threshold=128)
    assert result.feasible
    assert (2, 3) in result.feasible_coefficients
    # Reconstruct the global envelope only from globally feasible coefficient
    # pairs and check the ground truth always lies inside it.
    modes = np.zeros(f.shape, dtype=np.uint8)
    modes[fz] |= 1
    modes[bz] |= 2
    lo = np.full(f.shape, 256, dtype=np.int16)
    hi = np.full(f.shape, -1, dtype=np.int16)
    for kf, kb in result.feasible_coefficients:
        tables = tables_for(kf, kb)
        assert tables.feasible[modes, obs_f, obs_b].all()
        lo = np.minimum(lo, tables.f_min[modes, obs_f, obs_b])
        hi = np.maximum(hi, tables.f_max[modes, obs_f, obs_b])
    assert (lo.astype(np.int64) <= f).all()
    assert (f <= hi.astype(np.int64)).all()


def test_output_uses_only_black_white_gray_and_follows_envelope():
    _, _, _, _, front_gray, back_gray, fz, bz = synthetic_pair(seed=7)
    threshold = 100
    result = review(front_gray, back_gray, fz, bz, threshold=threshold)
    assert result.feasible
    image = result.image
    assert set(np.unique(image)).issubset({0, 255, GRAY_UNCERTAIN})

    modes = np.zeros(front_gray.shape, dtype=np.uint8)
    modes[fz] |= 1
    modes[bz] |= 2
    obs_f = 255 - front_gray
    obs_b = 255 - back_gray
    lo = np.full(front_gray.shape, 256, dtype=np.int16)
    hi = np.full(front_gray.shape, -1, dtype=np.int16)
    for kf, kb in result.feasible_coefficients:
        tables = tables_for(kf, kb)
        lo = np.minimum(lo, tables.f_min[modes, obs_f, obs_b])
        hi = np.maximum(hi, tables.f_max[modes, obs_f, obs_b])
    assert (image[hi < threshold] == 255).all()
    assert (image[lo >= threshold] == 0).all()
    straddle = (hi >= threshold) & (lo < threshold)
    assert (image[straddle] == GRAY_UNCERTAIN).all()


def test_front_ink_free_pixel_is_always_white():
    _, _, _, _, front_gray, back_gray, fz, bz = synthetic_pair()
    for threshold in (1, 128, 255):
        image = review(front_gray, back_gray, fz, bz, threshold=threshold).image
        assert image[0, 0] == 255  # f is pinned to 0 < every threshold >= 1


def test_infeasible_when_observation_contradicts_ink_free_mark():
    _, _, _, _, front_gray, back_gray, fz, bz = synthetic_pair()
    # A front ink-free pixel (f=0) can only show depth from bleed, at most
    # floor(5 * 255 / 10) = 127; forcing observed depth 255 is impossible.
    front_corrupt = front_gray.copy()
    front_corrupt[0, 0] = 0
    result = review(front_corrupt, back_gray, fz, bz, threshold=128)
    assert not result.feasible
    assert result.image is None
    assert len(result.infeasible_pixel_counts) == 36
    assert all(count >= 1 for count in result.infeasible_pixel_counts.values())


def test_infeasible_when_both_sides_pinned_but_observation_is_nonzero():
    # A pixel marked ink-free on both sides has f = b = 0, hence both observed
    # depths must be 0 (grayscale 255) regardless of the coefficients.
    rng = np.random.default_rng(11)
    front_gray = rng.integers(0, 256, size=(20, 20)).astype(np.uint8)
    back_gray = rng.integers(0, 256, size=(20, 20)).astype(np.uint8)
    fz = np.zeros((20, 20), bool)
    bz = np.zeros((20, 20), bool)
    fz[0, 0] = True
    bz[1, 1] = True
    fz[5, 5] = bz[5, 5] = True
    front_gray[5, 5] = 200  # observed front depth 55 while f = b = 0
    result = review(front_gray, back_gray, fz, bz, threshold=128)
    assert not result.feasible
    assert result.image is None


def test_ink_free_pixel_stays_white_at_max_threshold():
    _, _, _, _, front_gray, back_gray, fz, bz = synthetic_pair()
    result = review(front_gray, back_gray, fz, bz, threshold=255)
    assert result.feasible
    # Pinned f=0 can never reach threshold 255, so the marked pixel is white.
    assert (result.image[fz] == 255).all()
    assert set(np.unique(result.image)).issubset({0, 127, 255})


def test_review_is_deterministic():
    _, _, _, _, front_gray, back_gray, fz, bz = synthetic_pair()
    first = review(front_gray, back_gray, fz, bz, threshold=128)
    second = review(front_gray, back_gray, fz, bz, threshold=128)
    assert np.array_equal(first.image, second.image)
    assert first.feasible_coefficients == second.feasible_coefficients


def test_shape_mismatch_is_rejected():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 11), dtype=np.uint8)
    mask_a = np.zeros((10, 10), bool)
    mask_b = np.zeros((10, 11), bool)
    mask_a[0, 0] = mask_b[0, 0] = True
    with pytest.raises(ValueError):
        review(a, b, mask_a, mask_b, threshold=128)


def test_blank_sheet_is_feasible_with_zero_coefficients():
    blank = np.full((15, 15), 255, dtype=np.uint8)  # depth 0 everywhere
    fz = np.zeros((15, 15), bool)
    bz = np.zeros((15, 15), bool)
    fz[0, 0] = bz[0, 1] = True
    result = review(blank, blank, fz, bz, threshold=1)
    assert result.feasible
    assert (0, 0) in result.feasible_coefficients
    assert (result.image == 255).all()
