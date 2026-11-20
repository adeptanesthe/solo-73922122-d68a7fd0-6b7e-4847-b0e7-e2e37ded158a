"""Show-through (bleed-through) reconstruction reviewer.

Thin paper lets ink from the back of a sheet shine through to the front and
vice versa.  This module implements the exact feasibility analysis behind the
``/review`` endpoint.

Model (all quantities are integers in 0..255; depth = 255 - grayscale)
-----------------------------------------------------------------------
For each aligned pixel pair let:

* ``f``  : true front depth,
* ``b``  : true back depth,
* ``F``  : observed front depth (``255 - front_gray``),
* ``B``  : observed back depth  (``255 - back_gray``),
* ``kf`` : the front-side show-through coefficient (one value for the whole
           front image), integer 0..5,
* ``kb`` : the back-side show-through coefficient, integer 0..5.

The side-specific bleed amounts are::

    bleed_f = floor(kf * b / 10)
    bleed_b = floor(kb * f / 10)

and the observations are saturated additions::

    F = min(255, f + bleed_f)
    B = min(255, b + bleed_b)

Confirmed ink-free coordinates pin the true depth of the marked side to zero,
i.e. ``f = 0`` / ``b = 0`` at those pixels.

A *feasible reconstruction* is a choice of one coefficient pair ``(kf, kb)`` and
of one ``(f, b)`` pair per pixel that reproduces both observations everywhere
while honouring the ink-free marks.  Pixels are mutually independent once the
two coefficients are fixed, so feasibility and front-depth extrema can be
decided per pixel and intersected across all pixels.

Exact, dependency-light implementation
--------------------------------------
For fixed ``(kf, kb)`` there are only 256x256 = 65 536 possible true-depth
pairs, and each one maps deterministically to one observation pair.  We build,
once per process and cache for reuse, lookup tables indexed by ``(F, B)`` that
record for every ink-free constraint combination:

* whether any admissible ``(f, b)`` pair reproduces the observation,
* the minimum / maximum achievable front depth ``f``.

A 1024x1024 request then costs 36 coefficient pairs times a handful of
vectorised table lookups per pixel — no per-pixel search and no SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEPTH_MAX = 255
COEFF_MIN = 0
COEFF_MAX = 5
GRAY_UNCERTAIN = 127
# Four constraint modes: neither side pinned / front pinned (f=0) /
# back pinned (b=0) / both sides pinned.
N_CONSTRAINT_MODES = 4
# int16 sentinel for "no admissible (f, b) pair" in the extrema tables.
NO_SOLUTION = -1
# Neutral extrema used while aggregating: wider than any real depth (0..255).
_MIN_NEUTRAL = DEPTH_MAX + 1
_MAX_NEUTRAL = -1


@dataclass(frozen=True)
class ReviewTables:
    """Cached feasibility/extrema tables for one coefficient pair.

    Every array has shape (256, 256) indexed by observed depths (F, B).
    Constraint-mode index: bit 0 = front pinned to 0 (f=0),
    bit 1 = back pinned to 0 (b=0).
    """

    kf: int
    kb: int
    # bool: at least one admissible (f, b) pair exists
    feasible: np.ndarray  # shape (4, 256, 256)
    # int16: minimum / maximum front depth f among admissible pairs
    f_min: np.ndarray  # shape (4, 256, 256), NO_SOLUTION when infeasible
    f_max: np.ndarray  # shape (4, 256, 256), NO_SOLUTION when infeasible


def _build_tables(kf: int, kb: int) -> ReviewTables:
    """Build the four constraint-mode tables for one coefficient pair."""
    f = np.arange(DEPTH_MAX + 1, dtype=np.int16)
    b = np.arange(DEPTH_MAX + 1, dtype=np.int16)
    ff, bb = np.meshgrid(f, b, indexing="ij")  # each shape (256, 256)

    bleed_f = (kf * bb) // 10
    bleed_b = (kb * ff) // 10
    obs_f = np.minimum(DEPTH_MAX, ff + bleed_f).astype(np.int16)
    obs_b = np.minimum(DEPTH_MAX, bb + bleed_b).astype(np.int16)

    feasible = np.zeros((N_CONSTRAINT_MODES, DEPTH_MAX + 1, DEPTH_MAX + 1), dtype=bool)
    f_min = np.empty(
        (N_CONSTRAINT_MODES, DEPTH_MAX + 1, DEPTH_MAX + 1), dtype=np.int16
    )
    f_max = np.empty(
        (N_CONSTRAINT_MODES, DEPTH_MAX + 1, DEPTH_MAX + 1), dtype=np.int16
    )

    # mode bit 0 -> f == 0, mode bit 1 -> b == 0
    for mode in range(N_CONSTRAINT_MODES):
        keep = np.ones_like(ff, dtype=bool)
        if mode & 1:
            keep &= ff == 0
        if mode & 2:
            keep &= bb == 0
        lo = np.full((DEPTH_MAX + 1, DEPTH_MAX + 1), _MIN_NEUTRAL, dtype=np.int16)
        hi = np.full((DEPTH_MAX + 1, DEPTH_MAX + 1), _MAX_NEUTRAL, dtype=np.int16)
        if keep.any():
            idx = (obs_f[keep], obs_b[keep])
            feasible[mode][idx] = True
            # Unsorted ufunc aggregation: repeated observation indices combine
            # all admissible (f, b) pairs instead of overwriting one another.
            np.minimum.at(lo, idx, ff[keep])
            np.maximum.at(hi, idx, ff[keep])
        # Infeasible cells get the sentinel; feasible ones keep real extrema.
        f_min[mode] = np.where(feasible[mode], lo, NO_SOLUTION)
        f_max[mode] = np.where(feasible[mode], hi, NO_SOLUTION)

    return ReviewTables(kf=kf, kb=kb, feasible=feasible, f_min=f_min, f_max=f_max)


# Process-wide cache: at most 36 table sets (each ~1.3 MiB, ~47 MiB once the
# full coefficient space has been exercised), reused by every request.
_TABLE_CACHE: dict[tuple[int, int], ReviewTables] = {}


def tables_for(kf: int, kb: int) -> ReviewTables:
    key = (int(kf), int(kb))
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        cached = _build_tables(*key)
        _TABLE_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class ReviewResult:
    """Outcome of a review request."""

    feasible: bool
    # uint8 grayscale image, same size as the front input; None when infeasible
    image: np.ndarray | None = None
    feasible_coefficients: tuple[tuple[int, int], ...] = ()
    # How many pixels could not be explained per coefficient pair (diagnostics).
    infeasible_pixel_counts: dict[tuple[int, int], int] | None = None


def _constraint_modes(
    shape: tuple[int, int],
    front_zero: np.ndarray,
    back_zero: np.ndarray,
) -> np.ndarray:
    """Per-pixel constraint-mode index (0..3) from the ink-free masks."""
    modes = np.zeros(shape, dtype=np.uint8)
    modes[front_zero] |= 1
    modes[back_zero] |= 2
    return modes


def review(
    front_gray: np.ndarray,
    back_gray: np.ndarray,
    front_zero: np.ndarray,
    back_zero: np.ndarray,
    threshold: int,
) -> ReviewResult:
    """Run the exact show-through feasibility analysis.

    Parameters
    ----------
    front_gray, back_gray:
        uint8 arrays of identical shape; grayscale values of the aligned
        front/back scans.
    front_zero, back_zero:
        boolean arrays of the same shape; True marks confirmed ink-free
        coordinates of that side.
    threshold:
        integer 1..255; front depth ``f >= threshold`` counts as "confirmed ink".
    """
    if front_gray.shape != back_gray.shape:
        raise ValueError("正反面图像尺寸必须相同")
    if front_zero.shape != front_gray.shape or back_zero.shape != front_gray.shape:
        raise ValueError("无墨标记与图像尺寸不一致")

    obs_f = (DEPTH_MAX - front_gray).astype(np.int16)
    obs_b = (DEPTH_MAX - back_gray).astype(np.int16)
    modes = _constraint_modes(front_gray.shape, front_zero, back_zero)
    n_pixels = front_gray.size

    # Global extrema of the feasible front depth across every feasible
    # coefficient pair.  Initialised to the widest possible interval.
    global_min = np.full(front_gray.shape, DEPTH_MAX + 1, dtype=np.int16)
    global_max = np.full(front_gray.shape, -1, dtype=np.int16)
    feasible_pairs: list[tuple[int, int]] = []
    bad_counts: dict[tuple[int, int], int] = {}

    for kf in range(COEFF_MIN, COEFF_MAX + 1):
        for kb in range(COEFF_MIN, COEFF_MAX + 1):
            tables = tables_for(kf, kb)
            ok = tables.feasible[modes, obs_f, obs_b]
            n_bad = int(n_pixels - ok.sum())
            if n_bad:
                bad_counts[(kf, kb)] = n_bad
                continue
            feasible_pairs.append((kf, kb))
            lo = tables.f_min[modes, obs_f, obs_b]
            hi = tables.f_max[modes, obs_f, obs_b]
            np.minimum(global_min, lo, out=global_min)
            np.maximum(global_max, hi, out=global_max)

    if not feasible_pairs:
        return ReviewResult(
            feasible=False,
            image=None,
            feasible_coefficients=(),
            infeasible_pixel_counts=bad_counts,
        )

    # Output grayscale mapping (depth -> grayscale):
    #   every feasible reconstruction has f >= threshold  -> black (0)
    #   every feasible reconstruction has f <  threshold  -> white (255)
    #   otherwise (threshold attainable both below and at/above) -> gray (127)
    out = np.full(front_gray.shape, GRAY_UNCERTAIN, dtype=np.uint8)
    out[global_max < threshold] = 255
    out[global_min >= threshold] = 0

    return ReviewResult(
        feasible=True,
        image=out,
        feasible_coefficients=tuple(feasible_pairs),
        infeasible_pixel_counts=None,
    )
