"""Self-contained synthetic front/back sheet generator for the review demo.

The data is produced by the exact show-through model implemented by the
service, so a correctly working ``/review`` must find the generating
coefficient pair among the feasible ones and confirm every front stroke.
Kept independent of the test suite for the same reason as ``demo_data.py``.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Ground-truth coefficients used to synthesise the scans.
TRUE_KF = 3
TRUE_KB = 4
INK_THRESHOLD = 100
SHEET_SIZE = (120, 160)  # (height, width)


def true_depths(size: tuple[int, int] = SHEET_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Front/back true-depth sheets with a few crisp strokes on each side."""
    h, w = size
    front = np.zeros((h, w), dtype=np.int64)
    back = np.zeros((h, w), dtype=np.int64)
    # Front-only strokes.
    front[20:25, 10:120] = 200
    front[60:100, 80:85] = 150
    # Back-only strokes (these are the lines that show through to the front).
    back[30:90, 40:45] = 180
    back[10:15, 100:150] = 90
    return front, back


def scans(
    kf: int = TRUE_KF,
    kb: int = TRUE_KB,
    size: tuple[int, int] = SHEET_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grayscale (front, back) scans synthesised from the model."""
    front_depth, back_depth = true_depths(size)
    # The two ink-free marks must themselves stay truly ink-free.
    front_depth[0, 0] = 0
    back_depth[0, 1] = 0
    obs_front = np.minimum(255, front_depth + (kf * back_depth) // 10)
    obs_back = np.minimum(255, back_depth + (kb * front_depth) // 10)
    front_gray = (255 - obs_front).astype(np.uint8)
    back_gray = (255 - obs_back).astype(np.uint8)
    return front_gray, back_gray


def review_spec() -> dict:
    return {
        "front_ink_free": [{"x": 0, "y": 0}],
        "back_ink_free": [{"x": 1, "y": 0}],
        "front_ink_threshold": INK_THRESHOLD,
    }


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()
