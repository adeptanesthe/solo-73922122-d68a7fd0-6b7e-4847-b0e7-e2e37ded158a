"""Orchestration for the show-through review endpoint."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .imaging import DecodedImage
from .models import ReviewMeta, ReviewSpec
from .showthrough import GRAY_UNCERTAIN, review


class ReviewError(ValueError):
    """Raised when the review request itself is invalid (HTTP 4xx)."""


def _points_to_mask(
    points: list[tuple[int, int]], height: int, width: int, side: str
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x, y in points:
        if x >= width or y >= height:
            raise ReviewError(
                f"{side}的无墨坐标 ({x}, {y}) 超出图像范围（宽 {width}、高 {height}）"
            )
        mask[y, x] = True
    return mask


def _encode_png(array: np.ndarray) -> bytes:
    """Encode a uint8 grayscale array as a colour-type-0 PNG."""
    buf = io.BytesIO()
    Image.fromarray(array, mode="L").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def run_review(
    front: DecodedImage,
    back: DecodedImage,
    spec: ReviewSpec,
) -> tuple[bytes | None, ReviewMeta]:
    """Validate inputs, run the feasibility analysis and encode the result.

    Returns ``(png_bytes, metadata)``; ``png_bytes`` is ``None`` when no
    feasible reconstruction exists.
    """
    if front.array.shape != back.array.shape:
        raise ReviewError(
            f"正反面图像尺寸必须相同：正面 {front.width}x{front.height}，"
            f"背面 {back.width}x{back.height}"
        )
    height, width = front.array.shape
    front_zero = _points_to_mask(spec.front_points(), height, width, "正面")
    back_zero = _points_to_mask(spec.back_points(), height, width, "背面")

    result = review(
        front_gray=front.array,
        back_gray=back.array,
        front_zero=front_zero,
        back_zero=back_zero,
        threshold=spec.front_ink_threshold,
    )

    if not result.feasible:
        counts = result.infeasible_pixel_counts or {}
        best = min(counts.items(), key=lambda kv: kv[1]) if counts else None
        meta = ReviewMeta(
            status="infeasible",
            # Header JSON is encoded as ASCII; keep this message English so it
            # is safe to carry in X-Review-Meta. The Chinese explanation is
            # returned in the 422 response body by the endpoint.
            message=(
                "no feasible reconstruction: scans do not match the show-through model"
                + (
                    f"; best coefficient pair (kf={best[0][0]}, kb={best[0][1]}) "
                    f"still leaves {best[1]} unexplained pixel(s)"
                    if best is not None
                    else ""
                )
            ),
            width=width,
            height=height,
            front_ink_threshold=spec.front_ink_threshold,
        )
        return None, meta

    assert result.image is not None
    png = _encode_png(result.image)
    ink_pixels = int(np.count_nonzero(result.image == 0))
    blank_pixels = int(np.count_nonzero(result.image == 255))
    uncertain_pixels = int(np.count_nonzero(result.image == GRAY_UNCERTAIN))
    meta = ReviewMeta(
        status="reviewed",
        message=(
            f"review complete: {len(result.feasible_coefficients)} coefficient pair(s) "
            "reproduce all observations; black=front ink confirmed in every feasible "
            "reconstruction, white=no ink confirmed, gray=inconclusive (consult original)"
        ),
        width=width,
        height=height,
        front_ink_threshold=spec.front_ink_threshold,
        feasible_coefficients=list(result.feasible_coefficients),
        ink_pixels=ink_pixels,
        blank_pixels=blank_pixels,
        uncertain_pixels=uncertain_pixels,
    )
    return png, meta
