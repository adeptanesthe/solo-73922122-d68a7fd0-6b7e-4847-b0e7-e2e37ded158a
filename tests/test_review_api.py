"""End-to-end tests for the show-through review HTTP API."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from tests.fixtures import (
    png_bytes,
    png_bytes_16bit,
    png_bytes_gray_alpha,
    png_bytes_palette,
    png_bytes_rgb,
)

client = TestClient(app)


def make_sheet(
    seed: int = 0,
    size: tuple[int, int] = (60, 80),
    kf: int = 2,
    kb: int = 3,
) -> dict:
    """Build a synthetic aligned front/back pair with a known ground truth."""
    rng = np.random.default_rng(seed)
    h, w = size
    f = rng.integers(0, 256, size=(h, w)).astype(np.int64)
    b = rng.integers(0, 256, size=(h, w)).astype(np.int64)
    f[0, 0] = 0
    b[0, 1] = 0
    obs_f = np.minimum(255, f + (kf * b) // 10).astype(np.uint8)
    obs_b = np.minimum(255, b + (kb * f) // 10).astype(np.uint8)
    return {
        "front": (255 - obs_f).astype(np.uint8),
        "back": (255 - obs_b).astype(np.uint8),
        "spec": {
            "front_ink_free": [{"x": 0, "y": 0}],
            "back_ink_free": [{"x": 1, "y": 0}],
            "front_ink_threshold": 128,
        },
    }


def post_review(front: np.ndarray, back: np.ndarray, spec: dict, *, front_ct="image/png"):
    files = [
        ("front", ("front.png", png_bytes(front), front_ct)),
        ("back", ("back.png", png_bytes(back), "image/png")),
    ]
    return client.post("/review", files=files, data={"spec": json.dumps(spec)})


def decode_png(content: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(content)), dtype=np.uint8)


def test_docs_advertise_review_endpoint():
    schema = client.get("/openapi.json").json()
    assert "/review" in schema["paths"]


def test_review_returns_png_with_three_value_palette():
    sheet = make_sheet()
    r = post_review(sheet["front"], sheet["back"], sheet["spec"])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    meta = json.loads(r.headers["x-review-meta"])
    assert meta["status"] == "reviewed"
    assert (2, 3) in [tuple(pair) for pair in meta["feasible_coefficients"]]
    image = decode_png(r.content)
    assert image.shape == sheet["front"].shape
    assert set(np.unique(image)).issubset({0, 127, 255})
    # Pixel counts in the header match the returned image.
    assert meta["ink_pixels"] == int((image == 0).sum())
    assert meta["blank_pixels"] == int((image == 255).sum())
    assert meta["uncertain_pixels"] == int((image == 127).sum())
    assert meta["ink_pixels"] + meta["blank_pixels"] + meta["uncertain_pixels"] == image.size


def test_review_classification_is_stable_across_requests():
    sheet = make_sheet(seed=5)
    r1 = post_review(sheet["front"], sheet["back"], sheet["spec"])
    r2 = post_review(sheet["front"], sheet["back"], sheet["spec"])
    assert r1.content == r2.content


def test_front_ink_free_pixel_is_white_in_output():
    sheet = make_sheet()
    r = post_review(sheet["front"], sheet["back"], sheet["spec"])
    image = decode_png(r.content)
    assert image[0, 0] == 255


def test_infeasible_request_returns_422_without_image():
    sheet = make_sheet()
    front = sheet["front"].copy()
    front[0, 0] = 0  # depth 255 at a front ink-free pixel: impossible for kf <= 5
    r = post_review(front, sheet["back"], sheet["spec"])
    assert r.status_code == 422
    body = r.json()
    assert "透印模型不符" in body["detail"]
    meta = json.loads(r.headers["x-review-meta"])
    assert meta["status"] == "infeasible"
    assert meta["feasible_coefficients"] == []


def test_rejects_size_mismatch():
    sheet = make_sheet()
    back = sheet["back"][:, : -1]
    r = post_review(sheet["front"], back, sheet["spec"])
    assert r.status_code == 400
    assert "尺寸" in r.json()["detail"]


def test_rejects_coordinate_outside_image():
    sheet = make_sheet(size=(40, 50))  # height=40, width=50
    spec = {
        "front_ink_free": [{"x": 50, "y": 0}],  # x == width
        "back_ink_free": [{"x": 0, "y": 40}],  # y == height
        "front_ink_threshold": 128,
    }
    r = post_review(sheet["front"], sheet["back"], spec)
    assert r.status_code == 400
    assert "超出图像范围" in r.json()["detail"]


@pytest.mark.parametrize(
    "spec, fragment",
    [
        ({"back_ink_free": [{"x": 0, "y": 0}], "front_ink_threshold": 128}, "front_ink_free"),
        ({"front_ink_free": [{"x": 0, "y": 0}], "front_ink_threshold": 128}, "back_ink_free"),
        (
            {
                "front_ink_free": [{"x": 0, "y": 0}],
                "back_ink_free": [{"x": 0, "y": 0}],
                "front_ink_threshold": 0,
            },
            "front_ink_threshold",
        ),
        (
            {
                "front_ink_free": [{"x": 0, "y": 0}],
                "back_ink_free": [{"x": 0, "y": 0}],
                "front_ink_threshold": 256,
            },
            "front_ink_threshold",
        ),
        (
            {
                "front_ink_free": [{"x": -1, "y": 0}],
                "back_ink_free": [{"x": 0, "y": 0}],
                "front_ink_threshold": 128,
            },
            "x",
        ),
        (
            {
                "front_ink_free": [{"x": 0, "y": 0}],
                "back_ink_free": [{"x": 0, "y": 0}],
                "front_ink_threshold": 128,
                "unexpected": 1,
            },
            "unexpected",
        ),
    ],
)
def test_rejects_invalid_spec(spec, fragment):
    sheet = make_sheet()
    r = post_review(sheet["front"], sheet["back"], spec)
    assert r.status_code == 422
    assert fragment in r.text


def test_rejects_bad_spec_json():
    sheet = make_sheet()
    files = [
        ("front", ("front.png", png_bytes(sheet["front"]), "image/png")),
        ("back", ("back.png", png_bytes(sheet["back"]), "image/png")),
    ]
    r = client.post("/review", files=files, data={"spec": "{not json"})
    assert r.status_code == 400


def test_rejects_rgb_png():
    sheet = make_sheet()
    files = [
        ("front", ("front.png", png_bytes_rgb(sheet["front"]), "image/png")),
        ("back", ("back.png", png_bytes(sheet["back"]), "image/png")),
    ]
    r = client.post(
        "/review", files=files, data={"spec": json.dumps(sheet["spec"])}
    )
    assert r.status_code == 400
    assert "灰度" in r.json()["detail"] or "颜色类型" in r.json()["detail"]


def test_rejects_16bit_png():
    sheet = make_sheet()
    files = [
        ("front", ("front.png", png_bytes_16bit(sheet["front"]), "image/png")),
        ("back", ("back.png", png_bytes(sheet["back"]), "image/png")),
    ]
    r = client.post(
        "/review", files=files, data={"spec": json.dumps(sheet["spec"])}
    )
    assert r.status_code == 400
    assert "8 位" in r.json()["detail"]


def test_rejects_palette_and_alpha_png():
    sheet = make_sheet()
    for encoder, fragment in [
        (png_bytes_palette, "颜色类型"),
        (png_bytes_gray_alpha, "颜色类型"),
    ]:
        files = [
            ("front", ("front.png", encoder(sheet["front"]), "image/png")),
            ("back", ("back.png", png_bytes(sheet["back"]), "image/png")),
        ]
        r = client.post(
            "/review", files=files, data={"spec": json.dumps(sheet["spec"])}
        )
        assert r.status_code == 400
        assert fragment in r.json()["detail"]


def test_rejects_wrong_content_type():
    sheet = make_sheet()
    r = post_review(sheet["front"], sheet["back"], sheet["spec"], front_ct="image/jpeg")
    assert r.status_code == 400


def test_rejects_oversize_image():
    big = np.zeros((1025, 64), dtype=np.uint8)
    small = np.zeros((1025, 64), dtype=np.uint8)
    spec = {
        "front_ink_free": [{"x": 0, "y": 0}],
        "back_ink_free": [{"x": 1, "y": 0}],
        "front_ink_threshold": 128,
    }
    r = post_review(big, small, spec)
    assert r.status_code == 400
    assert "1024" in r.json()["detail"]


def test_duplicate_ink_free_coordinates_are_allowed():
    # Repeated coordinates simply collapse into one mask; the "at least one"
    # requirement is still satisfied and the request is valid.
    sheet = make_sheet()
    spec = {
        "front_ink_free": [{"x": 0, "y": 0}, {"x": 0, "y": 0}],
        "back_ink_free": [{"x": 1, "y": 0}],
        "front_ink_threshold": 128,
    }
    r = post_review(sheet["front"], sheet["back"], spec)
    assert r.status_code == 200, r.text


def test_threshold_changes_classification_boundary():
    sheet = make_sheet(seed=9)
    low = post_review(
        sheet["front"], sheet["back"], {**sheet["spec"], "front_ink_threshold": 1}
    )
    high = post_review(
        sheet["front"], sheet["back"], {**sheet["spec"], "front_ink_threshold": 255}
    )
    low_img = decode_png(low.content)
    high_img = decode_png(high.content)
    # Raising the threshold can only make the "ink in every reconstruction"
    # (black) set shrink, never move: black(t2) ⊆ black(t1) for t2 > t1.
    assert (high_img == 0).sum() <= (low_img == 0).sum()
    black_low = set(map(tuple, np.argwhere(low_img == 0).tolist()))
    black_high = set(map(tuple, np.argwhere(high_img == 0).tolist()))
    assert black_high.issubset(black_low)


def test_1024_sized_request_is_accepted_and_fast():
    side = 1024
    rng = np.random.default_rng(3)
    kf, kb = 1, 2
    f = rng.integers(0, 256, size=(side, side)).astype(np.int64)
    b = rng.integers(0, 256, size=(side, side)).astype(np.int64)
    f[0, 0] = 0
    b[0, 1] = 0
    obs_f = np.minimum(255, f + (kf * b) // 10).astype(np.uint8)
    obs_b = np.minimum(255, b + (kb * f) // 10).astype(np.uint8)
    front = (255 - obs_f).astype(np.uint8)
    back = (255 - obs_b).astype(np.uint8)
    spec = {
        "front_ink_free": [{"x": 0, "y": 0}],
        "back_ink_free": [{"x": 1, "y": 0}],
        "front_ink_threshold": 128,
    }
    import time

    start = time.perf_counter()
    r = post_review(front, back, spec)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200, r.text
    image = decode_png(r.content)
    assert image.shape == (side, side)
    assert set(np.unique(image)).issubset({0, 127, 255})
    assert elapsed < 30  # full 1024x1024 analysis must stay well within limits
