"""FastAPI application for tile alignment and show-through review."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import ValidationError

from . import __version__
from .imaging import DecodedImage, ImageDecodeError, decode_grayscale_png
from .models import (
    MAX_IMAGES,
    MAX_PAIRS,
    MIN_IMAGES,
    AlignResponse,
    AlignSpec,
    ReviewSpec,
)
from .review_service import ReviewError, run_review
from .service import align

DESCRIPTION = """
把一张大幅图纸切块扫描后，用**已声明的重叠图对**与各自的整数平移搜索窗，
确定各块在整张图上的位置。

### 输入
* `images`：3～5 张 **8 位灰度 PNG**（颜色类型 0、非隔行、每边 ≤ 1024 像素）。
  上传顺序即图号 0..n-1，图 0 为定位基准。
* `spec`：JSON 字符串，见下方 schema。包含
  - `pairs`：哪些图对重叠。每对给出 `a`、`b` 以及 `window`
    （`dx_min/dx_max/dy_min/dy_max`，闭区间，至多 81 个整数位置）或
    `candidates`（至多 81 个显式整数平移）；
  - `min_overlap_pixels`：整组共用的最少重叠像素数；
  - `max_mean_abs_diff`：整组共用的平均绝对灰度差上限。

### 坐标约定
第 i 张图左上角在图面上的坐标为 `(x_i, y_i)`；图对 `(a, b)` 的平移 `(dx, dy)`
表示 **b 的左上角位于 a 的坐标 `(dx, dy)`** 处，即
`(x_b, y_b) = (x_a + dx, y_a + dy)`。返回坐标以首图左上角为 `(0, 0)`。

### 结果
* `aligned`：所有图对同时达标且相对布局唯一，返回各图位置与拼接外框。
* `disconnected`：有图块无法经指定图对连到首图；`rescan_image_id` 指出建议补扫的一块。
* `no_solution`：无法同时对齐；`failing_pairs` 给出未达标图对及原因。
* `ambiguous`：多组位置方案都达标；`uncertain_image_ids` 指出位置不确定的图，
  `alternative_layouts` 给出至少两组示例。

### 示例 spec
```json
{
  "pairs": [
    {"a": 0, "b": 1, "window": {"dx_min": 446, "dx_max": 454, "dy_min": -2, "dy_max": 2}},
    {"a": 1, "b": 2, "window": {"dx_min": -454, "dx_max": -446, "dy_min": 448, "dy_max": 452}}
  ],
  "min_overlap_pixels": 20000,
  "max_mean_abs_diff": 3.0
}
```
在上方 `/docs` 页面点击 **Try it out**，先添加 3 张图，再把该 JSON（或 `scripts/make_demo.py`
生成的 `demo/spec.json`）粘贴到 `spec` 字段即可试用。窗口面积两类写法都受 81 个位置的上限约束：
矩形窗口按 `(dx_max-dx_min+1)*(dy_max-dy_min+1) ≤ 81` 计，`candidates` 按条数计。

## 透印复核 `/review`

薄纸图纸背面的线条会透到正面。上传**已镜像对齐**的正/背面 8 位灰度 PNG（同尺寸、
每边 ≤ 1024 px），以及两面各自至少一个**确认无墨**的像素坐标和正面墨迹深度阈值
（1～255 的整数）。深度定义为 `255 - 灰度`，取 0～255 的整数。

透印模型：每面各有一个全图共用的透印系数 `kf` / `kb`（整数 0～5）。对每个对齐像素，
设正面真实深度 `f`、背面真实深度 `b`，则

* 正面观测深度 `F = min(255, f + floor(kf*b/10))`
* 背面观测深度 `B = min(255, b + floor(kb*f/10))`

无墨坐标把标记一面在该像素的真实深度钉为 0。服务枚举全部 36 个系数对，只保留能在
**每个像素**同时复现两面观测且满足无墨标记的复原。返回与正面同尺寸的灰度 PNG：

* **黑 (0)**：在所有可行复原中正面真实深度都达到阈值（确认有墨）；
* **白 (255)**：在所有可行复原中都低于阈值（确认无墨）；
* **灰 (127)**：结论随可行复原变化，交档案员查原件。

不存在任何可行复原时返回 422，只提示扫描图与透印模型不符（不返回图像）。
复核元数据（可行系数对、黑/白/灰像素计数）在响应头 `X-Review-Meta`（JSON）中。
"""


app = FastAPI(
    title="图纸分块定位与透印复核 API",
    version=__version__,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "档案数字化"},
)


@app.get("/health", tags=["meta"], summary="存活检查")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _parse_spec(spec_text: str, n_images: int) -> AlignSpec:
    try:
        raw = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec 不是合法 JSON：{exc}") from exc
    try:
        parsed = AlignSpec.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json(include_url=False, include_context=False)),
        ) from exc
    if len(parsed.pairs) > MAX_PAIRS:
        raise HTTPException(status_code=422, detail=f"图对数量至多 {MAX_PAIRS}")
    for pid, pair in enumerate(parsed.pairs):
        if pair.a >= n_images or pair.b >= n_images:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"图对 {pid} 引用了不存在的图号 {max(pair.a, pair.b)}；"
                    f"本次上传了 {n_images} 张图（图号 0..{n_images - 1}）"
                ),
            )
    return parsed


MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # per tile; a 1024x1024 gray PNG is far below this


def _read_upload(upload: UploadFile, idx: int) -> bytes:
    """Read one upload with a hard size cap so a huge file cannot exhaust RAM."""
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"图 {idx} 超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB 上限",
        )
    return data


@app.post(
    "/align",
    response_model=AlignResponse,
    tags=["alignment"],
    summary="对齐分块扫描图并返回拼接坐标",
    response_description="对齐结果；HTTP 200 时结果的业务状态见 `status` 字段",
    responses={
        400: {"description": "请求格式或图像不符合要求"},
        422: {"description": "spec 结构或图号引用不合法"},
    },
)
def align_endpoint(
    images: Annotated[list[UploadFile], File(description="3～5 张 8 位灰度 PNG，按图号顺序上传")],
    spec: Annotated[str, Form(description="AlignSpec JSON 字符串（见接口说明）")],
) -> AlignResponse:
    if not (MIN_IMAGES <= len(images) <= MAX_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=f"需要 {MIN_IMAGES}～{MAX_IMAGES} 张图，收到 {len(images)} 张",
        )

    parsed = _parse_spec(spec, len(images))

    decoded = []
    for idx, upload in enumerate(images):
        ctype = (upload.content_type or "").split(";")[0].strip().lower()
        if ctype not in ("image/png", "application/octet-stream", ""):
            raise HTTPException(
                status_code=400,
                detail=f"图 {idx} 的 Content-Type 为 {upload.content_type!r}，只接受 image/png",
            )
        data = _read_upload(upload, idx)
        try:
            decoded.append(decode_grayscale_png(idx, data))
        except ImageDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return align(decoded, parsed)


def _parse_review_spec(spec_text: str) -> ReviewSpec:
    try:
        raw = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec 不是合法 JSON：{exc}") from exc
    try:
        return ReviewSpec.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json(include_url=False, include_context=False)),
        ) from exc


def _decode_review_image(upload: UploadFile, side: str) -> DecodedImage:
    ctype = (upload.content_type or "").split(";")[0].strip().lower()
    if ctype not in ("image/png", "application/octet-stream", ""):
        raise HTTPException(
            status_code=400,
            detail=f"{side}的 Content-Type 为 {upload.content_type!r}，只接受 image/png",
        )
    data = _read_upload(upload, side)
    try:
        return decode_grayscale_png(0 if side == "正面" else 1, data)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/review",
    tags=["show-through"],
    summary="透印复核：区分正面真墨迹与背面透印，返回三值标注 PNG",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "复核标注图（与正面同尺寸的 8 位灰度 PNG）；元数据见响应头 X-Review-Meta",
        },
        400: {"description": "请求格式或图像不符合要求"},
        422: {"description": "spec 结构不合法，或无可行复原（扫描图与透印模型不符）"},
    },
)
def review_endpoint(
    front: Annotated[UploadFile, File(description="正面 8 位灰度 PNG（已与背面对齐）")],
    back: Annotated[UploadFile, File(description="背面 8 位灰度 PNG（已镜像、与正面对齐）")],
    spec: Annotated[str, Form(description="ReviewSpec JSON 字符串（见接口说明）")],
) -> Response:
    parsed = _parse_review_spec(spec)
    front_img = _decode_review_image(front, "正面")
    back_img = _decode_review_image(back, "背面")

    try:
        png, meta = run_review(front_img, back_img, parsed)
    except ReviewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # HTTP headers must be latin-1; ensure_ascii keeps the metadata pure ASCII.
    header = json.dumps(meta.model_dump(), ensure_ascii=True)
    if meta.status != "reviewed":
        # No feasible reconstruction: only signal the model mismatch, no image.
        raise HTTPException(
            status_code=422,
            detail="无可行复原：扫描图与透印模型不符（不存在能同时复现两面全部观测"
            "并满足无墨标记的透印系数与真实深度组合）",
            headers={"X-Review-Meta": header},
        )

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Review-Meta": header,
            "Content-Disposition": 'inline; filename="review.png"',
        },
    )
