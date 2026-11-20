"""Generate demo requests for either API.

Alignment demo (default)::

    python scripts/make_demo.py [outdir]

writes ``demo/tileN.png`` (four overlapping crops of one synthetic drawing that
form a 2x2 mosaic), ``demo/spec.json`` and ``demo/request.sh``.  The pair list
includes the redundant cycle pair (0,3), so a unique answer requires the cycle
to agree -- a good end-to-end check of the service.

Show-through review demo::

    python scripts/make_demo.py review [outdir]

writes ``demo/front.png`` and ``demo/back.png`` (aligned grayscale scans
synthesised from the show-through model with known coefficients),
``demo/review_spec.json`` and ``demo/review_request.sh``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_data import OFFSETS, demo_tiles, png_bytes, window  # noqa: E402
from scripts.review_demo import review_spec, scans  # noqa: E402


def make_align_demo(outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    tiles = demo_tiles()
    for idx, tile in enumerate(tiles):
        (outdir / f"tile{idx}.png").write_bytes(png_bytes(tile))

    spec = {
        "pairs": [
            {"a": a, "b": b, "window": window(off, radius=2)} for (a, b), off in OFFSETS.items()
        ],
        "min_overlap_pixels": 20000,
        "max_mean_abs_diff": 3.0,
    }
    (outdir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    cmd = "curl -sS -X POST http://localhost:8000/align \\\n" + "".join(
        f"  -F 'images=@{outdir}/tile{idx}.png;type=image/png' \\\n"
        for idx in range(len(tiles))
    )
    cmd += f"  -F 'spec=<{outdir / 'spec.json'}'"
    (outdir / "request.sh").write_text(cmd + "\n", encoding="utf-8")
    print(f"wrote {len(tiles)} tiles, spec.json and request.sh to {outdir}/")
    return 0


def make_review_demo(outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    front, back = scans()
    (outdir / "front.png").write_bytes(png_bytes(front))
    (outdir / "back.png").write_bytes(png_bytes(back))
    spec = review_spec()
    (outdir / "review_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cmd = (
        f"curl -sS -D {outdir / 'review_headers.txt'} -o {outdir / 'review.png'} "
        "-X POST http://localhost:8000/review \\\n"
        f"  -F 'front=@{outdir / 'front.png'};type=image/png' \\\n"
        f"  -F 'back=@{outdir / 'back.png'};type=image/png' \\\n"
        f"  -F 'spec=<{outdir / 'review_spec.json'}'\n"
    )
    (outdir / "review_request.sh").write_text(cmd, encoding="utf-8")
    print(f"wrote front.png, back.png, review_spec.json and review_request.sh to {outdir}/")
    return 0


def main(argv: list[str]) -> int:
    if argv[1:2] == ["review"]:
        outdir = Path(argv[2]) if len(argv) > 2 else Path("demo")
        rc = make_review_demo(outdir)
    else:
        # Backwards-compatible default: the alignment demo.  An explicit
        # "align" keyword is accepted for symmetry with "review".
        rest = argv[1:]
        if rest[:1] == ["align"]:
            rest = rest[1:]
        outdir = Path(rest[0]) if rest else Path("demo")
        rc = make_align_demo(outdir)

    print("start the service with:  docker compose up --build")
    print("then run:                bash demo/review_request.sh (or demo/request.sh)")
    print("interactive docs:        http://localhost:8000/docs")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
