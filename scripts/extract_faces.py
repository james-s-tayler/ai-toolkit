#!/usr/bin/env python3
"""
Face extraction script.

Detects faces using InsightFace (RetinaFace ``det_10g`` from the ``buffalo_l``
pack), crops a square padded around each face, and resizes the crop to the
nearest of a set of target resolutions (default 512/768/1024/1280). Crops are
written to a destination directory.

Output filename pattern: ``<basename>_face_<index>.jpg``

The first run downloads the ``buffalo_l`` model pack to ``~/.insightface/models``.

Usage:
    python scripts/extract_faces.py --img_path /path/to/image.jpg \
        --output_dir /path/to/dataset \
        --padding 1.5 \
        --threshold 0.5 \
        --targets 512,768,1024,1280
"""

import argparse
import json
import os
import sys


DEFAULT_TARGETS = (512, 768, 1024, 1280)


def _nearest_target(side: int, targets) -> int:
    return int(min(targets, key=lambda t: abs(int(t) - side)))


def _square_crop(img, x1: int, y1: int, x2: int, y2: int, padding: float):
    """Return a square crop centered on the bbox, clamped to image bounds."""
    H, W = img.shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = max(bw, bh) * float(padding)
    side = min(side, float(min(H, W)))
    half = side / 2.0
    x0 = cx - half
    y0 = cy - half
    if x0 < 0:
        x0 = 0.0
    elif x0 + side > W:
        x0 = W - side
    if y0 < 0:
        y0 = 0.0
    elif y0 + side > H:
        y0 = H - side
    x0_i = int(round(x0))
    y0_i = int(round(y0))
    side_i = int(round(side))
    return img[y0_i:y0_i + side_i, x0_i:x0_i + side_i]


_app = None  # cached across calls within a single process


def _get_app(det_size: int = 640):
    global _app
    if _app is not None:
        return _app

    # Silence onnxruntime info logs unless something genuinely fails.
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

    import onnxruntime as ort  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        ctx_id = 0
        ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        ctx_id = -1
        ort_providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=ort_providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    _app = app
    return app


def extract_faces(
    img_path: str,
    output_dir: str,
    padding: float = 1.5,
    targets=DEFAULT_TARGETS,
    threshold: float = 0.5,
    det_size: int = 640,
):
    import cv2  # type: ignore

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not decode image: {img_path}")

    app = _get_app(det_size=det_size)
    detections = app.get(img)

    # Filter by confidence threshold and order high → low.
    filtered = [
        f for f in detections if float(getattr(f, "det_score", 1.0)) >= float(threshold)
    ]
    filtered.sort(key=lambda f: float(getattr(f, "det_score", 1.0)), reverse=True)

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(img_path))[0]
    saved = []
    for i, face in enumerate(filtered):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        crop = _square_crop(img, x1, y1, x2, y2, padding)
        if crop.size == 0:
            continue
        side = min(crop.shape[0], crop.shape[1])
        target = _nearest_target(side, targets)
        interp = cv2.INTER_AREA if target < side else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (target, target), interpolation=interp)
        out_path = os.path.join(output_dir, f"{base}_face_{i + 1}.jpg")
        cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved.append({
            "path": out_path,
            "size": target,
            "score": float(getattr(face, "det_score", 1.0)),
        })

    return saved


def _parse_targets(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    nums = [int(p) for p in parts]
    if not nums:
        raise argparse.ArgumentTypeError("targets must be a comma-separated list of ints")
    return tuple(nums)


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from an image using InsightFace.")
    parser.add_argument("--img_path", required=True, help="Path to the source image")
    parser.add_argument("--output_dir", required=True, help="Directory to write face crops into")
    parser.add_argument("--padding", type=float, default=1.5, help="Padding ratio around the face bbox (default 1.5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum detection confidence (default 0.5)")
    parser.add_argument("--det_size", type=int, default=640, help="Detector input size (default 640)")
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=DEFAULT_TARGETS,
        help="Comma-separated target resolutions (default 512,768,1024,1280)",
    )
    args = parser.parse_args()

    try:
        saved = extract_faces(
            args.img_path,
            args.output_dir,
            padding=args.padding,
            targets=args.targets,
            threshold=args.threshold,
            det_size=args.det_size,
        )
        print(json.dumps({"faces": saved}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
