#!/usr/bin/env python3
"""Classify a leafy vegetable by its segmented visible area."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def analyze(image: np.ndarray, reference_pixels: float | None, reference_cm: float,
            threshold_cm2: float, lower_hsv: tuple[int, int, int],
            upper_hsv: tuple[int, int, int]) -> dict:
    """Segment the largest green object and return its measurements and mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8),
                       np.array(upper_hsv, dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area_px": 0.0, "area_cm2": None, "label": "Not detected",
                "contour": None, "mask": mask}

    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    area_cm2 = None
    label = "Uncalibrated"
    if reference_pixels is not None:
        if reference_pixels <= 0 or reference_cm <= 0:
            raise ValueError("Reference dimensions must be positive.")
        pixels_per_cm = reference_pixels / reference_cm
        area_cm2 = area_px / (pixels_per_cm ** 2)
        label = "Large" if area_cm2 >= threshold_cm2 else "Small"
    return {"area_px": area_px, "area_cm2": area_cm2, "label": label,
            "contour": contour, "mask": mask}


def annotate(image: np.ndarray, result: dict, threshold_cm2: float) -> np.ndarray:
    output = image.copy()
    contour = result["contour"]
    if contour is not None:
        cv2.drawContours(output, [contour], -1, (0, 220, 0), 2)
        x, y, _, _ = cv2.boundingRect(contour)
        area = result["area_cm2"]
        measurement = f"{area:.2f} cm^2" if area is not None else f"{result['area_px']:.0f} px^2"
        text = f"{result['label']} | {measurement}"
        cv2.putText(output, text, (max(10, x), max(28, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, text, (max(10, x), max(28, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(output, "No green object detected", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    if result["area_cm2"] is not None:
        cv2.putText(output, f"Large threshold: {threshold_cm2:g} cm^2", (15, output.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def parse_hsv(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
        if (len(values) != 3 or not 0 <= values[0] <= 179
                or any(n < 0 or n > 255 for n in values[1:])):
            raise ValueError
        return values  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected H,S,V with H from 0 to 179 and S/V from 0 to 255, e.g. 25,40,30"
        ) from exc


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="Classify one image")
    mode.add_argument("--camera", type=int, metavar="INDEX", help="Classify a live camera stream (usually 0)")
    mode.add_argument("--batch", type=Path, help="Classify all supported images in a folder")
    parser.add_argument("--reference-pixels", type=float,
                        help="Measured pixel width of the reference object in the image")
    parser.add_argument("--reference-cm", type=float, default=2.5,
                        help="Real width of the reference object in cm (default: 2.5)")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="Area at/above which a calibrated leaf is Large, in cm^2 (default: 100)")
    parser.add_argument("--output", type=Path, help="Output image path; batch mode writes results.csv here")
    parser.add_argument("--lower-hsv", type=parse_hsv, default=(25, 40, 30), metavar="H,S,V")
    parser.add_argument("--upper-hsv", type=parse_hsv, default=(95, 255, 255), metavar="H,S,V")
    return parser.parse_args()


def process_image(path: Path, args: argparse.Namespace, output_path: Path | None = None) -> dict | None:
    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return None
    result = analyze(image, args.reference_pixels, args.reference_cm, args.threshold,
                     args.lower_hsv, args.upper_hsv)
    rendered = annotate(image, result, args.threshold)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), rendered):
            print(f"Could not save annotated image: {output_path}", file=sys.stderr)
    area = f"{result['area_cm2']:.2f} cm^2" if result["area_cm2"] is not None else f"{result['area_px']:.0f} px^2"
    print(f"{path.name}: {result['label']} — {area}")
    return result


def run_batch(args: argparse.Namespace) -> int:
    if not args.batch.is_dir():
        print(f"Not a folder: {args.batch}", file=sys.stderr)
        return 1
    out_dir = args.output or args.batch / "classified"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(p for p in args.batch.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS):
        result = process_image(path, args, out_dir / path.name)
        if result is not None:
            rows.append({"filename": path.name, "label": result["label"],
                         "area_cm2": "" if result["area_cm2"] is None else f"{result['area_cm2']:.4f}",
                         "area_px": f"{result['area_px']:.2f}"})
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label", "area_cm2", "area_px"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved annotated images and CSV in {out_dir}")
    return 0


def run_camera(args: argparse.Namespace) -> int:
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a camera frame", file=sys.stderr)
                return 1
            result = analyze(frame, args.reference_pixels, args.reference_cm, args.threshold,
                             args.lower_hsv, args.upper_hsv)
            cv2.imshow("Leafy Vegetable Size Classifier (q to quit)", annotate(frame, result, args.threshold))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = get_args()
    if (not math.isfinite(args.reference_cm) or args.reference_cm <= 0
            or not math.isfinite(args.threshold) or args.threshold < 0
            or (args.reference_pixels is not None
                and (not math.isfinite(args.reference_pixels) or args.reference_pixels <= 0))):
        print("Reference dimensions must be positive finite numbers and threshold must be a non-negative finite number.", file=sys.stderr)
        return 2
    if any(low > high for low, high in zip(args.lower_hsv, args.upper_hsv)):
        print("Each lower HSV value must be less than or equal to its upper value.", file=sys.stderr)
        return 2
    if args.image:
        if not args.image.is_file():
            print(f"Image not found: {args.image}", file=sys.stderr)
            return 1
        output = args.output or args.image.with_name(f"{args.image.stem}_classified{args.image.suffix}")
        return 0 if process_image(args.image, args, output) else 1
    if args.batch:
        return run_batch(args)
    return run_camera(args)


if __name__ == "__main__":
    raise SystemExit(main())
