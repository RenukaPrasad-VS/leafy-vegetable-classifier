#!/usr/bin/env python3
"""Measure and classify leaf area using a YOLO instance-segmentation model."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="Classify one image")
    mode.add_argument("--camera", type=int, metavar="INDEX", help="Classify a live camera stream (usually 0)")
    mode.add_argument("--batch", type=Path, help="Classify supported images in a folder")
    parser.add_argument("--model", default="yolo26n-seg.pt",
                        help="YOLO segmentation weights path or Ultralytics model name (default: yolo26n-seg.pt)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Minimum detection confidence from 0 to 1 (default: 0.25)")
    parser.add_argument("--reference-pixels", type=float,
                        help="Measured pixel width of the reference object in the image")
    parser.add_argument("--reference-cm", type=float, default=2.5,
                        help="Real width of the reference object in cm (default: 2.5)")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="Area at/above which a calibrated leaf is Large, in cm^2 (default: 100)")
    parser.add_argument("--output", type=Path,
                        help="Output image path; batch mode writes annotated images and results.csv here")
    return parser.parse_args()


def segment_largest(model, image: np.ndarray, confidence: float,
                    reference_pixels: float | None, reference_cm: float,
                    threshold_cm2: float) -> dict:
    """Use the largest predicted instance mask and measure its visible area."""
    prediction = model.predict(source=image, conf=confidence, verbose=False)[0]
    height, width = image.shape[:2]
    contour = None
    mask = np.zeros((height, width), dtype=np.uint8)

    if prediction.masks is not None and len(prediction.masks.xy):
        polygons = [np.rint(poly).astype(np.int32) for poly in prediction.masks.xy if len(poly) >= 3]
        if polygons:
            candidates = []
            for polygon in polygons:
                candidate = np.zeros_like(mask)
                cv2.fillPoly(candidate, [polygon], 255)
                candidates.append(candidate)
            mask = max(candidates, key=cv2.countNonZero)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)

    area_px = float(cv2.countNonZero(mask))
    area_cm2 = None
    label = "Not detected" if area_px == 0 else "Uncalibrated"
    if area_px > 0 and reference_pixels is not None:
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
        cv2.putText(output, "No segmented object detected", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    if result["area_cm2"] is not None:
        cv2.putText(output, f"Large threshold: {threshold_cm2:g} cm^2", (15, output.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def process_image(path: Path, args: argparse.Namespace, model,
                  output_path: Path | None = None) -> dict | None:
    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return None
    result = segment_largest(model, image, args.confidence, args.reference_pixels,
                             args.reference_cm, args.threshold)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotate(image, result, args.threshold)):
            print(f"Could not save annotated image: {output_path}", file=sys.stderr)
    area = (f"{result['area_cm2']:.2f} cm^2" if result["area_cm2"] is not None
            else f"{result['area_px']:.0f} px^2")
    print(f"{path.name}: {result['label']} — {area}")
    return result


def run_batch(args: argparse.Namespace, model) -> int:
    if not args.batch.is_dir():
        print(f"Not a folder: {args.batch}", file=sys.stderr)
        return 1
    out_dir = args.output or args.batch / "classified_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = sorted(p for p in args.batch.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    for path in paths:
        result = process_image(path, args, model, out_dir / path.name)
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


def run_camera(args: argparse.Namespace, model) -> int:
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
            result = segment_largest(model, frame, args.confidence, args.reference_pixels,
                                     args.reference_cm, args.threshold)
            cv2.imshow("YOLO Leaf Size Classifier (q to quit)", annotate(frame, result, args.threshold))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = get_args()
    if (not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1
            or not math.isfinite(args.reference_cm) or args.reference_cm <= 0
            or not math.isfinite(args.threshold) or args.threshold < 0
            or (args.reference_pixels is not None
                and (not math.isfinite(args.reference_pixels) or args.reference_pixels <= 0))):
        print("Confidence must be from 0 to 1, reference dimensions must be positive finite numbers, "
              "and threshold must be a non-negative finite number.", file=sys.stderr)
        return 2
    if args.image and not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("YOLO requires Ultralytics. Install dependencies with: python3 -m pip install -r requirements.txt",
              file=sys.stderr)
        return 2

    try:
        model = YOLO(args.model)
    except Exception as exc:
        print(f"Could not load YOLO model '{args.model}': {exc}", file=sys.stderr)
        return 1
    if args.image:
        output = args.output or args.image.with_name(f"{args.image.stem}_yolo{args.image.suffix}")
        return 0 if process_image(args.image, args, model, output) else 1
    if args.batch:
        return run_batch(args, model)
    return run_camera(args, model)


if __name__ == "__main__":
    raise SystemExit(main())
