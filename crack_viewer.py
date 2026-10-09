"""Interactive crack viewer: tune the crack detection with sliders and see the result live.

Usage:
    python crack_viewer.py demo/sample_01.png
    python crack_viewer.py my_image.tif --scale 0.5 --um-per-px 0.217 --params my_settings.json

Sliders (window "Controls"):
    CLAHE x10        local contrast enhancement (0 = off)
    Method           0 = black-hat, 1 = Frangi line filter
    Line Strength    gain on the line response (20 = x1)
    Threshold        0 = show the line response only; >0 = segment cracks
    Gap Bridge       joins small gaps between crack pieces
    Min Length       removes crack pieces shorter than this (pixels)
    View             0 = original, 1 = response, 2 = mask, 3 = cracks on image
    Line Width       display thickness of the drawn cracks

Keys:
    s   save the current result (overlay, skeleton, settings, measurements) to results/
    r   reset all sliders to the default settings
    q   quit (Esc also works)
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from crack_pipeline import METHODS, CrackPipeline, Params, load_gray, overlay

CTRL, VIEW = "Controls", "Crack Viewer"
VIEW_NAMES = ("Original", "Line response", "Mask", "Cracks on image")
SLIDERS = [  # (label, Params field or None, max)
    ("CLAHE x10", "clahe", 80),
    ("Method", "method", 1),
    ("Line Strength", "gain", 100),
    ("Threshold", "threshold", 255),
    ("Gap Bridge", "bridge", 10),
    ("Min Length", "min_length", 400),
    ("View", None, 3),
    ("Line Width", None, 4),
]


def create_controls(p: Params, view=3, width=1):
    cv2.namedWindow(CTRL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(VIEW, cv2.WINDOW_NORMAL)
    defaults = {**asdict(p), "View": view, "Line Width": width}
    for label, field, maximum in SLIDERS:
        value = defaults[field] if field else defaults[label]
        cv2.createTrackbar(label, CTRL, int(value), maximum, lambda _: None)


def set_controls(p: Params, view=3, width=1):
    values = {**asdict(p), "View": view, "Line Width": width}
    for label, field, _ in SLIDERS:
        cv2.setTrackbarPos(label, CTRL, int(values[field] if field else values[label]))


def read_controls(scale):
    vals = {label: cv2.getTrackbarPos(label, CTRL) for label, _, _ in SLIDERS}
    p = Params(scale=scale, **{f: vals[l] for l, f, _ in SLIDERS if f})
    p.gain = max(p.gain, 1)
    return p, vals["View"], max(vals["Line Width"], 1)


def render(pipe, p, view, width, um_per_px=None):
    """Build the image shown in the viewer window, with a status line."""
    res = pipe.run(p, um_per_px)
    if view == 0:
        img = cv2.cvtColor(pipe.gray, cv2.COLOR_GRAY2BGR)
    elif view == 1 or res["skeleton"] is None:          # no segmentation yet: show the response
        img = cv2.cvtColor(res["response"], cv2.COLOR_GRAY2BGR)
    elif view == 2:
        img = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2BGR)
    else:
        img = overlay(res["enhanced"], res["skeleton"], thickness=width)

    status = f"{VIEW_NAMES[view]} | {METHODS[p.method]}"
    if res["stats"]:
        s = res["stats"]
        if "length_um" in s:
            status += f" | cracks: {s['cracks']}  length: {s['length_um']:.0f} um  density: {s['density_per_um']:.4f} um/um2"
        else:
            status += f" | cracks: {s['cracks']}  length: {s['length_px']:.0f} px"
    elif p.threshold == 0:
        status += " | set Threshold > 0 to segment cracks"
    bar = np.full((28, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar]), res


def save_result(image_path, pipe, p, res, width, out_dir="results"):
    out = Path(out_dir); out.mkdir(exist_ok=True)
    stem = Path(image_path).stem
    p.to_json(out / f"{stem}_params.json")
    if res["skeleton"] is not None:
        cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay(res["enhanced"], res["skeleton"], width))
        cv2.imwrite(str(out / f"{stem}_skeleton.png"), res["skeleton"])
        cv2.imwrite(str(out / f"{stem}_mask.png"), res["mask"])
        with open(out / f"{stem}_measurements.json", "w") as f:
            json.dump(res["stats"], f, indent=2)
    print(f"Saved results for {stem} to {out}/")


def main():
    ap = argparse.ArgumentParser(description="Interactive crack detection viewer")
    ap.add_argument("image", help="path to the microscope image")
    ap.add_argument("--scale", type=float, default=0.5, help="resize factor for speed (default 0.5)")
    ap.add_argument("--um-per-px", type=float, default=None, help="micrometres per pixel of the original image")
    ap.add_argument("--params", default=None, help="start from settings saved earlier (JSON)")
    args = ap.parse_args()

    start = Params.from_json(args.params) if args.params else Params()
    start.scale = args.scale
    pipe = CrackPipeline(load_gray(args.image), scale=args.scale)
    create_controls(start)

    print(__doc__.split("Keys:")[1])
    while True:
        p, view, width = read_controls(args.scale)
        frame, res = render(pipe, p, view, width, args.um_per_px)
        cv2.imshow(VIEW, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            save_result(args.image, pipe, p, res, width)
        if key == ord("r"):
            set_controls(Params(scale=args.scale))
        if cv2.getWindowProperty(VIEW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
