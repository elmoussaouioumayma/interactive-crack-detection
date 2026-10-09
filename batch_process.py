"""Apply saved settings to a whole folder of images and export the measurements.

Tune the settings once in the viewer (press 's' to save them), then:

    python batch_process.py demo/ --params results/sample_01_params.json
    python batch_process.py my_images/ --params settings.json --um-per-px 0.217 --out batch_results

Writes one overlay image per input image and a CSV with crack count, total length
and crack density for every image. If a ground-truth mask named <image>_truth.png
exists next to an image, precision, recall and F1 against it are added.
"""

import argparse
import csv
from pathlib import Path

import cv2

from crack_pipeline import CrackPipeline, Params, compare_to_truth, load_gray, overlay

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description="Batch crack detection with saved settings")
    ap.add_argument("folder", help="folder with images")
    ap.add_argument("--params", required=True, help="settings JSON saved from the viewer")
    ap.add_argument("--um-per-px", type=float, default=None, help="micrometres per pixel")
    ap.add_argument("--out", default="batch_results", help="output folder")
    args = ap.parse_args()

    p = Params.from_json(args.params)
    if p.threshold <= 0:
        raise SystemExit("Threshold in the settings is 0, so no cracks would be segmented.")
    images = sorted(f for f in Path(args.folder).iterdir()
                    if f.suffix.lower() in EXTENSIONS and "_truth" not in f.stem)
    if not images:
        raise SystemExit(f"No images found in {args.folder}")

    out = Path(args.out); out.mkdir(exist_ok=True)
    rows = []
    for path in images:
        pipe = CrackPipeline(load_gray(path), scale=p.scale)
        res = pipe.run(p, args.um_per_px)
        cv2.imwrite(str(out / f"{path.stem}_overlay.png"), overlay(res["enhanced"], res["skeleton"]))
        row = {"image": path.name, **res["stats"]}
        truth_path = path.with_name(f"{path.stem}_truth.png")       # optional ground-truth mask
        if truth_path.exists():
            row.update(compare_to_truth(res["skeleton"], load_gray(truth_path), p.scale))
        rows.append(row)
        print(f"{path.name}: {row}")

    with open(out / "measurements.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} images processed. Results in {out}/measurements.csv")


if __name__ == "__main__":
    main()
