"""Crack detection pipeline for microscope images.

All processing lives here, separate from the user interface, so the same
settings give the same result in the interactive viewer and in batch mode.

Pipeline:  grayscale -> CLAHE -> line filter (black-hat or Frangi) -> gain
           -> threshold -> gap bridging -> skeleton -> length filter -> measurements
"""

import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize

METHODS = ("Black-hat", "Frangi")


@dataclass
class Params:
    scale: float = 0.5        # resize factor before processing (speed)
    clahe: int = 20           # CLAHE clip limit x10 (0 = off)
    method: int = 1           # 0 = black-hat, 1 = Frangi vesselness filter
    gain: int = 20            # line strength: response x gain/20
    threshold: int = 40       # 0..255 on the line response (0 = show response only)
    bridge: int = 2           # radius of the closing that joins small gaps (0 = off)
    min_length: int = 40      # remove skeleton pieces shorter than this (pixels, processing scale)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def load_gray(path):
    """Read an image as 8-bit grayscale (also handles 16-bit TIFFs)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def resize(img, scale):
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def enhance(gray, clahe):
    """Local contrast enhancement (CLAHE); returns the input if clahe == 0."""
    if clahe <= 0:
        return gray
    return cv2.createCLAHE(clipLimit=clahe / 10.0, tileGridSize=(8, 8)).apply(gray)


def line_response(enhanced, method):
    """Map where dark, thin, line-like structures (cracks) are bright, scaled to 0..255."""
    if method == 1:
        # Frangi filter: responds to tube-like structures; black_ridges -> dark cracks
        resp = frangi(enhanced, sigmas=range(1, 4), black_ridges=True)
    else:
        # Black-hat: closing minus image -> dark features narrower than the kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        resp = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel).astype(float)
    resp = np.nan_to_num(resp)
    top = np.percentile(resp, 99.5)          # robust scaling: a few extreme pixels do not dominate
    return np.clip(resp / (top + 1e-12) * 255, 0, 255).astype(np.uint8)


def apply_gain(response, gain):
    return cv2.convertScaleAbs(response, alpha=gain / 20.0)


def segment(response, threshold, bridge):
    """Binary crack mask from the line response."""
    _, mask = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)
    if bridge > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def skeleton_filtered(mask, min_length):
    """One-pixel skeleton with pieces shorter than min_length removed.

    Length is measured as the number of skeleton pixels of each connected piece,
    which is the length of a 1-pixel-wide line.
    """
    skel = skeletonize(mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(min_length, 1)
    return (keep[labels] * 255).astype(np.uint8), int(keep.sum())


class CrackPipeline:
    """Runs the pipeline on one image and caches the slow steps (Frangi, skeleton)."""

    def __init__(self, gray_full, scale=0.5):
        self.full_shape = gray_full.shape
        self.scale = scale
        self.gray = resize(gray_full, scale)
        self._cache = {}

    def _cached(self, key, fn):
        if key not in self._cache:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = fn()
        return self._cache[key]

    def run(self, p: Params, um_per_px=None):
        """Return a dict with all intermediate images and the measurements."""
        enhanced = self._cached(("enh", p.clahe), lambda: enhance(self.gray, p.clahe))
        response = self._cached(("resp", p.clahe, p.method), lambda: line_response(enhanced, p.method))
        boosted = apply_gain(response, p.gain)
        out = {"enhanced": enhanced, "response": boosted, "mask": None, "skeleton": None,
               "stats": None}
        if p.threshold <= 0:
            return out
        key = ("skel", p.clahe, p.method, p.gain, p.threshold, p.bridge, p.min_length)
        mask, (skel, n_cracks) = self._cached(key, lambda: _segment_and_skeleton(boosted, p))
        out.update(mask=mask, skeleton=skel, stats=measure(skel, n_cracks, self.scale, um_per_px))
        return out


def _segment_and_skeleton(response, p):
    mask = segment(response, p.threshold, p.bridge)
    return mask, skeleton_filtered(mask, p.min_length)


def measure(skeleton, n_cracks, scale, um_per_px=None):
    """Crack count, total length and density, converted back to original image pixels."""
    h, w = skeleton.shape
    length_px = (skeleton > 0).sum() / scale              # length in original-resolution pixels
    area_px = (h / scale) * (w / scale)
    stats = {"cracks": n_cracks, "length_px": round(float(length_px), 1),
             "density_per_px": round(float(length_px / area_px), 6)}
    if um_per_px:
        stats["length_um"] = round(float(length_px * um_per_px), 1)
        stats["density_per_um"] = round(float(length_px * um_per_px / (area_px * um_per_px ** 2)), 6)
    return stats


def overlay(gray, skeleton, thickness=1, color=(0, 0, 255)):
    """Draw the skeleton in color on top of the grayscale image."""
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lines = skeleton
    if thickness > 1:
        lines = cv2.dilate(skeleton, np.ones((thickness, thickness), np.uint8))
    view[lines > 0] = color
    return view


def compare_to_truth(skeleton, truth_full, scale, tolerance=2):
    """Precision and recall of a detected skeleton against a ground-truth crack mask.

    A detected pixel counts as correct if a true crack lies within `tolerance` pixels
    (at processing scale), and a true crack pixel counts as found if a detected
    crack lies within `tolerance` pixels.
    """
    truth = resize(truth_full, scale) > 0
    truth_skel = skeletonize(truth)
    pred = skeleton > 0
    dist_to_truth = cv2.distanceTransform((~truth_skel).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((dist_to_truth[pred] <= tolerance).mean()) if pred.any() else 0.0
    recall = float((dist_to_pred[truth_skel] <= tolerance).mean()) if truth_skel.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
