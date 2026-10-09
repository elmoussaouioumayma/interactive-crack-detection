"""Create synthetic microscope-like images of a cracked coating, with ground truth.

The images imitate the typical difficulties of optical micrographs of cracked
coatings: uneven illumination, grain and nodule texture, noise, dark pits,
polishing scratches, and cracks of different width and contrast (some faint).
The crack network is generated from a Voronoi diagram, which gives polygonal
cells similar to a real microcrack network.

    python make_demo_images.py            # writes demo/sample_01.png ... and *_truth.png
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Voronoi


def crack_network(h, w, n_cells, rng):
    """Draw a Voronoi crack network; returns (darkness map, ground-truth mask)."""
    pts = rng.uniform([-0.1 * w, -0.1 * h], [1.1 * w, 1.1 * h], size=(n_cells, 2))
    vor = Voronoi(pts)
    dark = np.zeros((h, w), np.float32)
    truth = np.zeros((h, w), np.uint8)
    for (a, b) in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        p0, p1 = vor.vertices[a], vor.vertices[b]
        # wiggle the crack with a few jittered midpoints
        n = 6
        t = np.linspace(0, 1, n)[:, None]
        path = p0 + t * (p1 - p0)
        normal = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / (np.linalg.norm(p1 - p0) + 1e-9)
        path[1:-1] += normal * rng.normal(0, 2.5, size=(n - 2, 1))
        path = path.round().astype(np.int32)
        strength = rng.choice([rng.uniform(25, 45), rng.uniform(55, 100)], p=[0.25, 0.75])  # some faint
        width = int(rng.choice([1, 2, 2, 3]))
        cv2.polylines(dark, [path], False, float(strength), width, cv2.LINE_AA)
        cv2.polylines(truth, [path], False, 255, 1, cv2.LINE_8)
    return dark, truth


def make_image(seed, h=900, w=1200):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # background: uneven illumination + nodular texture + fine grain
    img = 175 + 25 * np.cos((xx - rng.uniform(0, w)) / w * np.pi) * np.cos((yy - rng.uniform(0, h)) / h * np.pi)
    nodules = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6)
    img += 90 * nodules
    img += rng.normal(0, 6, (h, w))

    # polishing scratches: faint, long, parallel lines
    angle = rng.uniform(0, np.pi)
    for _ in range(25):
        c = rng.uniform(-w, 2 * w)
        x0, y0 = c, 0
        x1, y1 = c + np.cos(angle) * 3 * h, np.sin(angle) * 3 * h
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), float(img.mean() - rng.uniform(8, 15)), 1, cv2.LINE_AA)

    # dark pits (round spots, not cracks)
    for _ in range(40):
        cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(2, 6)),
                   float(rng.uniform(60, 110)), -1, cv2.LINE_AA)

    dark, truth = crack_network(h, w, n_cells=int(rng.uniform(35, 55)), rng=rng)
    img = img - cv2.GaussianBlur(dark, (0, 0), 0.8)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)               # slight optical blur
    return np.clip(img, 0, 255).astype(np.uint8), truth


def main():
    out = Path("demo"); out.mkdir(exist_ok=True)
    for i, seed in enumerate([7, 21, 42], start=1):
        img, truth = make_image(seed)
        cv2.imwrite(str(out / f"sample_{i:02d}.png"), img)
        cv2.imwrite(str(out / f"sample_{i:02d}_truth.png"), truth)
        print(f"demo/sample_{i:02d}.png  (true crack length: {(truth > 0).sum()} px)")


if __name__ == "__main__":
    main()
