import os
import argparse
import numpy as np


def angle_to_index(angle, n_angles):
    a = int(angle)
    if n_angles <= 0:
        raise ValueError("n_angles must be > 0")
    # Map human-friendly degree (1..n) to 0-based index. Allow angle==n (e.g. 360).
    mod = a % n_angles
    idx = mod - 1 if mod != 0 else n_angles - 1
    return idx


def circle_mask(h, w, x0, x1, y0, y1):
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    radius = min(cx - x0, cy - y0)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return mask


def extract_values(proj, angle_idx, bbox):
    # proj expected shape: (n_angles, H, W) or (n_angles, H*W) -- handle 2D per angle
    slice_ = proj[angle_idx]
    if slice_.ndim == 1:
        # try to infer square shape
        length = slice_.size
        s = int(np.sqrt(length))
        if s * s != length:
            raise ValueError("Unable to reshape 1D projection to square. Provide 2D data.")
        slice_ = slice_.reshape((s, s))
    h, w = slice_.shape
    x0, x1, y0, y1 = bbox
    mask = circle_mask(h, w, x0, x1, y0, y1)
    values = slice_[mask]
    return values


def process_file(path, angles, bbox):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return {}
    proj = np.load(path)
    n_angles = proj.shape[0]
    results = {}
    for angle in angles:
        idx = angle_to_index(angle, n_angles)
        vals = extract_values(proj, idx, bbox)
        results[angle] = vals
        tag = os.path.splitext(os.path.basename(path))[0]
        outname = f"circle_values_{tag}_angle_{angle}.npy"
        np.save(outname, vals)
        print(f"Saved {outname}: count={vals.size} mean={vals.mean() if vals.size else float('nan'):.6f}")
    return results


def main():
    p = argparse.ArgumentParser(description="Extract circular pixel values from projection arrays by angle")
    p.add_argument("--files", nargs="+", default=["projections_poisson_high.npy", "projections_poisson_low.npy"], help=".npy projection files to process")
    p.add_argument("--angles", nargs="+", type=int, default=[90, 360], help="Angles (degrees) to extract")
    p.add_argument("--bbox", nargs=4, type=int, default=[64, 192, 64, 192], help="x0 x1 y0 y1 bounding box for circle")
    args = p.parse_args()

    for f in args.files:
        print(f"Processing {f} ...")
        process_file(f, args.angles, args.bbox)


if __name__ == "__main__":
    main()
