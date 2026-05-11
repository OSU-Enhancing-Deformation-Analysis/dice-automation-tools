#!/usr/bin/env python3
"""
Same as merge_pair_absolute.py but outputs a COMPLETE GRID — every pixel in
the reliable zone is emitted, with failed pixels marked SIGMA=-1, FLAG=11
and dx/dy=0. This makes the visualizer auto-detect isGrid=true and use the
fast heatmap path instead of scattergl.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HEADER = ("SUBSET_ID,COORDINATE_X,COORDINATE_Y,DISPLACEMENT_X,DISPLACEMENT_Y,"
          "SIGMA,GAMMA,BETA,STATUS_FLAG,UNCERTAINTY,"
          "VSG_STRAIN_XX,VSG_STRAIN_YY,VSG_STRAIN_XY")

# Image / reliable zone parameters (matches generate_tiles_systematic.py)
IMG_H, IMG_W = 883, 1024
TRIM = 17

def merge_pair_grid(pair_dir: Path, out_path: Path) -> tuple[int, int]:
    """Returns (n_total, n_valid)."""
    tile_dirs = sorted(pair_dir.glob("tile_*_rot000_flip0"))
    if not tile_dirs:
        return 0, 0

    # Build full-image grid initialized to NaN
    full_dx = np.full((IMG_H, IMG_W), np.nan, dtype=np.float32)
    full_dy = np.full((IMG_H, IMG_W), np.nan, dtype=np.float32)

    for td in tile_dirs:
        flow = np.load(td / "flow.npy")          # (H, W, 2) float32
        meta = json.loads((td / "metadata.json").read_text())
        y0, x0 = meta["original_position"]
        h, w = flow.shape[:2]
        full_dx[y0:y0 + h, x0:x0 + w] = flow[..., 0]
        full_dy[y0:y0 + h, x0:x0 + w] = flow[..., 1]

    # Emit every pixel in the reliable zone (whether valid or NaN)
    y_range = range(TRIM, IMG_H - TRIM)
    x_range = range(TRIM, IMG_W - TRIM)

    rows = []
    sid = 0
    n_valid = 0
    for y in y_range:
        for x in x_range:
            dx = full_dx[y, x]
            dy = full_dy[y, x]
            if np.isfinite(dx) and np.isfinite(dy):
                rows.append(f"{sid},{x},{y},{dx:.6f},{dy:.6f},0.01,0.1,0,4,0,0,0,0")
                n_valid += 1
            else:
                # Missing pixel: still emit, marked as failed
                rows.append(f"{sid},{x},{y},0,0,-1,0,0,11,0,0,0,0")
            sid += 1

    out_path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return len(rows), n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_dirs = sorted(p for p in args.tiles_root.glob("pair_*") if p.is_dir())
    print(f"Found {len(pair_dirs)} pairs")

    t0 = time.time()
    for pd in pair_dirs:
        out = args.output_dir / f"DICe_solution_{pd.name}.txt"
        n_total, n_valid = merge_pair_grid(pd, out)
        size_mb = out.stat().st_size / 1e6
        pct = 100 * n_valid / n_total if n_total else 0
        print(f"  {pd.name}: {n_total:,} grid pts ({n_valid:,} valid, {pct:.1f}%) -> {size_mb:.1f} MB")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
