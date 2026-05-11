#!/usr/bin/env python3
"""
merge_pair_absolute.py — Merge all tiles of one pair into a single
DICe-format .txt with absolute (image-space) coordinates, for use with
dic_visualizer.html.

Only rot000_flip0 (original) variants are used; rotations would duplicate
at the same coords after un-rotation. Overlapping pixels from the
"anchored last tile" strategy are deduplicated (same coords sample the
same DICe data, so last-write-wins is fine).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HEADER = ("SUBSET_ID,COORDINATE_X,COORDINATE_Y,DISPLACEMENT_X,DISPLACEMENT_Y,"
          "SIGMA,GAMMA,BETA,STATUS_FLAG,UNCERTAINTY,"
          "VSG_STRAIN_XX,VSG_STRAIN_YY,VSG_STRAIN_XY")

def merge_pair(pair_dir: Path, out_path: Path) -> int:
    tile_dirs = sorted(pair_dir.glob("tile_*_rot000_flip0"))
    if not tile_dirs:
        return 0

    chunks = []
    for td in tile_dirs:
        flow = np.load(td / "flow.npy")          # (H, W, 2) float32
        meta = json.loads((td / "metadata.json").read_text())
        y0, x0 = meta["original_position"]

        valid = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
        ys, xs = np.where(valid)
        dxs = flow[ys, xs, 0]
        dys = flow[ys, xs, 1]
        chunks.append(np.column_stack([
            xs + x0, ys + y0, dxs, dys
        ]).astype(np.float64))

    arr = np.concatenate(chunks, axis=0)         # (N, 4)

    # Dedup overlapping pixels — combine x,y to single int key
    keys = (arr[:, 0].astype(np.int64) * 100000) + arr[:, 1].astype(np.int64)
    _, uniq_idx = np.unique(keys, return_index=True)
    arr = arr[uniq_idx]

    # Sort by (Y, X) for visualizer-friendly ordering
    order = np.lexsort((arr[:, 0], arr[:, 1]))
    arr = arr[order]

    # Write with constant SIGMA/GAMMA/STATUS values (success markers)
    n = arr.shape[0]
    sids = np.arange(n, dtype=np.int64)
    out = np.column_stack([
        sids, arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64),
        arr[:, 2], arr[:, 3],
        np.full(n, 0.01), np.full(n, 0.1),       # SIGMA, GAMMA
        np.zeros(n), np.full(n, 4),              # BETA, STATUS_FLAG (success)
        np.zeros(n),                             # UNCERTAINTY
        np.zeros(n), np.zeros(n), np.zeros(n),   # strain xx,yy,xy
    ])
    fmt = ("%d,%d,%d,%.6f,%.6f,%.2f,%.1f,"
           "%d,%d,%d,%d,%d,%d")
    np.savetxt(out_path, out, fmt=fmt, header=HEADER, comments="")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_dirs = sorted(p for p in args.tiles_root.glob("pair_*") if p.is_dir())
    print(f"Found {len(pair_dirs)} pairs in {args.tiles_root}")

    t0 = time.time()
    for pd in pair_dirs:
        out = args.output_dir / f"DICe_solution_{pd.name}.txt"
        n = merge_pair(pd, out)
        size_mb = out.stat().st_size / 1e6
        print(f"  {pd.name}: {n:,} pixels -> {out.name} ({size_mb:.1f} MB)")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
