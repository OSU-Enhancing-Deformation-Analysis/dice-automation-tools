#!/usr/bin/env python3
"""
Convert tile flow.npy files back to DICe-native .txt format.

Why this exists
---------------
Brock's RAFT training pipeline reads DICe-native CSV-style .txt files
(SUBSET_ID, COORDINATE_X, COORDINATE_Y, DISPLACEMENT_X, DISPLACEMENT_Y,
SIGMA, GAMMA, ...) and uses SIGMA as a per-pixel mask.

Our generate_tiles.py emits flow.npy (shape: (H, W, 2), float32) where
channel 0 is dx and channel 1 is dy, with NaN marking pixels where DICe
failed. This script is the bridge: each tile -> one DICe_solution.txt
that Brock's loader can ingest unchanged.

Output format (header is taken verbatim from Dr Chen's reference file):
SUBSET_ID,COORDINATE_X,COORDINATE_Y,DISPLACEMENT_X,DISPLACEMENT_Y,
SIGMA,GAMMA,BETA,STATUS_FLAG,UNCERTAINTY,VSG_STRAIN_XX,VSG_STRAIN_YY,VSG_STRAIN_XY

Per-pixel encoding
------------------
Valid pixel (non-NaN flow):
    SIGMA=0.01, GAMMA=0.1, STATUS_FLAG=4 (DICe "success")
Failed pixel (NaN flow):
    SIGMA=-1, GAMMA=0, STATUS_FLAG=11, dx/dy set to 0
    (matches DICe's own failure convention: failed points have SIGMA=-1)

Coordinate convention
---------------------
Default: tile-local (0..H-1 in y, 0..W-1 in x).
Pass --absolute to add metadata.json's original_position offset so coords
are in the parent SEM image frame (matches Dr Chen's reference style).

Usage
-----
Single tile:
    python npy_to_dice_txt.py <tile_dir>
        -> writes <tile_dir>/DICe_solution.txt

Whole pair (all tile_NNNN_rot000_flip0 dirs in a pair folder):
    python npy_to_dice_txt.py <pair_dir> --batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DICE_HEADER = (
    "SUBSET_ID,COORDINATE_X,COORDINATE_Y,DISPLACEMENT_X,DISPLACEMENT_Y,"
    "SIGMA,GAMMA,BETA,STATUS_FLAG,UNCERTAINTY,"
    "VSG_STRAIN_XX,VSG_STRAIN_YY,VSG_STRAIN_XY"
)


def convert_tile(tile_dir: Path, absolute_coords: bool = False, out_path: Path | None = None) -> Path:
    """Convert one tile folder's flow.npy -> DICe_solution.txt.

    Returns the output path.
    """
    flow_path = tile_dir / "flow.npy"
    meta_path = tile_dir / "metadata.json"
    if not flow_path.exists():
        raise FileNotFoundError(f"flow.npy not found in {tile_dir}")

    flow = np.load(flow_path)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(
            f"Expected flow.npy shape (H,W,2); got {flow.shape}"
        )
    height, width, _ = flow.shape

    # Optional offset from tile-local to absolute parent-image coords.
    offset_y, offset_x = 0, 0
    if absolute_coords:
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json missing in {tile_dir}; cannot do absolute coords"
            )
        meta = json.loads(meta_path.read_text())
        # original_position is [row, col] = [y, x]
        offset_y, offset_x = meta["original_position"]

    # Build output rows. Row-major: y outer, x inner. SUBSET_ID = y*W + x.
    if out_path is None:
        out_path = tile_dir / "DICe_solution.txt"
    with out_path.open("w") as f:
        f.write(DICE_HEADER + "\n")
        for y in range(height):
            for x in range(width):
                dx = float(flow[y, x, 0])
                dy = float(flow[y, x, 1])
                subset_id = y * width + x
                coord_x = x + offset_x
                coord_y = y + offset_y

                if np.isnan(dx) or np.isnan(dy):
                    # DICe convention: failed points have SIGMA=-1, FLAG=11
                    sigma = -1.0
                    gamma = 0.0
                    status_flag = 11.0
                    dx_out = 0.0
                    dy_out = 0.0
                else:
                    sigma = 0.01
                    gamma = 0.1
                    status_flag = 4.0  # DICe "success"
                    dx_out = dx
                    dy_out = dy

                # Match Dr Chen's reference scientific notation: "%.4E"
                f.write(
                    f"{subset_id},"
                    f"{coord_x:.4E},{coord_y:.4E},"
                    f"{dx_out:.4E},{dy_out:.4E},"
                    f"{sigma:.4E},{gamma:.4E},"
                    f"0.0000E+00,"        # BETA
                    f"{status_flag:.4E},"  # STATUS_FLAG (Dr Chen writes it as float, e.g. 4.0000E+00)
                    f"0.0000E+00,"        # UNCERTAINTY
                    f"0.0000E+00,0.0000E+00,0.0000E+00\n"  # strains
                )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert tile flow.npy back to DICe .txt format"
    )
    parser.add_argument("path", type=Path, help="Tile dir or pair dir")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat path as pair dir; convert all tile_*_rot000_flip0 subdirs",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute parent-image coords (default: tile-local 0..127)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Write output files into this dir instead of next to flow.npy. "
            "In batch mode, files are named DICe_solution_NNN.txt sequentially "
            "(compatible with dic_visualizer.html). In single mode, file is "
            "named after the tile (e.g. tile_0000_rot000_flip0.txt)."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        if not args.path.is_dir():
            print(f"Pair dir not found: {args.path}", file=sys.stderr)
            return 1
        tile_dirs = sorted(args.path.glob("tile_*_rot000_flip0"))
        if not tile_dirs:
            print(f"No tile_*_rot000_flip0 subdirs in {args.path}", file=sys.stderr)
            return 1
        print(f"Converting {len(tile_dirs)} tiles from {args.path.name}")
        for idx, td in enumerate(tile_dirs):
            if args.output_dir is not None:
                out_path = args.output_dir / f"DICe_solution_{idx:03d}.txt"
            else:
                out_path = None
            out = convert_tile(td, absolute_coords=args.absolute, out_path=out_path)
            print(f"  {td.name} -> {out.name}")
    else:
        if args.output_dir is not None:
            out_path = args.output_dir / f"{args.path.name}.txt"
        else:
            out_path = None
        out = convert_tile(args.path, absolute_coords=args.absolute, out_path=out_path)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
