#!/usr/bin/env python3
"""
generate_tiles.py
Cut DICe exploration results into 128x128 training tiles for ML.

For each pair directory under --results-dir this tool:
  1. Loads the reference/deformed grayscale SEM frames.
  2. Reads the DICe_solution_*.txt CSV and reconstructs a dense
     (H, W, 2) displacement field with NaN where DICe had no data or failed.
  3. Builds a quality mask (SIGMA < --sigma-threshold AND GAMMA < --gamma-threshold).
  4. Walks a non-overlapping grid of --tile-size (edge-underflow tiles are dropped).
  5. Keeps tiles with valid-pixel ratio above --min-valid-ratio and writes:
         ref/<name>.tif   uint8 reference tile
         def/<name>.tif   uint8 deformed tile
         flow/<name>.npy  float32 (tile, tile, 2) flow tile, channels [dx, dy]
  6. Writes metadata.json per pair and summary.json at the output root.

Usage:
    python3 generate_tiles.py \\
        --results-dir exploration_results_eswg007_ref60 \\
        --images-dir  processed_datasets/ESWG007/preprocessed \\
        --ref-frame   60 \\
        --output-dir  training_tiles_128
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # pylint: disable=import-error,no-name-in-module
import numpy as np
import pandas as pd

# cv2 is a C extension and pylint cannot introspect its members.
# pylint: disable=no-member


DEFAULT_TILE_SIZE = 128
DEFAULT_SIGMA_THRESHOLD = 0.03
DEFAULT_GAMMA_THRESHOLD = 0.5
DEFAULT_MIN_VALID_RATIO = 0.5

PAIR_DIR_REGEX = re.compile(r"^pair_f(\d+)_vs_f(\d+)$")
SOLUTION_FILE_REGEX = re.compile(r"^DICe_solution_(\d+)\.txt$")
LOGGER = logging.getLogger("generate_tiles")


@dataclass(frozen=True)
class TileConfig:
    """Runtime knobs controlling tile extraction and quality filtering."""
    tile_size: int
    sigma_threshold: float
    gamma_threshold: float
    min_valid_ratio: float


@dataclass
class PairData:  # pylint: disable=too-many-instance-attributes
    """Dense arrays needed to tile a single image pair."""
    pair_id: str
    ref_idx: int
    def_idx: int
    ref_image: np.ndarray
    def_image: np.ndarray
    flow: np.ndarray
    quality_mask: np.ndarray
    failed_mask: np.ndarray

    @property
    def image_shape(self) -> Tuple[int, int]:
        """Return the (H, W) shape shared by all per-pair arrays."""
        return (int(self.ref_image.shape[0]), int(self.ref_image.shape[1]))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cut DICe exploration results into training tiles.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Directory containing pair_fXXXX_vs_fYYYY subdirectories.",
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True,
        help="Directory of preprocessed frames (frame_NNNN.tif).",
    )
    parser.add_argument(
        "--ref-frame", type=int, required=True,
        help="Reference frame number shared by every pair.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training_tiles_128"),
        help="Root directory for tile outputs (created if missing).",
    )
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="Square tile size in pixels.",
    )
    parser.add_argument(
        "--sigma-threshold", type=float, default=DEFAULT_SIGMA_THRESHOLD,
        help="Upper bound on SIGMA for a valid correlation.",
    )
    parser.add_argument(
        "--gamma-threshold", type=float, default=DEFAULT_GAMMA_THRESHOLD,
        help="Upper bound on GAMMA for a valid correlation.",
    )
    parser.add_argument(
        "--min-valid-ratio", type=float, default=DEFAULT_MIN_VALID_RATIO,
        help="Minimum fraction of valid pixels required to keep a tile (strict >).",
    )
    parser.add_argument(
        "--only-pair", type=str, default=None,
        help="Optional single pair directory name to process (for smoke tests).",
    )
    return parser.parse_args()


def load_grayscale_image(path: Path) -> np.ndarray:
    """Read an 8-bit grayscale TIFF into a (H, W) uint8 array."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def load_dice_solution(csv_path: Path) -> pd.DataFrame:
    """Read a DICe solution CSV. Files emitted by explore_frame_intervals.py
    begin directly with the CSV header, so no comment-line stripping is needed."""
    return pd.read_csv(csv_path)


def build_dense_fields(  # pylint: disable=too-many-locals
    df: pd.DataFrame,
    image_shape: Tuple[int, int],
    config: TileConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct dense arrays from a sparse DICe DataFrame.

    Returns:
        flow:    float32 (H, W, 2), NaN where the ROI has no data or DICe failed.
        failed:  bool (H, W), True at pixels where DICe reported a failure.
        quality: bool (H, W), True where the correlation succeeded AND
                 SIGMA < config.sigma_threshold AND GAMMA < config.gamma_threshold.
    """
    height, width = image_shape
    flow = np.full((height, width, 2), np.nan, dtype=np.float32)
    failed = np.zeros((height, width), dtype=bool)
    quality = np.zeros((height, width), dtype=bool)

    xs = df["COORDINATE_X"].to_numpy(dtype=np.int64)
    ys = df["COORDINATE_Y"].to_numpy(dtype=np.int64)
    dx = df["DISPLACEMENT_X"].to_numpy(dtype=np.float32)
    dy = df["DISPLACEMENT_Y"].to_numpy(dtype=np.float32)
    sigma = df["SIGMA"].to_numpy(dtype=np.float32)
    gamma = df["GAMMA"].to_numpy(dtype=np.float32)

    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys = xs[in_bounds], ys[in_bounds]
    dx, dy = dx[in_bounds], dy[in_bounds]
    sigma, gamma = sigma[in_bounds], gamma[in_bounds]

    # DICe v3.0-beta.8 marks failed correlations with SIGMA == -1 (and
    # STATUS_FLAG == 11); successful rows have SIGMA >= 0 and STATUS_FLAG == 4.
    # The SIGMA test is the robust cross-version signal.
    success = sigma >= 0.0
    failed[ys[~success], xs[~success]] = True

    flow[ys[success], xs[success], 0] = dx[success]
    flow[ys[success], xs[success], 1] = dy[success]

    valid = success & (sigma < config.sigma_threshold) & (gamma < config.gamma_threshold)
    quality[ys[valid], xs[valid]] = True
    return flow, failed, quality


def iter_tile_grid(
    image_shape: Tuple[int, int], tile_size: int,
) -> List[Tuple[int, int, int, int]]:
    """Return (row, col, y0, x0) tuples for a non-overlapping tile grid.

    Partial edge tiles are dropped so every returned tile is exactly tile_size
    pixels on both sides.
    """
    height, width = image_shape
    rows = height // tile_size
    cols = width // tile_size
    return [
        (row, col, row * tile_size, col * tile_size)
        for row in range(rows)
        for col in range(cols)
    ]


def tile_statistics(
    flow_tile: np.ndarray,
    quality_tile: np.ndarray,
    failed_tile: np.ndarray,
) -> Dict[str, float]:
    """Compute summary statistics for a single tile."""
    total = quality_tile.size
    valid_pixels = int(quality_tile.sum())
    failed_pixels = int(failed_tile.sum())
    valid_ratio = valid_pixels / total if total else 0.0

    if valid_pixels:
        with np.errstate(invalid="ignore"):
            mean_dx = float(np.nanmean(flow_tile[..., 0]))
            mean_dy = float(np.nanmean(flow_tile[..., 1]))
            magnitude = np.sqrt(flow_tile[..., 0] ** 2 + flow_tile[..., 1] ** 2)
            max_abs_disp = float(np.nanmax(magnitude))
    else:
        mean_dx = 0.0
        mean_dy = 0.0
        max_abs_disp = 0.0

    return {
        "valid_ratio": valid_ratio,
        "valid_pixels": valid_pixels,
        "failed_pixels": failed_pixels,
        "mean_disp_x": 0.0 if np.isnan(mean_dx) else mean_dx,
        "mean_disp_y": 0.0 if np.isnan(mean_dy) else mean_dy,
        "max_abs_disp": 0.0 if np.isnan(max_abs_disp) else max_abs_disp,
    }


def save_tile(
    pair_out: Path,
    name: str,
    ref_tile: np.ndarray,
    def_tile: np.ndarray,
    flow_tile: np.ndarray,
) -> None:
    """Persist one tile as ref/def TIFFs plus a flow .npy file."""
    ref_path = pair_out / "ref" / f"{name}.tif"
    def_path = pair_out / "def" / f"{name}.tif"
    flow_path = pair_out / "flow" / f"{name}.npy"
    if not cv2.imwrite(str(ref_path), ref_tile):
        raise IOError(f"Failed to write reference tile: {ref_path}")
    if not cv2.imwrite(str(def_path), def_tile):
        raise IOError(f"Failed to write deformed tile: {def_path}")
    np.save(flow_path, flow_tile)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def process_one_tile(
    pair_data: PairData,
    pair_out: Path,
    config: TileConfig,
    row: int,
    col: int,
    y0: int,
    x0: int,
) -> Optional[Dict[str, Any]]:
    """Evaluate one tile and, if it passes the quality filter, write it."""
    ts = config.tile_size
    y1, x1 = y0 + ts, x0 + ts
    quality_tile = pair_data.quality_mask[y0:y1, x0:x1]
    valid_ratio = float(quality_tile.mean())
    if valid_ratio <= config.min_valid_ratio:
        return None

    flow_tile = pair_data.flow[y0:y1, x0:x1].copy()
    failed_tile = pair_data.failed_mask[y0:y1, x0:x1]
    ref_tile = np.ascontiguousarray(pair_data.ref_image[y0:y1, x0:x1])
    def_tile = np.ascontiguousarray(pair_data.def_image[y0:y1, x0:x1])

    name = f"row{row}_col{col}"
    save_tile(pair_out, name, ref_tile, def_tile, flow_tile)

    stats = tile_statistics(flow_tile, quality_tile, failed_tile)
    return {
        "name": name,
        "row": row,
        "col": col,
        "bounds": [int(y0), int(x0), int(y1), int(x1)],
        **stats,
    }
# pylint: enable=too-many-arguments,too-many-positional-arguments,too-many-locals


def extract_pair_tiles(
    pair_data: PairData,
    pair_out: Path,
    config: TileConfig,
) -> Tuple[List[Dict[str, Any]], int]:
    """Walk the tile grid, save kept tiles, and return their metadata."""
    for subdir in ("ref", "def", "flow"):
        (pair_out / subdir).mkdir(parents=True, exist_ok=True)

    kept: List[Dict[str, Any]] = []
    tiles_possible = 0
    for row, col, y0, x0 in iter_tile_grid(pair_data.image_shape, config.tile_size):
        tiles_possible += 1
        entry = process_one_tile(pair_data, pair_out, config, row, col, y0, x0)
        if entry is not None:
            kept.append(entry)
    return kept, tiles_possible


def find_solution_csv(pair_dir: Path) -> Optional[Path]:
    """Return the DICe_solution_*.txt file inside a pair directory, or None."""
    matches = sorted(
        p for p in pair_dir.iterdir()
        if p.is_file() and SOLUTION_FILE_REGEX.match(p.name)
    )
    return matches[0] if matches else None


def load_pair(
    pair_dir: Path,
    images_dir: Path,
    ref_idx: int,
    def_idx: int,
    config: TileConfig,
) -> PairData:
    """Load images and DICe CSV for one pair and build dense fields."""
    ref_path = images_dir / f"frame_{ref_idx:04d}.tif"
    def_path = images_dir / f"frame_{def_idx:04d}.tif"
    ref_image = load_grayscale_image(ref_path)
    def_image = load_grayscale_image(def_path)
    if ref_image.shape != def_image.shape:
        raise ValueError(
            f"{pair_dir.name}: reference/deformed shape mismatch "
            f"{ref_image.shape} vs {def_image.shape}"
        )

    csv_path = find_solution_csv(pair_dir)
    if csv_path is None:
        raise FileNotFoundError(f"No DICe_solution_*.txt in {pair_dir}")

    size_mb = csv_path.stat().st_size / (1024 * 1024)
    LOGGER.info("Loading DICe CSV %s (%.1f MB)", csv_path.name, size_mb)
    df = load_dice_solution(csv_path)
    flow, failed_mask, quality_mask = build_dense_fields(
        df, (int(ref_image.shape[0]), int(ref_image.shape[1])), config,
    )
    return PairData(
        pair_id=pair_dir.name,
        ref_idx=ref_idx,
        def_idx=def_idx,
        ref_image=ref_image,
        def_image=def_image,
        flow=flow,
        quality_mask=quality_mask,
        failed_mask=failed_mask,
    )


def write_pair_metadata(
    pair_out: Path,
    pair_data: PairData,
    config: TileConfig,
    kept: List[Dict[str, Any]],
    tiles_possible: int,
) -> float:
    """Write metadata.json for a processed pair and return its mean valid ratio."""
    mean_valid_ratio = (
        sum(tile["valid_ratio"] for tile in kept) / len(kept) if kept else 0.0
    )
    metadata = {
        "pair_id": pair_data.pair_id,
        "reference_frame": pair_data.ref_idx,
        "deformed_frame": pair_data.def_idx,
        "image_shape": list(pair_data.image_shape),
        "tile_size": config.tile_size,
        "sigma_threshold": config.sigma_threshold,
        "gamma_threshold": config.gamma_threshold,
        "min_valid_ratio": config.min_valid_ratio,
        "tiles_possible": tiles_possible,
        "tiles_kept": len(kept),
        "mean_valid_ratio": mean_valid_ratio,
        "tiles": kept,
    }
    with (pair_out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return mean_valid_ratio


def process_pair(
    pair_dir: Path,
    images_dir: Path,
    output_dir: Path,
    ref_frame: int,
    config: TileConfig,
) -> Optional[Dict[str, Any]]:
    """Process one pair directory and return its aggregated summary dict."""
    match = PAIR_DIR_REGEX.match(pair_dir.name)
    if match is None:
        LOGGER.warning("Skipping unrecognized directory: %s", pair_dir.name)
        return None
    ref_idx = int(match.group(1))
    def_idx = int(match.group(2))
    if ref_idx != ref_frame:
        LOGGER.warning(
            "Skipping %s: reference frame %d != expected %d",
            pair_dir.name, ref_idx, ref_frame,
        )
        return None

    pair_data = load_pair(pair_dir, images_dir, ref_idx, def_idx, config)
    pair_out = output_dir / pair_dir.name
    kept, tiles_possible = extract_pair_tiles(pair_data, pair_out, config)
    mean_valid_ratio = write_pair_metadata(
        pair_out, pair_data, config, kept, tiles_possible,
    )

    LOGGER.info(
        "%s: kept %d / %d tiles (mean valid ratio %.3f)",
        pair_dir.name, len(kept), tiles_possible, mean_valid_ratio,
    )
    return {
        "pair_id": pair_dir.name,
        "reference_frame": ref_idx,
        "deformed_frame": def_idx,
        "tiles_possible": tiles_possible,
        "tiles_kept": len(kept),
        "mean_valid_ratio": mean_valid_ratio,
    }


def discover_pair_dirs(
    results_dir: Path, only_pair: Optional[str],
) -> List[Path]:
    """Find pair subdirectories, optionally filtered to a single name."""
    pair_dirs = sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and PAIR_DIR_REGEX.match(p.name)
    )
    if only_pair is not None:
        pair_dirs = [p for p in pair_dirs if p.name == only_pair]
    return pair_dirs


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    config: TileConfig,
    per_pair: List[Dict[str, Any]],
) -> None:
    """Write the top-level summary.json covering every processed pair."""
    total_tiles = sum(s["tiles_kept"] for s in per_pair)
    tiles_possible_per_pair = per_pair[0]["tiles_possible"] if per_pair else 0
    payload = {
        "results_dir": str(args.results_dir),
        "images_dir": str(args.images_dir),
        "output_dir": str(output_dir),
        "reference_frame": args.ref_frame,
        "tile_size": config.tile_size,
        "sigma_threshold": config.sigma_threshold,
        "gamma_threshold": config.gamma_threshold,
        "min_valid_ratio": config.min_valid_ratio,
        "pairs_processed": len(per_pair),
        "tiles_possible_per_pair": tiles_possible_per_pair,
        "total_tiles_kept": total_tiles,
        "by_pair": per_pair,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    """Entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.results_dir.is_dir():
        LOGGER.error("Results directory not found: %s", args.results_dir)
        return 1
    if not args.images_dir.is_dir():
        LOGGER.error("Images directory not found: %s", args.images_dir)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TileConfig(
        tile_size=args.tile_size,
        sigma_threshold=args.sigma_threshold,
        gamma_threshold=args.gamma_threshold,
        min_valid_ratio=args.min_valid_ratio,
    )

    pair_dirs = discover_pair_dirs(args.results_dir, args.only_pair)
    if not pair_dirs:
        LOGGER.error("No pair directories found in %s", args.results_dir)
        return 1

    per_pair: List[Dict[str, Any]] = []
    for pair_dir in pair_dirs:
        summary = process_pair(
            pair_dir, args.images_dir, args.output_dir, args.ref_frame, config,
        )
        if summary is not None:
            per_pair.append(summary)

    write_summary(args.output_dir, args, config, per_pair)
    total_tiles = sum(s["tiles_kept"] for s in per_pair)
    LOGGER.info(
        "Done: %d pairs processed, %d tiles written to %s",
        len(per_pair), total_tiles, args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
