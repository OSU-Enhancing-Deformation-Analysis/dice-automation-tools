#!/usr/bin/env python3
"""
generate_tiles.py — Training tile generator with edge trimming, random crops,
and 8-way augmentation.

Pipeline (per image pair):
  1. Load full-resolution DICe displacement field (step_size=1) from CSV.
  2. Load corresponding reference and deformed SEM images.
  3. Edge trim: discard outer 17 px on all sides (subset_size / 2 = 35 / 2).
     The reliable zone for 1024 x 883 images is 990 x 849.
  4. Random crop: sample N unique 128x128 tile positions within the reliable
     zone using ``default_rng().choice(..., replace=False)``.  Seeded by a
     hash of the pair name for reproducibility.
  5. Quality filter: keep tiles where >= 50% of DICe pixels are valid
     (non-NaN).
  6. Pre-augmentation statistics: accumulate per-pair and global (dx, dy)
     mean/std over the original tiles.  These stats MUST be computed before
     augmentation — the 8-way D4 symmetry would otherwise collapse mean_dx
     and mean_dy to ~0 and erase the directional-bias signal Dr. Chen uses
     for review (currently mean_dy ~= -8.8, reflecting y-axis tensile load).
  7. Augmentation: 8 variants per surviving tile — 4 rotations (0/90/180/270
     deg CCW) x 2 flip states (none / horizontal).  Displacement vectors are
     transformed consistently with the spatial transform.
  8. Output: ref.tif + def.tif + flow.npy + metadata.json per tile variant,
     plus a top-level summary.json.

Inputs:
  DICe results  — exploration_results_eswg007_ref60/
                   exploration_results_eswg007_ref60_rerun/  (takes precedence)
  SEM images    — processed_datasets/ESWG007/preprocessed/

Output (default):  training_tiles_128_v2/

Usage:
  python3 generate_tiles.py
  python3 generate_tiles.py --output-dir out/ --force
  python3 generate_tiles.py --help
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TILE_SIZE = 128
DEFAULT_EDGE_TRIM = 17            # subset_size / 2 = 35 / 2 ~ 17
DEFAULT_TILES_PER_PAIR = 32
DEFAULT_MIN_VALID_RATIO = 0.5

DEFAULT_RESULTS_DIRS: Tuple[Path, ...] = (
    Path("exploration_results_eswg007_ref60"),
    Path("exploration_results_eswg007_ref60_rerun"),
)
DEFAULT_IMAGES_DIR = Path("processed_datasets/ESWG007/preprocessed")
DEFAULT_OUTPUT_DIR = Path("training_tiles_128_v2")
DEFAULT_REF_FRAME = 60

PAIR_DIR_RE = re.compile(r"^pair_f(\d+)_vs_f(\d+)$")
SOLUTION_FILE_RE = re.compile(r"^DICe_solution_(\d+)\.txt$")

LOG = logging.getLogger("generate_tiles")


@dataclass(frozen=True)
class TileConfig:
    """Runtime knobs controlling tile generation."""

    tile_size: int
    edge_trim: int
    tiles_per_pair: int
    min_valid_ratio: float


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def rotate_flow_vectors(flow: np.ndarray, k: int) -> np.ndarray:
    """Rotate (dx, dy) vectors to match ``np.rot90(image, k)``.

    ``np.rot90(k=1)`` maps pixel (row, col) to (W-1-col, row).
    The corresponding vector transform:

        k=1 (90 deg CCW):  (dx, dy) -> ( dy, -dx)
        k=2 (180 deg):     (dx, dy) -> (-dx, -dy)
        k=3 (270 deg CCW): (dx, dy) -> (-dy,  dx)
    """
    k = k % 4
    if k == 0:
        return flow
    dx, dy = flow[..., 0], flow[..., 1]
    if k == 1:
        new_dx, new_dy = dy, -dx
    elif k == 2:
        new_dx, new_dy = -dx, -dy
    else:
        new_dx, new_dy = -dy, dx
    return np.stack([new_dx, new_dy], axis=-1)


def apply_augmentation(
    ref: np.ndarray,
    def_img: np.ndarray,
    flow: np.ndarray,
    rotation_deg: int,
    flip_horizontal: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one rotation + optional horizontal flip.

    Order: rotate first, then flip.  Returned arrays are contiguous copies
    whenever a transform is applied; the identity case (rot=0, flip=False)
    returns the original arrays unchanged.
    """
    k = rotation_deg // 90

    if k > 0:
        ref = np.ascontiguousarray(np.rot90(ref, k=k))
        def_img = np.ascontiguousarray(np.rot90(def_img, k=k))
        flow = np.ascontiguousarray(np.rot90(flow, k=k))
        flow = rotate_flow_vectors(flow, k)

    if flip_horizontal:
        ref = np.ascontiguousarray(np.fliplr(ref))
        def_img = np.ascontiguousarray(np.fliplr(def_img))
        flow = np.ascontiguousarray(np.fliplr(flow))
        flow[..., 0] = -flow[..., 0]

    return ref, def_img, flow


def sanity_test_augmentation() -> None:
    """Assert displacement-vector transforms are correct, then return.

    Tests a unit vector (1, 0) through all four rotations, horizontal flip,
    combined rotation + flip, and a non-trivial vector (1, 2).

    Exits with ``AssertionError`` if any check fails.
    """
    h, w = 4, 4
    ref = np.zeros((h, w), dtype=np.uint8)
    def_img = np.zeros((h, w), dtype=np.uint8)

    def _field(dx_val: float, dy_val: float) -> np.ndarray:
        f = np.empty((h, w, 2), dtype=np.float32)
        f[..., 0] = dx_val
        f[..., 1] = dy_val
        return f

    def _check(result: np.ndarray, exp_dx: float, exp_dy: float,
               label: str) -> None:
        assert np.allclose(result[..., 0], exp_dx), (
            f"{label} dx: expected {exp_dx}, got {result[..., 0].mean():.4f}")
        assert np.allclose(result[..., 1], exp_dy), (
            f"{label} dy: expected {exp_dy}, got {result[..., 1].mean():.4f}")

    # --- Unit vector (1, 0) through pure rotations ---
    _, _, f = apply_augmentation(ref, def_img, _field(1, 0), 90, False)
    _check(f, 0.0, -1.0, "90 deg CCW")

    _, _, f = apply_augmentation(ref, def_img, _field(1, 0), 180, False)
    _check(f, -1.0, 0.0, "180 deg")

    _, _, f = apply_augmentation(ref, def_img, _field(1, 0), 270, False)
    _check(f, 0.0, 1.0, "270 deg CCW")

    # --- Horizontal flip ---
    _, _, f = apply_augmentation(ref, def_img, _field(1, 0), 0, True)
    _check(f, -1.0, 0.0, "H-flip")

    # --- 90 deg CCW + flip: (1,0) -> (0,-1) -> flip dx -> (0,-1) ---
    _, _, f = apply_augmentation(ref, def_img, _field(1, 0), 90, True)
    _check(f, 0.0, -1.0, "90 deg CCW + H-flip")

    # --- Non-trivial vector (1, 2) at 90 deg CCW: (dy=2, -dx=-1) ---
    _, _, f = apply_augmentation(ref, def_img, _field(1, 2), 90, False)
    _check(f, 2.0, -1.0, "90 deg CCW (1,2)")

    LOG.info("Sanity tests passed")


# ---------------------------------------------------------------------------
# DICe I/O
# ---------------------------------------------------------------------------

def find_solution_csv(pair_dir: Path) -> Optional[Path]:
    """Return the first ``DICe_solution_*.txt`` in *pair_dir*, or None."""
    for p in sorted(pair_dir.iterdir()):
        if p.is_file() and SOLUTION_FILE_RE.match(p.name):
            return p
    return None


def load_dice_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Read a DICe solution CSV and return the five needed columns.

    Column lookup is by header name (robust against upstream DICe schema
    changes) rather than by fixed column index.  Returns a dict mapping
    column names to 1-D numpy arrays.
    """
    with open(csv_path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")

    col_map = {name.strip(): i for i, name in enumerate(header)}
    needed = [
        "COORDINATE_X", "COORDINATE_Y",
        "DISPLACEMENT_X", "DISPLACEMENT_Y",
        "SIGMA",
    ]
    usecols = tuple(col_map[n] for n in needed)

    data = np.loadtxt(
        csv_path, delimiter=",", skiprows=1, usecols=usecols,
    )
    return {name: data[:, i] for i, name in enumerate(needed)}


def build_dense_flow(
    cols: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Reconstruct a dense (H, W, 2) float32 displacement field.

    Pixels with no DICe data or failed correlations (SIGMA == -1) are NaN.
    """
    height, width = image_shape
    flow = np.full((height, width, 2), np.nan, dtype=np.float32)

    xs = cols["COORDINATE_X"].astype(np.int64)
    ys = cols["COORDINATE_Y"].astype(np.int64)
    dx = cols["DISPLACEMENT_X"].astype(np.float32)
    dy = cols["DISPLACEMENT_Y"].astype(np.float32)
    sigma = cols["SIGMA"].astype(np.float32)

    good = (
        (sigma >= 0.0)
        & (xs >= 0) & (xs < width)
        & (ys >= 0) & (ys < height)
    )
    flow[ys[good], xs[good], 0] = dx[good]
    flow[ys[good], xs[good], 1] = dy[good]
    return flow


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_grayscale(path: Path) -> np.ndarray:
    """Read a grayscale TIFF into an (H, W) uint8 array."""
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def discover_pairs(
    results_dirs: List[Path],
    ref_frame: int,
    only_pair: Optional[str] = None,
) -> Dict[str, Path]:
    """Find pair directories.  Later dirs override earlier (rerun wins)."""
    pairs: Dict[str, Path] = {}
    for rdir in results_dirs:
        if not rdir.is_dir():
            LOG.warning("Results directory not found, skipping: %s", rdir)
            continue
        for entry in sorted(rdir.iterdir()):
            if not entry.is_dir():
                continue
            m = PAIR_DIR_RE.match(entry.name)
            if m is None:
                continue
            if int(m.group(1)) != ref_frame:
                continue
            if only_pair is not None and entry.name != only_pair:
                continue
            if entry.name in pairs:
                LOG.info("Rerun overrides %s", entry.name)
            pairs[entry.name] = entry
    return dict(sorted(pairs.items()))


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

def pair_seed(pair_name: str) -> int:
    """Deterministic seed from pair name for reproducible random crops."""
    return int(hashlib.sha256(pair_name.encode()).hexdigest()[:8], 16)


def sample_positions(
    image_shape: Tuple[int, int],
    pair_name: str,
    config: TileConfig,
) -> List[Tuple[int, int]]:
    """Sample unique (y_start, x_start) positions from the reliable zone.

    Uses ``default_rng().choice(..., replace=False)`` so every returned
    position is distinct.  Each position guarantees the full tile lies
    within the edge-trimmed area.
    """
    h, w = image_shape
    trim, tile = config.edge_trim, config.tile_size
    y_count = h - 2 * trim - tile + 1
    x_count = w - 2 * trim - tile + 1
    if y_count <= 0 or x_count <= 0:
        LOG.error("Image %dx%d too small for tiling", w, h)
        return []

    total = y_count * x_count
    n_sample = min(config.tiles_per_pair, total)
    rng = np.random.default_rng(pair_seed(pair_name))
    flat = rng.choice(total, size=n_sample, replace=False)
    return [
        (int(trim + fi // x_count), int(trim + fi % x_count))
        for fi in flat.tolist()
    ]


def tile_valid_ratio(flow_tile: np.ndarray) -> float:
    """Fraction of non-NaN pixels in a flow tile."""
    return float(np.isfinite(flow_tile[..., 0]).mean())


def tile_stats(flow_tile: np.ndarray) -> Dict[str, float]:
    """Mean and std of dx, dy over valid (non-NaN) pixels."""
    def safe(v: float) -> float:
        return 0.0 if not np.isfinite(v) else v

    with np.errstate(invalid="ignore"):
        return {
            "mean_dx": safe(float(np.nanmean(flow_tile[..., 0]))),
            "mean_dy": safe(float(np.nanmean(flow_tile[..., 1]))),
            "std_dx":  safe(float(np.nanstd(flow_tile[..., 0]))),
            "std_dy":  safe(float(np.nanstd(flow_tile[..., 1]))),
        }


def save_variant(
    tile_dir: Path,
    ref_tile: np.ndarray,
    def_tile: np.ndarray,
    flow_tile: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """Persist one augmented tile variant to its own directory."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ref_tile, mode="L").save(tile_dir / "ref.tif")
    Image.fromarray(def_tile, mode="L").save(tile_dir / "def.tif")
    np.save(tile_dir / "flow.npy", flow_tile)
    with (tile_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def _save_augmented_variants(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    pair_out: Path,
    ref_idx: int,
    def_idx: int,
    tile_index: int,
    position: Tuple[int, int],
    valid_ratio: float,
    ref_tile: np.ndarray,
    def_tile: np.ndarray,
    flow_tile: np.ndarray,
) -> int:
    """Write the 8 augmented variants of one tile; return count written."""
    y0, x0 = position
    written = 0
    for rot in (0, 90, 180, 270):
        for flip in (False, True):
            ar, ad, af = apply_augmentation(
                ref_tile, def_tile, flow_tile, rot, flip,
            )
            stats = tile_stats(af)
            meta = {
                "pair": f"f{ref_idx:04d}_vs_f{def_idx:04d}",
                "tile_index": tile_index,
                "original_position": [int(y0), int(x0)],
                "rotation_deg": rot,
                "flip_horizontal": flip,
                "valid_ratio": round(valid_ratio, 4),
                **{k: round(v, 6) for k, v in stats.items()},
            }
            vname = f"tile_{tile_index:04d}_rot{rot:03d}_flip{int(flip)}"
            save_variant(pair_out / vname, ar, ad, af, meta)
            written += 1
    return written


def process_pair(  # pylint: disable=too-many-locals,too-many-statements
    pair_name: str,
    pair_dir: Path,
    images_dir: Path,
    output_dir: Path,
    config: TileConfig,
) -> Optional[Dict[str, Any]]:
    """Load, sample, filter, augment, and save tiles for one image pair.

    Returns a summary dict (with internal accumulator keys prefixed ``_``)
    or None on error.
    """
    m = PAIR_DIR_RE.match(pair_name)
    if m is None:
        return None
    ref_idx, def_idx = int(m.group(1)), int(m.group(2))

    # Load images
    ref_img = load_grayscale(images_dir / f"frame_{ref_idx:04d}.tif")
    def_img = load_grayscale(images_dir / f"frame_{def_idx:04d}.tif")
    img_shape = (ref_img.shape[0], ref_img.shape[1])

    # Load DICe displacement field
    csv_path = find_solution_csv(pair_dir)
    if csv_path is None:
        LOG.error("No DICe CSV in %s — skipping", pair_dir)
        return None
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    LOG.info("  Loading %s (%.1f MB) ...", csv_path.name, size_mb)
    cols = load_dice_csv(csv_path)
    flow = build_dense_flow(cols, img_shape)
    del cols

    # Sample unique random tile positions in the reliable zone
    positions = sample_positions(img_shape, pair_name, config)
    pair_out = output_dir / pair_name

    kept = 0
    rejected = 0
    # Per-pair accumulators (original tiles, pre-augmentation)
    sum_dx = 0.0
    sum_dy = 0.0
    sum_sq_dx = 0.0
    sum_sq_dy = 0.0
    n_valid_total = 0

    for tidx, (y0, x0) in enumerate(positions):
        ft = flow[y0:y0 + config.tile_size,
                  x0:x0 + config.tile_size].copy()
        vr = tile_valid_ratio(ft)
        if vr < config.min_valid_ratio:
            rejected += 1
            continue

        rt = ref_img[y0:y0 + config.tile_size,
                     x0:x0 + config.tile_size].copy()
        dt = def_img[y0:y0 + config.tile_size,
                     x0:x0 + config.tile_size].copy()

        # Pre-augmentation stats: aggregated on the raw DICe tile before any
        # rotation or flip.  Under 8-way D4 symmetry the augmented mean_dx
        # and mean_dy collapse to ~0, erasing the directional-bias signal;
        # Dr. Chen's review depends on this signal, so stats must live here.
        valid = np.isfinite(ft[..., 0])
        nv = int(valid.sum())
        if nv > 0:
            dx_v = ft[valid, 0].astype(np.float64)
            dy_v = ft[valid, 1].astype(np.float64)
            sum_dx += float(dx_v.sum())
            sum_dy += float(dy_v.sum())
            sum_sq_dx += float((dx_v ** 2).sum())
            sum_sq_dy += float((dy_v ** 2).sum())
            n_valid_total += nv

        kept += _save_augmented_variants(
            pair_out, ref_idx, def_idx, tidx, (y0, x0), vr, rt, dt, ft,
        )

    n_surviving = len(positions) - rejected
    if rejected > len(positions) // 2:
        LOG.warning(
            "  %s: %d / %d candidate tiles rejected (>50%%)",
            pair_name, rejected, len(positions),
        )

    # Compute per-pair displacement statistics
    if n_valid_total > 0:
        mean_dx = sum_dx / n_valid_total
        mean_dy = sum_dy / n_valid_total
        var_dx = max(0.0, sum_sq_dx / n_valid_total - mean_dx ** 2)
        var_dy = max(0.0, sum_sq_dy / n_valid_total - mean_dy ** 2)
        std_dx = float(np.sqrt(var_dx))
        std_dy = float(np.sqrt(var_dy))
    else:
        mean_dx = mean_dy = std_dx = std_dy = 0.0

    LOG.info(
        "%s: %d/%d tiles x 8 = %d variants  "
        "(mean_dx=%.4f  mean_dy=%.4f)",
        pair_name, n_surviving, len(positions), kept, mean_dx, mean_dy,
    )

    return {
        "pair": pair_name,
        "source_dir": str(pair_dir),
        "tiles_sampled": len(positions),
        "tiles_rejected": rejected,
        "tiles_kept_original": n_surviving,
        "tiles_generated": kept,
        "mean_dx": round(mean_dx, 6),
        "mean_dy": round(mean_dy, 6),
        "std_dx": round(std_dx, 6),
        "std_dy": round(std_dy, 6),
        # Internal accumulators for global-stats aggregation
        "_sum_dx": sum_dx,
        "_sum_dy": sum_dy,
        "_sum_sq_dx": sum_sq_dx,
        "_sum_sq_dy": sum_sq_dy,
        "_n_valid": n_valid_total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate 128x128 training tiles from full-resolution DICe "
            "output, with edge trimming and 8-way augmentation."
        ),
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        dest="results_dirs",
        type=Path,
        metavar="DIR",
        help=(
            "DICe results root containing pair_fXXXX_vs_fYYYY directories. "
            "May be passed multiple times; later roots override earlier ones. "
            "Defaults to the two ESWG007 ref60 roots if omitted."
        ),
    )
    parser.add_argument(
        "--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
        help="Directory containing frame_NNNN.tif source images.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output root for generated tiles.",
    )
    parser.add_argument(
        "--ref-frame", type=int, default=DEFAULT_REF_FRAME,
        help="Expected reference frame number shared by every pair.",
    )
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="Square tile size in pixels.",
    )
    parser.add_argument(
        "--edge-trim", type=int, default=DEFAULT_EDGE_TRIM,
        help="Border width trimmed from all four image edges.",
    )
    parser.add_argument(
        "--tiles-per-pair", type=int, default=DEFAULT_TILES_PER_PAIR,
        help="Unique random crop positions sampled per pair before filtering.",
    )
    parser.add_argument(
        "--min-valid-ratio", type=float, default=DEFAULT_MIN_VALID_RATIO,
        help="Minimum finite-flow coverage required to keep a sampled tile.",
    )
    parser.add_argument(
        "--only-pair", type=str, default=None,
        metavar="DIRNAME",
        help="Optional single pair directory name (e.g. pair_f0060_vs_f0061).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite --output-dir if it already exists and is non-empty.",
    )
    return parser.parse_args(argv)


def validate_config(config: TileConfig) -> Optional[str]:
    """Return an error message if *config* is invalid, else None."""
    if config.tile_size <= 0:
        return "tile_size must be positive"
    if config.edge_trim < 0:
        return "edge_trim must be non-negative"
    if config.tiles_per_pair <= 0:
        return "tiles_per_pair must be positive"
    if not 0.0 <= config.min_valid_ratio <= 1.0:
        return "min_valid_ratio must be within [0, 1]"
    return None


def ensure_output_dir(output_dir: Path, force: bool) -> None:
    """Create or clear *output_dir*.

    Raises :class:`FileExistsError` if it exists and is non-empty and
    *force* is False.  With *force*, an existing non-empty directory is
    removed and recreated.
    """
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not force:
                raise FileExistsError(
                    f"Output directory not empty: {output_dir}. "
                    f"Use --force to overwrite."
                )
            LOG.warning("Overwriting non-empty output directory: %s",
                        output_dir)
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
        # else: empty dir is fine, leave as-is
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _aggregate_global_stats(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Pool per-pair accumulators into global dx/dy mean and std."""
    g_sum_dx = sum(r["_sum_dx"] for r in results)
    g_sum_dy = sum(r["_sum_dy"] for r in results)
    g_sum_sq_dx = sum(r["_sum_sq_dx"] for r in results)
    g_sum_sq_dy = sum(r["_sum_sq_dy"] for r in results)
    g_n = sum(r["_n_valid"] for r in results)
    if g_n <= 0:
        return {"mean_dx": 0.0, "mean_dy": 0.0, "std_dx": 0.0, "std_dy": 0.0}
    mean_dx = g_sum_dx / g_n
    mean_dy = g_sum_dy / g_n
    std_dx = float(np.sqrt(max(0.0, g_sum_sq_dx / g_n - mean_dx ** 2)))
    std_dy = float(np.sqrt(max(0.0, g_sum_sq_dy / g_n - mean_dy ** 2)))
    return {
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "std_dx": std_dx,
        "std_dy": std_dy,
    }


def write_summary(
    output_dir: Path,
    results_dirs: List[Path],
    config: TileConfig,
    results: List[Dict[str, Any]],
    elapsed: float,
) -> Dict[str, int]:
    """Write summary.json and return totals for the final log line."""
    per_pair_public = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ]
    total_generated = sum(r["tiles_generated"] for r in results)
    total_rejected = sum(r["tiles_rejected"] for r in results)
    global_stats = _aggregate_global_stats(results)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "source_dice_dirs": [str(d) for d in results_dirs],
        "num_pairs": len(results),
        "tile_size": config.tile_size,
        "edge_trim": config.edge_trim,
        "tiles_per_pair_requested": config.tiles_per_pair,
        "min_valid_ratio": config.min_valid_ratio,
        "augmentations_per_tile": 8,
        "total_tiles_generated": total_generated,
        "total_tiles_rejected_quality": total_rejected,
        "elapsed_seconds": round(elapsed, 1),
        "global_stats": {k: round(v, 6) for k, v in global_stats.items()},
        "per_pair_stats": per_pair_public,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return {
        "total_generated": total_generated,
        "total_rejected": total_rejected,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args(argv)
    config = TileConfig(
        tile_size=args.tile_size,
        edge_trim=args.edge_trim,
        tiles_per_pair=args.tiles_per_pair,
        min_valid_ratio=args.min_valid_ratio,
    )
    err = validate_config(config)
    if err is not None:
        LOG.error("Invalid config: %s", err)
        return 1

    if not args.images_dir.is_dir():
        LOG.error("Images directory not found: %s", args.images_dir)
        return 1

    # --- Sanity check augmentation before touching real data ---
    try:
        sanity_test_augmentation()
    except AssertionError as exc:
        LOG.error("Augmentation sanity test FAILED: %s", exc)
        return 1

    # --- Discover image pairs ---
    results_dirs = (
        list(args.results_dirs) if args.results_dirs
        else list(DEFAULT_RESULTS_DIRS)
    )
    pairs = discover_pairs(results_dirs, args.ref_frame, args.only_pair)
    if not pairs:
        LOG.error("No pair directories found")
        return 1
    LOG.info("Found %d pairs to process", len(pairs))

    # --- Prepare output directory (may rmtree on --force) ---
    try:
        ensure_output_dir(args.output_dir, args.force)
    except FileExistsError as exc:
        LOG.error("%s", exc)
        return 1

    # --- Process every pair ---
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    for i, (pair_name, pair_dir) in enumerate(pairs.items(), 1):
        LOG.info("[%d/%d] Processing %s ...", i, len(pairs), pair_name)
        info = process_pair(
            pair_name, pair_dir, args.images_dir, args.output_dir, config,
        )
        if info is not None:
            results.append(info)
    elapsed = time.time() - t0

    totals = write_summary(args.output_dir, results_dirs, config,
                           results, elapsed)

    # --- Final report ---
    LOG.info("=" * 60)
    LOG.info("Sanity tests:          PASSED")
    LOG.info("Total tiles generated: %d", totals["total_generated"])
    LOG.info("Total tiles rejected:  %d", totals["total_rejected"])
    LOG.info("Time elapsed:          %.1f s", elapsed)
    LOG.info("Output:                %s/", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
