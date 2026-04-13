#!/usr/bin/env python3
"""
visualize_tiles.py
Render overview and sample figures for a training_tiles_128 output directory.

Produces two PNGs:

  1. --summary-fig (default: tile_summary.png)
     Left bar chart: tiles kept per pair.
     Right bar chart: mean flow magnitude per pair (pixels), averaged over
     every kept tile's dense displacement field.

  2. --samples-fig (default: tile_samples.png)
     The pair with the most kept tiles, showing the top --n-samples tiles
     ranked by valid_ratio. One row per tile with four columns:
     ref image, def image, dx heatmap, dy heatmap.

Usage:
    python3 visualize_tiles.py
    python3 visualize_tiles.py \\
        --tiles-dir training_tiles_128 \\
        --summary-fig tile_summary.png \\
        --samples-fig tile_samples.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2  # pylint: disable=import-error,no-name-in-module
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# cv2 is a C extension and pylint cannot introspect its members.
# pylint: disable=no-member


DEFAULT_TILES_DIR = Path("training_tiles_128")
DEFAULT_SUMMARY_FIG = Path("tile_summary.png")
DEFAULT_SAMPLES_FIG = Path("tile_samples.png")
DEFAULT_N_SAMPLES = 4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize training_tiles_128 tile outputs with matplotlib.",
    )
    parser.add_argument(
        "--tiles-dir", type=Path, default=DEFAULT_TILES_DIR,
        help="Root directory produced by generate_tiles.py.",
    )
    parser.add_argument(
        "--summary-fig", type=Path, default=DEFAULT_SUMMARY_FIG,
        help="Output path for the overview bar-chart figure.",
    )
    parser.add_argument(
        "--samples-fig", type=Path, default=DEFAULT_SAMPLES_FIG,
        help="Output path for the tile samples figure.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help="Number of top-quality sample tiles to show in the samples figure.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def short_pair_label(pair_id: str) -> str:
    """Abbreviate 'pair_f0060_vs_f0061' to 'f60-f61' for compact tick labels."""
    parts = pair_id.replace("pair_", "").split("_vs_")
    if len(parts) != 2:
        return pair_id
    left = parts[0].lstrip("f").lstrip("0") or "0"
    right = parts[1].lstrip("f").lstrip("0") or "0"
    return f"f{left}-f{right}"


def compute_pair_mean_magnitude(
    tiles_dir: Path, pair_id: str, tiles: List[Dict[str, Any]],
) -> float:
    """Return the mean flow magnitude across all kept tiles in a pair (pixels).

    The magnitude is computed per pixel (sqrt(dx^2 + dy^2)), averaged within each
    tile with nanmean (NaN pixels are skipped), then averaged across tiles.
    """
    if not tiles:
        return 0.0
    per_tile_means: List[float] = []
    for tile in tiles:
        flow_path = tiles_dir / pair_id / "flow" / f"{tile['name']}.npy"
        flow = np.load(flow_path)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        with np.errstate(invalid="ignore"):
            per_tile_means.append(float(np.nanmean(magnitude)))
    return float(np.nanmean(per_tile_means))


def _annotate_bars(ax: plt.Axes, values: List[float], fmt: str) -> None:
    """Write each bar's value above its bar on a bar-chart axis."""
    for rect, value in zip(ax.patches, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def render_summary_figure(
    pair_ids: List[str],
    tiles_kept: List[int],
    mean_magnitudes: List[float],
    output_path: Path,
) -> None:
    """Write the overview bar-chart figure (tiles kept + mean magnitude)."""
    labels = [short_pair_label(p) for p in pair_ids]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    ax_left.bar(labels, tiles_kept, color="steelblue")
    ax_left.set_title("Tiles kept per pair")
    ax_left.set_xlabel("Pair")
    ax_left.set_ylabel("Tiles kept")
    ax_left.tick_params(axis="x", rotation=45)
    _annotate_bars(ax_left, [float(v) for v in tiles_kept], "{:.0f}")

    ax_right.bar(labels, mean_magnitudes, color="coral")
    ax_right.set_title("Mean flow magnitude per pair")
    ax_right.set_xlabel("Pair")
    ax_right.set_ylabel("|displacement| (pixels)")
    ax_right.tick_params(axis="x", rotation=45)
    _annotate_bars(ax_right, mean_magnitudes, "{:.2f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_tile_arrays(
    tiles_dir: Path, pair_id: str, tile_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ref image, def image, and flow array for a single tile."""
    pair_root = tiles_dir / pair_id
    ref = cv2.imread(
        str(pair_root / "ref" / f"{tile_name}.tif"), cv2.IMREAD_UNCHANGED,
    )
    dfm = cv2.imread(
        str(pair_root / "def" / f"{tile_name}.tif"), cv2.IMREAD_UNCHANGED,
    )
    flow = np.load(pair_root / "flow" / f"{tile_name}.npy")
    return ref, dfm, flow


def _hide_ticks(ax: plt.Axes) -> None:
    """Remove tick marks and labels from an axis used for an image."""
    ax.set_xticks([])
    ax.set_yticks([])


# pylint: disable=too-many-arguments,too-many-positional-arguments
def render_tile_row(
    axes_row: np.ndarray,
    tile: Dict[str, Any],
    tiles_dir: Path,
    pair_id: str,
    fig: Figure,
) -> None:
    """Render one tile row (ref, def, dx, dy) into the given axes."""
    ref_img, def_img, flow = load_tile_arrays(tiles_dir, pair_id, tile["name"])
    dx = flow[..., 0]
    dy = flow[..., 1]
    with np.errstate(invalid="ignore"):
        vmax = float(np.nanmax(np.abs(flow))) if np.isfinite(flow).any() else 1.0
    vmax = max(vmax, 1e-6)

    axes_row[0].imshow(ref_img, cmap="gray")
    axes_row[0].set_title(f"{tile['name']} ref\nvalid={tile['valid_ratio']:.2f}")
    _hide_ticks(axes_row[0])

    axes_row[1].imshow(def_img, cmap="gray")
    axes_row[1].set_title("def")
    _hide_ticks(axes_row[1])

    im_dx = axes_row[2].imshow(dx, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes_row[2].set_title(f"dx (mean={tile['mean_disp_x']:.2f})")
    _hide_ticks(axes_row[2])
    fig.colorbar(im_dx, ax=axes_row[2], fraction=0.046, pad=0.04)

    im_dy = axes_row[3].imshow(dy, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes_row[3].set_title(f"dy (mean={tile['mean_disp_y']:.2f})")
    _hide_ticks(axes_row[3])
    fig.colorbar(im_dy, ax=axes_row[3], fraction=0.046, pad=0.04)
# pylint: enable=too-many-arguments,too-many-positional-arguments


def render_samples_figure(
    tiles_dir: Path,
    pair_id: str,
    tiles: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write the multi-row tile samples figure for one pair."""
    rows = len(tiles)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.2 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    for i, tile in enumerate(tiles):
        render_tile_row(axes[i], tile, tiles_dir, pair_id, fig)
    fig.suptitle(
        f"Top {rows} tiles from {short_pair_label(pair_id)} by valid_ratio",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def select_best_pair(by_pair: List[Dict[str, Any]]) -> str:
    """Return the pair_id with the largest tiles_kept value."""
    if not by_pair:
        raise ValueError("summary.json has no pairs")
    return max(by_pair, key=lambda entry: entry["tiles_kept"])["pair_id"]


def select_top_tiles(
    tiles: List[Dict[str, Any]], n: int,
) -> List[Dict[str, Any]]:
    """Return the top-n tiles by valid_ratio (descending)."""
    ordered = sorted(tiles, key=lambda t: t["valid_ratio"], reverse=True)
    return ordered[:n]


def collect_pair_stats(
    tiles_dir: Path, by_pair: List[Dict[str, Any]],
) -> Tuple[List[str], List[int], List[float], Dict[str, List[Dict[str, Any]]]]:
    """Gather the per-pair arrays needed for the summary figure."""
    pair_ids: List[str] = []
    tiles_kept: List[int] = []
    mean_magnitudes: List[float] = []
    pair_tiles: Dict[str, List[Dict[str, Any]]] = {}
    for entry in by_pair:
        pair_id = entry["pair_id"]
        metadata = load_json(tiles_dir / pair_id / "metadata.json")
        pair_tiles[pair_id] = metadata["tiles"]
        pair_ids.append(pair_id)
        tiles_kept.append(int(entry["tiles_kept"]))
        mean_magnitudes.append(
            compute_pair_mean_magnitude(tiles_dir, pair_id, metadata["tiles"])
        )
    return pair_ids, tiles_kept, mean_magnitudes, pair_tiles


def main() -> int:
    """Entry point."""
    args = parse_args()
    if not args.tiles_dir.is_dir():
        print(f"ERROR: tiles dir not found: {args.tiles_dir}", file=sys.stderr)
        return 1

    summary_path = args.tiles_dir / "summary.json"
    if not summary_path.is_file():
        print(f"ERROR: missing summary.json at {summary_path}", file=sys.stderr)
        return 1

    summary = load_json(summary_path)
    by_pair = summary.get("by_pair", [])
    if not by_pair:
        print("ERROR: summary.json has no pairs", file=sys.stderr)
        return 1

    pair_ids, tiles_kept, mean_magnitudes, pair_tiles = collect_pair_stats(
        args.tiles_dir, by_pair,
    )
    render_summary_figure(pair_ids, tiles_kept, mean_magnitudes, args.summary_fig)

    best_pair_id = select_best_pair(by_pair)
    best_tiles = select_top_tiles(pair_tiles[best_pair_id], args.n_samples)
    if not best_tiles:
        print(
            f"WARNING: pair {best_pair_id} has no tiles; skipping samples figure",
            file=sys.stderr,
        )
    else:
        render_samples_figure(
            args.tiles_dir, best_pair_id, best_tiles, args.samples_fig,
        )

    print(f"Wrote {args.summary_fig} and {args.samples_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
