#!/usr/bin/env python3
"""
visualize_tiles.py — summary plots for a training_tiles_128_v2 output root.

Produces three PNGs:

  1. tile_summary.png — single-panel per-pair bar chart of |mean flow|,
     sqrt(mean_dx^2 + mean_dy^2) in pixels.  Pairs ordered by
     deformed-frame number.

  2. tile_statistics.png — global dx and dy histograms over every original
     (rot=0, flip=0) tile, with mean / ±1 std overlays drawn from
     summary.json's global_stats.  Histograms are restricted to the
     pre-augmentation variants so the directional-bias signal
     (mean_dy ~= -8.8 for ESWG007) stays visible; the 8-way D4 augmentation
     would otherwise collapse mean_dx and mean_dy to ~0.

  3. tile_samples.png — the 8 augmentation variants (4 rotations x 2 flips)
     of a single high-displacement tile, arranged as 8 rows x 4 columns
     (ref, def, dx, dy).  Rotating (dx, dy) under D4 swaps and negates
     components, which is most visible on a tile whose original mean flow
     is large; the showcased tile is picked automatically as the one with
     the largest sqrt(mean_dx^2 + mean_dy^2) across all pre-augmentation
     tiles.

Usage:
    python3 visualize_tiles.py
    python3 visualize_tiles.py \\
        --tiles-dir training_tiles_128_v2 --output-dir .
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_TILES_DIR = Path("training_tiles_128_v2")
DEFAULT_OUTPUT_DIR = Path(".")
ORIGINAL_TILE_GLOB = "pair_*/tile_*_rot000_flip0/flow.npy"
ORIGINAL_META_GLOB = "pair_*/tile_*_rot000_flip0/metadata.json"
PAIR_RE = re.compile(r"pair_f(\d+)_vs_f(\d+)")
TILE_DIR_RE = re.compile(r"tile_(\d+)_rot000_flip0")

# 8 augmentation variants produced by generate_tiles.py, in a
# row-friendly reading order: identity first, then flip, then each
# rotation pair.
AUGMENTATION_VARIANTS: Tuple[Tuple[int, int], ...] = (
    (0, 0), (0, 1),
    (90, 0), (90, 1),
    (180, 0), (180, 1),
    (270, 0), (270, 1),
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize training_tiles_128_v2 dataset statistics.",
    )
    parser.add_argument(
        "--tiles-dir", type=Path, default=DEFAULT_TILES_DIR,
        help="Root produced by generate_tiles.py (must contain summary.json).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write tile_summary.png and tile_statistics.png.",
    )
    return parser.parse_args(argv)


def short_pair_label(pair_id: str) -> str:
    """Abbreviate ``pair_f0060_vs_f0065`` -> ``f60-f65`` for tick labels."""
    m = PAIR_RE.match(pair_id)
    if m is None:
        return pair_id
    return f"f{int(m.group(1))}-f{int(m.group(2))}"


def pair_def_frame(pair_id: str) -> int:
    """Return the deformed-frame number from a pair directory name."""
    m = PAIR_RE.match(pair_id)
    return int(m.group(2)) if m else 0


def annotate_bars(ax: plt.Axes, values: List[float], fmt: str) -> None:
    """Write each bar's value just above its bar on a bar-chart axis."""
    for rect, value in zip(ax.patches, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            fmt.format(value),
            ha="center", va="bottom", fontsize=8,
        )


def render_summary_figure(
    per_pair: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write tile_summary.png: per-pair |mean flow| magnitude bar chart."""
    ordered = sorted(per_pair, key=lambda p: pair_def_frame(p["pair"]))
    labels = [short_pair_label(p["pair"]) for p in ordered]
    magnitudes = [
        math.sqrt(float(p["mean_dx"]) ** 2 + float(p["mean_dy"]) ** 2)
        for p in ordered
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(labels, magnitudes, color="tomato")
    ax.set_title(
        f"Per-pair mean flow magnitude across {len(ordered)} frame pairs "
        "(pre-augmentation)",
        fontsize=14,
    )
    ax.set_xlabel("Pair")
    ax.set_ylabel("sqrt(mean_dx^2 + mean_dy^2)  [pixels]")
    ax.tick_params(axis="x", rotation=45)
    ax.margins(y=0.15)
    annotate_bars(ax, magnitudes, "{:.2f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def collect_flow_pixels(tiles_dir: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Concatenate dx/dy from every pre-augmentation tile.

    Restricts to ``rot000_flip0`` variants — these are the identity
    augmentations, i.e. the raw DICe crops.  NaNs are dropped.
    Returns ``(dx_flat, dy_flat, n_tiles)``.
    """
    files = sorted(tiles_dir.glob(ORIGINAL_TILE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No rot000_flip0 flow files found under {tiles_dir}"
        )
    dx_parts: List[np.ndarray] = []
    dy_parts: List[np.ndarray] = []
    for path in files:
        flow = np.load(path)
        dx = flow[..., 0].ravel()
        dy = flow[..., 1].ravel()
        valid = np.isfinite(dx) & np.isfinite(dy)
        dx_parts.append(dx[valid])
        dy_parts.append(dy[valid])
    return np.concatenate(dx_parts), np.concatenate(dy_parts), len(files)


def _symmetric_range(dx: np.ndarray, dy: np.ndarray) -> Tuple[float, float]:
    """Symmetric x-range covering the 99.5th percentile of |dx| and |dy|."""
    tail = max(
        float(np.percentile(np.abs(dx), 99.5)),
        float(np.percentile(np.abs(dy), 99.5)),
    )
    limit = max(5.0 * math.ceil(tail / 5.0), 5.0)
    return (-limit, limit)


def _draw_component_hist(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    ax: plt.Axes,
    data: np.ndarray,
    mean_v: float,
    std_v: float,
    color: str,
    label: str,
    x_range: Tuple[float, float],
) -> None:
    """Render one histogram subplot with mean / ±std overlays."""
    ax.hist(
        data, bins=120, range=x_range, color=color,
        edgecolor="black", linewidth=0.3,
    )
    ax.axvline(mean_v, color="black", linestyle="-", linewidth=1.5,
               label=f"mean = {mean_v:.3f}")
    ax.axvline(mean_v - std_v, color="black", linestyle="--", linewidth=1)
    ax.axvline(mean_v + std_v, color="black", linestyle="--", linewidth=1,
               label=f"+/-1 std ({std_v:.3f})")
    ax.set_title(f"{label} distribution")
    ax.set_xlabel(f"{label}  [pixels]")
    ax.set_ylabel("pixel count")
    ax.set_xlim(x_range)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.98, 0.97,
        f"mean = {mean_v:.3f}\nstd  = {std_v:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def render_statistics_figure(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dx: np.ndarray,
    dy: np.ndarray,
    global_stats: Dict[str, float],
    n_tiles: int,
    n_pairs: int,
    output_path: Path,
) -> None:
    """Write tile_statistics.png: global dx and dy histograms."""
    x_range = _symmetric_range(dx, dy)
    fig, (ax_dx, ax_dy) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    _draw_component_hist(
        ax_dx, dx,
        float(global_stats["mean_dx"]), float(global_stats["std_dx"]),
        "steelblue", "dx", x_range,
    )
    _draw_component_hist(
        ax_dy, dy,
        float(global_stats["mean_dy"]), float(global_stats["std_dy"]),
        "tomato", "dy", x_range,
    )

    fig.suptitle(
        "Global displacement distribution "
        f"(pre-augmentation, {n_tiles} tiles from {n_pairs} pairs)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Augmentation-variants figure
# ---------------------------------------------------------------------------

def find_max_flow_tile(tiles_dir: Path) -> Tuple[str, str, float, float]:
    """Locate the pre-augmentation tile with largest mean flow magnitude.

    Scans every ``pair_*/tile_*_rot000_flip0/metadata.json`` and picks the
    tile maximising sqrt(mean_dx^2 + mean_dy^2).  Returns
    ``(pair_name, tile_index_str, mean_dx, mean_dy)``.
    """
    best_mag = -1.0
    best_pair = ""
    best_tile = ""
    best_dx = 0.0
    best_dy = 0.0
    for meta_path in tiles_dir.glob(ORIGINAL_META_GLOB):
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        mean_dx = float(meta["mean_dx"])
        mean_dy = float(meta["mean_dy"])
        mag = math.sqrt(mean_dx * mean_dx + mean_dy * mean_dy)
        if mag <= best_mag:
            continue
        m = TILE_DIR_RE.match(meta_path.parent.name)
        if m is None:
            continue
        best_mag = mag
        best_pair = meta_path.parent.parent.name
        best_tile = m.group(1)
        best_dx = mean_dx
        best_dy = mean_dy
    if best_mag < 0:
        raise FileNotFoundError(
            f"No rot000_flip0 metadata.json files under {tiles_dir}"
        )
    return best_pair, best_tile, best_dx, best_dy


def load_variant(
    tiles_dir: Path,
    pair_name: str,
    tile_idx: str,
    rotation: int,
    flip: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (ref, def, flow) for one augmentation variant."""
    variant_dir = (
        tiles_dir
        / pair_name
        / f"tile_{tile_idx}_rot{rotation:03d}_flip{flip}"
    )
    ref = np.asarray(Image.open(variant_dir / "ref.tif"))
    def_img = np.asarray(Image.open(variant_dir / "def.tif"))
    flow = np.load(variant_dir / "flow.npy")
    return ref, def_img, flow


def _flow_symmetric_vmax(
    flows: List[np.ndarray],
    floor: float = 1.0,
) -> float:
    """Return a single symmetric color-scale bound shared across flows."""
    peak = 0.0
    for flow in flows:
        finite = flow[np.isfinite(flow)]
        if finite.size == 0:
            continue
        peak = max(peak, float(np.max(np.abs(finite))))
    return max(peak, floor)


def render_samples_figure(  # pylint: disable=too-many-locals
    tiles_dir: Path,
    output_path: Path,
) -> Optional[Tuple[str, str]]:
    """Write tile_samples.png: 8 augmentation variants of one tile.

    Lay out an 8x4 grid.  Rows = augmentation variants (rot x flip).
    Columns = (ref, def, dx, dy).  dx/dy share a single symmetric
    colormap so the component swap/negation under rotation is visible at
    a glance.  Returns (pair_name, tile_idx) on success, else None.
    """
    try:
        pair_name, tile_idx, orig_dx, orig_dy = find_max_flow_tile(tiles_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None

    loaded: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for rotation, flip in AUGMENTATION_VARIANTS:
        ref, def_img, flow = load_variant(
            tiles_dir, pair_name, tile_idx, rotation, flip,
        )
        loaded.append((rotation, flip, ref, def_img, flow))

    vmax = _flow_symmetric_vmax([v[4] for v in loaded])

    fig, axes = plt.subplots(8, 4, figsize=(11, 20))
    fig.suptitle(
        "8 augmentation variants of "
        f"{short_pair_label(pair_name)} / tile_{tile_idx}  "
        f"(original mean_dx={orig_dx:+.2f}, mean_dy={orig_dy:+.2f} px)",
        fontsize=13, y=0.995,
    )

    for row, (rotation, flip, ref, def_img, flow) in enumerate(loaded):
        flip_tag = "flip" if flip else "no-flip"
        identity_tag = "  [identity]" if rotation == 0 and flip == 0 else ""
        axes[row, 0].imshow(ref, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].imshow(def_img, cmap="gray", vmin=0, vmax=255)
        im = axes[row, 2].imshow(
            flow[..., 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        axes[row, 3].imshow(
            flow[..., 1], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )

        mean_dx = float(np.nanmean(flow[..., 0]))
        mean_dy = float(np.nanmean(flow[..., 1]))
        axes[row, 0].set_ylabel(
            f"rot={rotation:>3}\n{flip_tag}{identity_tag}",
            fontsize=10, rotation=0, labelpad=42, va="center",
        )
        if row == 0:
            axes[row, 0].set_title("ref", fontsize=11)
            axes[row, 1].set_title("def", fontsize=11)
        axes[row, 2].set_title(f"dx  (mean={mean_dx:+.2f})", fontsize=10)
        axes[row, 3].set_title(f"dy  (mean={mean_dy:+.2f})", fontsize=10)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.tight_layout(rect=(0.04, 0.02, 0.93, 0.98))
    # Shared colorbar for dx/dy (right-hand side).
    cbar_ax = fig.add_axes((0.945, 0.08, 0.012, 0.84))
    fig.colorbar(im, cax=cbar_ax, label="displacement [pixels]")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return pair_name, tile_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:  # pylint: disable=too-many-locals
    """Entry point."""
    args = parse_args(argv)
    tiles_dir: Path = args.tiles_dir
    if not tiles_dir.is_dir():
        print(f"ERROR: tiles dir not found: {tiles_dir}", file=sys.stderr)
        return 1

    summary_path = tiles_dir / "summary.json"
    if not summary_path.is_file():
        print(f"ERROR: missing summary.json at {summary_path}",
              file=sys.stderr)
        return 1

    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    per_pair = summary.get("per_pair_stats", [])
    global_stats = summary.get("global_stats", {})
    if not per_pair or not global_stats:
        print("ERROR: summary.json missing per_pair_stats or global_stats",
              file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_png = args.output_dir / "tile_summary.png"
    stats_png = args.output_dir / "tile_statistics.png"
    samples_png = args.output_dir / "tile_samples.png"

    render_summary_figure(per_pair, summary_png)

    try:
        dx, dy, n_original = collect_flow_pixels(tiles_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    render_statistics_figure(
        dx, dy, global_stats, n_original, len(per_pair), stats_png,
    )

    sample_info = render_samples_figure(tiles_dir, samples_png)

    total_tiles = int(summary.get("total_tiles_generated", 0))
    sample_tag = (
        f" showcased tile: {sample_info[0]}/tile_{sample_info[1]}."
        if sample_info is not None else ""
    )
    print(
        f"Saved {summary_png.name}, {stats_png.name}, and "
        f"{samples_png.name}. "
        f"Global: mean_dx={global_stats['mean_dx']:.3f}, "
        f"mean_dy={global_stats['mean_dy']:.3f}, "
        f"std_dx={global_stats['std_dx']:.3f}, "
        f"std_dy={global_stats['std_dy']:.3f}. "
        f"{len(per_pair)} pairs, {total_tiles} tiles.{sample_tag}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())