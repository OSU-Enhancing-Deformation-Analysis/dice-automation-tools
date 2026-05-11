"""
Render the systematic-crop merged absolute txt files as a GIF showing
displacement field progression across frame pairs.
Each frame: dx + dy heatmaps for one pair, fixed colorscale across all frames
so deformation growth is visible.
"""
import re
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = Path("demo_eswg007_systematic_merged_absolute")
OUTPUT_GIF = Path("systematic_coverage.gif")

H, W = 883, 1024

# Fixed color scales (from training set stats)
DX_VMIN, DX_VMAX = -10, 10
DY_VMIN, DY_VMAX = -30, 5

files = sorted(INPUT_DIR.glob("DICe_solution_pair_*.txt"))
print(f"Found {len(files)} pair files")

frames_paths = []
tmp_dir = Path("/tmp/gif_frames")
tmp_dir.mkdir(exist_ok=True)

for i, f in enumerate(files):
    # Parse only x, y, dx, dy columns (1, 2, 3, 4)
    data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
    xs = data[:, 0].astype(int)
    ys = data[:, 1].astype(int)
    dxs = data[:, 2]
    dys = data[:, 3]

    img_dx = np.full((H, W), np.nan)
    img_dy = np.full((H, W), np.nan)
    img_dx[ys, xs] = dxs
    img_dy[ys, xs] = dys

    m = re.search(r"pair_f(\d+)_vs_f(\d+)", f.name)
    ref_f = int(m.group(1))
    def_f = int(m.group(2))
    interval = def_f - ref_f

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=100)
    im0 = axes[0].imshow(img_dx, cmap="RdBu_r", vmin=DX_VMIN, vmax=DX_VMAX)
    axes[0].set_title("Δx displacement (px)", fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.04)

    im1 = axes[1].imshow(img_dy, cmap="RdBu_r", vmin=DY_VMIN, vmax=DY_VMAX)
    axes[1].set_title("Δy displacement (px)", fontsize=12)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.04)

    fig.suptitle(
        f"ESWG007  ref f{ref_f:04d}  →  def f{def_f:04d}    "
        f"(interval = {interval:>2d} frames,   {len(xs):,} systematic pixels)",
        fontsize=13,
    )
    plt.tight_layout()
    out_png = tmp_dir / f"frame_{i:03d}.png"
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    frames_paths.append(out_png)
    print(f"  [{i+1}/{len(files)}] {f.name} -> {out_png.name}")

# Assemble GIF
print(f"\nAssembling {len(frames_paths)} frames...")
images = [Image.open(p).convert("RGB") for p in frames_paths]
images[0].save(OUTPUT_GIF, save_all=True, append_images=images[1:], duration=500, loop=0, optimize=True)
size_mb = OUTPUT_GIF.stat().st_size / 1e6
print(f"Saved {OUTPUT_GIF}  ({size_mb:.1f} MB)")
