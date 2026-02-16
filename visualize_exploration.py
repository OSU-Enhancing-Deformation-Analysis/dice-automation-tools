#!/usr/bin/env python3
"""
Visualize DICe exploration results for Dr. Chen presentation.
Shows: success/fail spatial map, displacement field, and interval comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

# ── Config ──────────────────────────────────────────────────────────
BASE = Path.home() / "Documents/Capstone/dice-automation-tools"
RESULTS = BASE / "exploration_results"
IMG_DIR = BASE / "processed_datasets/A6_Video/preprocessed"
OUT_DIR = BASE / "exploration_visualizations"
OUT_DIR.mkdir(exist_ok=True)

# Intervals that completed successfully
INTERVALS = {
    "pair_f0001_vs_f0002": 1,
    "pair_f0001_vs_f0010": 9,
    "pair_f0001_vs_f0015": 14,
    "pair_f0001_vs_f0031": 30,
}

FLAG_SUCCESS = 4.0  # FEATURE_MATCHING success
FLAG_FAIL = 11.0


def load_dice_result(pair_name):
    """Load DICe solution file for a given pair."""
    folder = RESULTS / pair_name
    csvs = sorted(folder.glob("DICe_solution_*.txt"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    return df


def load_reference_image():
    """Try to load the reference SEM image for background."""
    try:
        from PIL import Image
        img_path = IMG_DIR / "frame_0001.tif"
        if img_path.exists():
            return np.array(Image.open(img_path))
    except ImportError:
        pass
    return None


def plot_success_fail_map(df, interval, ref_img=None):
    """Plot spatial distribution of successful vs failed correlation points."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if ref_img is not None:
        ax.imshow(ref_img, cmap='gray', alpha=0.5)

    fail = df[df['STATUS_FLAG'] == FLAG_FAIL]
    ok = df[df['STATUS_FLAG'] == FLAG_SUCCESS]

    # Plot failed points (light, transparent)
    ax.scatter(fail['COORDINATE_X'], fail['COORDINATE_Y'],
               c='red', s=0.01, alpha=0.05, label=f'Failed (FLAG=11): {len(fail):,}')
    # Plot successful points
    ax.scatter(ok['COORDINATE_X'], ok['COORDINATE_Y'],
               c='lime', s=0.1, alpha=0.3, label=f'Success (FLAG=4): {len(ok):,}')

    total = len(df)
    rate = len(ok) / total * 100
    ax.set_title(f'DICe Correlation Success Map — {interval}-frame interval\n'
                 f'Success: {len(ok):,} / {total:,} ({rate:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(loc='upper right', fontsize=10, markerscale=50)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    fig.tight_layout()
    out = OUT_DIR / f"success_map_interval_{interval}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_displacement_field(df, interval, ref_img=None):
    """Plot displacement vectors for successful points."""
    ok = df[df['STATUS_FLAG'] == FLAG_SUCCESS].copy()
    if len(ok) == 0:
        print(f"  No successful points for interval {interval}, skipping displacement plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Displacement magnitude
    ok['DISP_MAG'] = np.sqrt(ok['DISPLACEMENT_X']**2 + ok['DISPLACEMENT_Y']**2)

    for ax_idx, (col, title, cmap) in enumerate([
        ('DISPLACEMENT_X', 'Displacement X (px)', 'RdBu_r'),
        ('DISPLACEMENT_Y', 'Displacement Y (px)', 'RdBu_r'),
        ('DISP_MAG', 'Displacement Magnitude (px)', 'hot'),
    ]):
        ax = axes[ax_idx]
        if ref_img is not None:
            ax.imshow(ref_img, cmap='gray', alpha=0.3)

        vals = ok[col].values
        if col == 'DISP_MAG':
            vmin, vmax = 0, np.percentile(vals, 99)
        else:
            vlim = max(abs(np.percentile(vals, 1)), abs(np.percentile(vals, 99)))
            vmin, vmax = -vlim, vlim

        sc = ax.scatter(ok['COORDINATE_X'], ok['COORDINATE_Y'],
                        c=vals, s=0.3, alpha=0.5, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

    fig.suptitle(f'Displacement Field — {interval}-frame interval  '
                 f'({len(ok):,} successful points)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"displacement_field_interval_{interval}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_strain_field(df, interval, ref_img=None):
    """Plot strain components for successful points."""
    ok = df[df['STATUS_FLAG'] == FLAG_SUCCESS].copy()
    if len(ok) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax_idx, (col, title) in enumerate([
        ('VSG_STRAIN_XX', 'Strain εxx'),
        ('VSG_STRAIN_YY', 'Strain εyy'),
        ('VSG_STRAIN_XY', 'Strain εxy'),
    ]):
        ax = axes[ax_idx]
        if ref_img is not None:
            ax.imshow(ref_img, cmap='gray', alpha=0.3)

        vals = ok[col].values
        vlim = max(abs(np.percentile(vals, 2)), abs(np.percentile(vals, 98)))
        if vlim == 0:
            vlim = 1e-6

        sc = ax.scatter(ok['COORDINATE_X'], ok['COORDINATE_Y'],
                        c=vals, s=0.3, alpha=0.5, cmap='RdBu_r',
                        vmin=-vlim, vmax=vlim)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

    fig.suptitle(f'Strain Field — {interval}-frame interval  '
                 f'({len(ok):,} successful points)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"strain_field_interval_{interval}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_interval_comparison(all_data):
    """Bar chart comparing success rates across intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    intervals = sorted(all_data.keys())
    success_counts = [all_data[i]['success'] for i in intervals]
    total = all_data[intervals[0]]['total']
    rates = [c / total * 100 for c in success_counts]

    # Success rate bar chart
    ax = axes[0]
    bars = ax.bar(range(len(intervals)), rates, color=['#2ecc71', '#3498db', '#e67e22', '#e74c3c'])
    ax.set_xticks(range(len(intervals)))
    ax.set_xticklabels([f'{i} frame{"s" if i > 1 else ""}' for i in intervals])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Correlation Success Rate by Frame Interval', fontweight='bold')
    for bar, rate, count in zip(bars, rates, success_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{rate:.1f}%\n({count:,})', ha='center', va='bottom', fontsize=10)

    # Mean displacement magnitude for successful points
    ax = axes[1]
    mean_disps = [all_data[i]['mean_disp'] for i in intervals]
    bars = ax.bar(range(len(intervals)), mean_disps, color=['#2ecc71', '#3498db', '#e67e22', '#e74c3c'])
    ax.set_xticks(range(len(intervals)))
    ax.set_xticklabels([f'{i} frame{"s" if i > 1 else ""}' for i in intervals])
    ax.set_ylabel('Mean Displacement Magnitude (px)')
    ax.set_title('Mean Displacement of Successful Points', fontweight='bold')
    for bar, val in zip(bars, mean_disps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    fig.suptitle('DICe Frame Interval Exploration — A6_Video Dataset\n'
                 f'(step_size=1, subset_size=35, sssig_threshold=144)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = OUT_DIR / "interval_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sigma_gamma_distribution(df, interval):
    """Plot SIGMA and GAMMA distributions for successful points."""
    ok = df[df['STATUS_FLAG'] == FLAG_SUCCESS]
    if len(ok) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(ok['SIGMA'], bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_xlabel('SIGMA (displacement uncertainty)')
    ax.set_ylabel('Count')
    ax.set_title(f'SIGMA Distribution — {interval}-frame interval', fontweight='bold')
    ax.axvline(ok['SIGMA'].median(), color='red', ls='--', label=f'Median: {ok["SIGMA"].median():.4f}')
    ax.legend()

    ax = axes[1]
    ax.hist(ok['GAMMA'], bins=100, color='coral', edgecolor='none', alpha=0.8)
    ax.set_xlabel('GAMMA (matching quality, lower=better)')
    ax.set_ylabel('Count')
    ax.set_title(f'GAMMA Distribution — {interval}-frame interval', fontweight='bold')
    ax.axvline(ok['GAMMA'].median(), color='red', ls='--', label=f'Median: {ok["GAMMA"].median():.4f}')
    ax.legend()

    fig.tight_layout()
    out = OUT_DIR / f"quality_distribution_interval_{interval}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading reference image...")
    ref_img = load_reference_image()
    if ref_img is not None:
        print(f"  Loaded: {ref_img.shape}")
    else:
        print("  No reference image found (plots will have no background)")

    all_data = {}

    for pair_name, interval in sorted(INTERVALS.items(), key=lambda x: x[1]):
        print(f"\n{'='*60}")
        print(f"Processing: {pair_name} (interval={interval} frames)")
        print(f"{'='*60}")

        df = load_dice_result(pair_name)
        if df is None:
            print("  No data found, skipping")
            continue

        ok = df[df['STATUS_FLAG'] == FLAG_SUCCESS]
        disp_mag = np.sqrt(ok['DISPLACEMENT_X']**2 + ok['DISPLACEMENT_Y']**2)

        all_data[interval] = {
            'success': len(ok),
            'total': len(df),
            'mean_disp': disp_mag.mean() if len(ok) > 0 else 0,
        }

        print(f"  Total points: {len(df):,}")
        print(f"  Successful: {len(ok):,} ({len(ok)/len(df)*100:.1f}%)")
        if len(ok) > 0:
            print(f"  Mean displacement: {disp_mag.mean():.4f} px")
            print(f"  SIGMA median: {ok['SIGMA'].median():.4f}")
            print(f"  GAMMA median: {ok['GAMMA'].median():.4f}")

        plot_success_fail_map(df, interval, ref_img)
        plot_displacement_field(df, interval, ref_img)
        plot_strain_field(df, interval, ref_img)
        plot_sigma_gamma_distribution(df, interval)

    if all_data:
        print(f"\n{'='*60}")
        print("Generating comparison chart...")
        print(f"{'='*60}")
        plot_interval_comparison(all_data)

    print(f"\nAll visualizations saved to: {OUT_DIR}")
    print("Done!")