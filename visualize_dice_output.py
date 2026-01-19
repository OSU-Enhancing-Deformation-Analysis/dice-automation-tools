#!/usr/bin/env python3
"""
visualize_dice_output.py
Visualize displacement vectors and strain fields from DICe output files.

Usage:
    # Displacement vectors
    python visualize_dice_output.py DICe_solution_050.txt --scale 10 --show

    # Strain field
    python visualize_dice_output.py DICe_solution_050.txt --strain --strain-component VSG_STRAIN_XX

    # Without background image
    python visualize_dice_output.py DICe_solution_050.txt --no-background -o output.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_dice_output(filepath):
    """
    Load DICe output file, skipping header comments.

    Args:
        filepath: Path to DICe output file (.txt)

    Returns:
        tuple: (DataFrame with data, reference_image_path, deformed_image_path)
    """
    reference_image = None
    deformed_image = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find header info and data start
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('*** Reference image:'):
            reference_image = line.split(':', 1)[1].strip()
        if line.startswith('*** Deformed image:'):
            deformed_image = line.split(':', 1)[1].strip()
        if line.startswith('SUBSET_ID'):
            header_idx = i
            break

    # Read CSV data starting from header
    df = pd.read_csv(filepath, skiprows=header_idx)

    return df, reference_image, deformed_image


def plot_displacement_vectors(df, background_image=None, output_path=None,
                              scale=10, title="Displacement Vectors"):
    """
    Plot displacement vectors as quiver plot with optional background.

    Args:
        df: DataFrame with COORDINATE_X/Y and DISPLACEMENT_X/Y columns
        background_image: Path to background image (optional)
        output_path: Path to save output image (optional)
        scale: Arrow scale factor
        title: Plot title

    Returns:
        tuple: (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Load and display background image
    if background_image and Path(background_image).exists():
        img = Image.open(background_image)
        ax.imshow(img, cmap='gray', alpha=0.7)
        print(f"  Background: {background_image}")
    elif background_image:
        print(f"  Warning: Background image not found: {background_image}")

    x = df['COORDINATE_X'].values
    y = df['COORDINATE_Y'].values
    u = df['DISPLACEMENT_X'].values
    v = df['DISPLACEMENT_Y'].values

    # Calculate magnitude for coloring
    magnitude = np.sqrt(u**2 + v**2)

    # Create quiver plot
    quiver = ax.quiver(x, y, u, v, magnitude,
                       scale=1 / scale, scale_units='xy',
                       cmap='jet', alpha=0.9, width=0.003)

    plt.colorbar(quiver, ax=ax, label='Displacement Magnitude (pixels)')

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def plot_strain_field(df, background_image=None, strain_component='VSG_STRAIN_XX',
                      output_path=None):
    """
    Plot strain field as scatter plot with optional background.

    Args:
        df: DataFrame with coordinate and strain columns
        background_image: Path to background image (optional)
        strain_component: Which strain component to plot
        output_path: Path to save output image (optional)

    Returns:
        tuple: (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Load and display background image
    if background_image and Path(background_image).exists():
        img = Image.open(background_image)
        ax.imshow(img, cmap='gray', alpha=0.7)

    x = df['COORDINATE_X'].values
    y = df['COORDINATE_Y'].values

    if strain_component not in df.columns:
        print(f"Error: Column '{strain_component}' not found in data")
        print(f"Available columns: {list(df.columns)}")
        return fig, ax

    strain = df[strain_component].values

    scatter = ax.scatter(x, y, c=strain, cmap='RdBu_r', s=50, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label=strain_component)

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Strain Field: {strain_component}')
    ax.set_aspect('equal')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def print_data_summary(df, filepath):
    """Print summary statistics of DICe output data."""
    print(f"\nData Summary for {Path(filepath).name}:")
    print(f"  Total points: {len(df)}")

    if 'DISPLACEMENT_X' in df.columns and 'DISPLACEMENT_Y' in df.columns:
        dx = df['DISPLACEMENT_X']
        dy = df['DISPLACEMENT_Y']
        mag = np.sqrt(dx**2 + dy**2)
        print(
            f"  Displacement X: min={dx.min():.3f}, max={dx.max():.3f}, mean={dx.mean():.3f}")
        print(
            f"  Displacement Y: min={dy.min():.3f}, max={dy.max():.3f}, mean={dy.mean():.3f}")
        print(
            f"  Magnitude: min={mag.min():.3f}, max={mag.max():.3f}, mean={mag.mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DICe displacement and strain output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s DICe_solution_050.txt --scale 10 --show
  %(prog)s DICe_solution_050.txt --strain --strain-component VSG_STRAIN_XX
  %(prog)s DICe_solution_050.txt --no-background -o output.png
        """
    )

    parser.add_argument('input_file', help='DICe output file (.txt)')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--scale', type=float, default=10,
                        help='Arrow scale factor (default: 10)')
    parser.add_argument('--strain', action='store_true',
                        help='Plot strain field instead of displacement')
    parser.add_argument('--strain-component', default='VSG_STRAIN_XX',
                        choices=[
                            'VSG_STRAIN_XX',
                            'VSG_STRAIN_YY',
                            'VSG_STRAIN_XY'],
                        help='Strain component to plot (default: VSG_STRAIN_XX)')
    parser.add_argument('--no-background', action='store_true',
                        help='Do not show background image')
    parser.add_argument('--show', action='store_true',
                        help='Display plot interactively')
    parser.add_argument('--summary', action='store_true',
                        help='Print data summary statistics')

    args = parser.parse_args()

    # Check input file
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}")
        return 1

    # Load data
    print(f"Loading: {args.input_file}")
    df, ref_image, def_image = load_dice_output(args.input_file)
    print(f"  Loaded {len(df)} data points")

    if args.summary:
        print_data_summary(df, args.input_file)

    # Determine background image
    background = None if args.no_background else ref_image

    # Generate output path if not specified
    if args.output is None:
        input_path = Path(args.input_file)
        if args.strain:
            args.output = str(input_path.with_suffix('.strain.png'))
        else:
            args.output = str(input_path.with_suffix('.displacement.png'))

    # Plot
    if args.strain:
        plot_strain_field(df, background, args.strain_component, args.output)
    else:
        title = f"Displacement Vectors (scale: {args.scale}x)"
        plot_displacement_vectors(
            df, background, args.output, args.scale, title)

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    exit(main())

