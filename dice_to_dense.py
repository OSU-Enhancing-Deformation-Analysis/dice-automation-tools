#!/usr/bin/env python3
"""
DICe to Dense Flow Field Converter

Converts sparse DICe displacement data to dense flow fields for ML training.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def parse_dice_file(filepath):
    """
    Parse a DICe solution file and extract displacement data.

    Returns:
        dict with keys: points (N,2), displacements (N,2), metadata
    """
    filepath = Path(filepath)

    # Read the CSV, skipping comment lines starting with *
    df = pd.read_csv(filepath, comment='*')

    # Extract coordinates and displacements
    coords_x = df['COORDINATE_X'].values
    coords_y = df['COORDINATE_Y'].values
    disp_x = df['DISPLACEMENT_X'].values
    disp_y = df['DISPLACEMENT_Y'].values

    # Also grab quality metrics for later use
    sigma = df['SIGMA'].values if 'SIGMA' in df.columns else None
    gamma = df['GAMMA'].values if 'GAMMA' in df.columns else None

    return {
        'coords': np.column_stack([coords_x, coords_y]),
        'displacements': np.column_stack([disp_x, disp_y]),
        'sigma': sigma,
        'gamma': gamma,
        'num_points': len(coords_x)
    }


def interpolate_to_dense(coords, values, image_shape, method='cubic'):
    """
    Interpolate sparse points to a dense grid.

    Args:
        coords: (N, 2) array of (x, y) coordinates
        values: (N,) array of values to interpolate
        image_shape: (height, width) tuple
        method: interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        (height, width) dense array
    """
    height, width = image_shape

    # Create dense grid
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Interpolate
    dense = griddata(
        coords,           # (N, 2) points
        values,           # (N,) values
        (grid_x, grid_y), # grid to interpolate onto
        method=method,
        fill_value=0.0    # fill NaN regions with 0
    )

    return dense


def dice_to_dense_flow(dice_file, image_shape, method='cubic', quiet=False):
    """
    Convert a DICe solution file to a dense flow field.

    Args:
        dice_file: path to DICe solution file
        image_shape: (height, width) of the original image
        method: interpolation method
        quiet: suppress print output

    Returns:
        (2, H, W) dense flow field (dx, dy)
    """
    # Parse the file
    data = parse_dice_file(dice_file)

    if not quiet:
        print(f"Parsed {data['num_points']} points")
        print(f"  Coordinate X range: [{data['coords'][:,0].min():.1f}, {data['coords'][:,0].max():.1f}]")
        print(f"  Coordinate Y range: [{data['coords'][:,1].min():.1f}, {data['coords'][:,1].max():.1f}]")
        print(f"  Displacement X range: [{data['displacements'][:,0].min():.4f}, {data['displacements'][:,0].max():.4f}]")
        print(f"  Displacement Y range: [{data['displacements'][:,1].min():.4f}, {data['displacements'][:,1].max():.4f}]")

        if data['gamma'] is not None:
            print(f"  GAMMA range: [{data['gamma'].min():.4f}, {data['gamma'].max():.4f}]")
        if data['sigma'] is not None:
            print(f"  SIGMA range: [{data['sigma'].min():.6f}, {data['sigma'].max():.6f}]")

        print(f"\nInterpolating to {image_shape[1]}x{image_shape[0]} dense grid using '{method}' method...")

    # Interpolate displacement_x
    dense_dx = interpolate_to_dense(
        data['coords'],
        data['displacements'][:, 0],
        image_shape,
        method=method
    )

    # Interpolate displacement_y
    dense_dy = interpolate_to_dense(
        data['coords'],
        data['displacements'][:, 1],
        image_shape,
        method=method
    )

    # Stack into (2, H, W) format
    flow_field = np.stack([dense_dx, dense_dy], axis=0).astype(np.float32)

    if not quiet:
        print(f"  Dense flow field shape: {flow_field.shape}")
        print(f"  Dense DX range: [{flow_field[0].min():.4f}, {flow_field[0].max():.4f}]")
        print(f"  Dense DY range: [{flow_field[1].min():.4f}, {flow_field[1].max():.4f}]")

        # Count NaN regions (outside convex hull of points)
        nan_mask = np.isnan(dense_dx) | np.isnan(dense_dy)
        nan_pct = nan_mask.sum() / nan_mask.size * 100
        print(f"  Fill regions (outside data): {nan_pct:.1f}%")

    return flow_field, data


def visualize_flow(flow_field, sparse_data, output_path, image_shape):
    """Create visualization of the flow field."""
    _, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Dense displacement X (heatmap)
    ax = axes[0, 0]
    im = ax.imshow(flow_field[0], cmap='RdBu_r', origin='upper')
    ax.set_title('Dense Displacement X (pixels)')
    plt.colorbar(im, ax=ax)

    # Plot 2: Dense displacement Y (heatmap)
    ax = axes[0, 1]
    im = ax.imshow(flow_field[1], cmap='RdBu_r', origin='upper')
    ax.set_title('Dense Displacement Y (pixels)')
    plt.colorbar(im, ax=ax)

    # Plot 3: Displacement magnitude
    ax = axes[1, 0]
    magnitude = np.sqrt(flow_field[0]**2 + flow_field[1]**2)
    im = ax.imshow(magnitude, cmap='viridis', origin='upper')
    ax.set_title('Displacement Magnitude (pixels)')
    plt.colorbar(im, ax=ax)

    # Plot 4: Quiver plot (subsampled)
    ax = axes[1, 1]
    # Subsample for visibility
    step = 20
    Y, X = np.mgrid[0:image_shape[0]:step, 0:image_shape[1]:step]
    U = flow_field[0][::step, ::step]
    V = flow_field[1][::step, ::step]

    # Scale arrows for visibility
    scale_factor = 50
    ax.quiver(X, Y, U * scale_factor, V * scale_factor,
              magnitude[::step, ::step], cmap='viridis', scale=1, scale_units='xy')
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  # Flip Y axis to match image coordinates
    ax.set_aspect('equal')
    ax.set_title(f'Displacement Vectors (scaled {scale_factor}x)')

    # Add sparse points overlay
    coords = sparse_data['coords']
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=5, alpha=0.5, label='DICe points')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Convert DICe output to dense flow field')
    parser.add_argument('input', help='Path to DICe solution file')
    parser.add_argument('--output', '-o', help='Output .npy file path (default: input_dense.npy)')
    parser.add_argument('--width', type=int, default=1024, help='Image width (default: 1024)')
    parser.add_argument('--height', type=int, default=883, help='Image height (default: 883)')
    parser.add_argument('--method', default='cubic', choices=['linear', 'cubic', 'nearest'],
                        help='Interpolation method (default: cubic)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        output_path = input_path.with_suffix('.npy')

    image_shape = (args.height, args.width)

    print(f"DICe to Dense Flow Converter")
    print(f"=" * 50)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Image shape: {image_shape[1]}x{image_shape[0]}")
    print()

    # Convert
    flow_field, sparse_data = dice_to_dense_flow(input_path, image_shape, method=args.method)

    # Save
    np.save(output_path, flow_field)
    print(f"\nSaved dense flow field to: {output_path}")

    # Visualize
    if not args.no_viz:
        viz_path = output_path.with_suffix('.png')
        visualize_flow(flow_field, sparse_data, viz_path, image_shape)

    print("\nDone!")


if __name__ == "__main__":
    main()
