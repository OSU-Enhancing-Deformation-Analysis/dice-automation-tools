#!/usr/bin/env python3
"""
Batch DICe to Dense Flow Converter

Processes all DICe solution files and converts them to dense flow fields
for ML training.
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

from dice_to_dense import dice_to_dense_flow, visualize_flow


def get_image_dimensions(dataset_dir):
    """Get image dimensions from the first preprocessed image."""
    preprocess_dir = dataset_dir / "preprocessed"
    if not preprocess_dir.exists():
        preprocess_dir = dataset_dir  # Fallback

    # Find first tif file
    tif_files = list(preprocess_dir.glob("*.tif"))
    if not tif_files:
        return None

    img = Image.open(tif_files[0])
    return (img.height, img.width)


def process_dataset(dataset_dir, output_dir, viz_every=10, method='cubic'):
    """
    Process all DICe solution files in a dataset directory.

    Args:
        dataset_dir: Path to dataset (e.g., processed_datasets/A1_Video)
        output_dir: Path to output directory
        viz_every: Create visualization every N frames (0 to disable)
        method: Interpolation method

    Returns:
        dict with processing statistics
    """
    dataset_name = dataset_dir.name
    dice_dir = dataset_dir / "dice_output"

    if not dice_dir.exists():
        return {'error': f'No dice_output directory found in {dataset_dir}'}

    # Get image dimensions
    image_shape = get_image_dimensions(dataset_dir)
    if image_shape is None:
        return {'error': f'Could not determine image dimensions for {dataset_dir}'}

    # Find all solution files
    solution_files = sorted(dice_dir.glob("DICe_solution_*.txt"))
    if not solution_files:
        return {'error': f'No DICe solution files found in {dice_dir}'}

    # Create output directories
    dense_dir = output_dir / "dense_flow"
    viz_dir = output_dir / "visualizations"
    dense_dir.mkdir(parents=True, exist_ok=True)
    if viz_every > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    print(f"  DICe files: {len(solution_files)}")
    print(f"  Image shape: {image_shape[1]}x{image_shape[0]}")
    print(f"  Output: {output_dir}")

    # Statistics accumulators
    stats = {
        'dataset': dataset_name,
        'num_files': len(solution_files),
        'image_shape': list(image_shape),
        'method': method,
        'processed': 0,
        'errors': 0,
        'dx_min': float('inf'),
        'dx_max': float('-inf'),
        'dy_min': float('inf'),
        'dy_max': float('-inf'),
        'magnitude_max': 0,
        'total_points': 0,
    }

    start_time = time.time()

    for i, solution_file in enumerate(solution_files, start=1):
        # Extract frame number from filename (e.g., DICe_solution_050.txt -> 050)
        frame_num = solution_file.stem.split('_')[-1]

        try:
            # Convert to dense flow (suppress print output for batch mode)
            flow_field, sparse_data = dice_to_dense_flow(
                solution_file, image_shape, method=method, quiet=True
            )

            # Save numpy file
            npy_path = dense_dir / f"flow_{frame_num}.npy"
            np.save(npy_path, flow_field)

            # Update statistics
            stats['processed'] += 1
            stats['dx_min'] = min(stats['dx_min'], float(flow_field[0].min()))
            stats['dx_max'] = max(stats['dx_max'], float(flow_field[0].max()))
            stats['dy_min'] = min(stats['dy_min'], float(flow_field[1].min()))
            stats['dy_max'] = max(stats['dy_max'], float(flow_field[1].max()))
            magnitude = np.sqrt(flow_field[0]**2 + flow_field[1]**2)
            stats['magnitude_max'] = max(stats['magnitude_max'], float(magnitude.max()))
            stats['total_points'] += sparse_data['num_points']

            # Create visualization every N frames
            if viz_every > 0 and (i % viz_every == 0 or i == 1):
                viz_path = viz_dir / f"flow_{frame_num}.png"
                visualize_flow(flow_field, sparse_data, viz_path, image_shape)

        except Exception as e:
            print(f"    ERROR processing {solution_file.name}: {e}")
            stats['errors'] += 1
            continue

        # Progress indicator
        if i % 20 == 0 or i == len(solution_files):
            print(f"  Processed {i}/{len(solution_files)} files")

    elapsed = time.time() - start_time
    stats['elapsed_seconds'] = elapsed
    stats['avg_points_per_frame'] = stats['total_points'] / max(stats['processed'], 1)

    print(f"  Done in {elapsed:.1f}s")
    print(f"  DX range: [{stats['dx_min']:.4f}, {stats['dx_max']:.4f}]")
    print(f"  DY range: [{stats['dy_min']:.4f}, {stats['dy_max']:.4f}]")
    print(f"  Max magnitude: {stats['magnitude_max']:.4f} pixels")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert DICe output to dense flow fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single dataset
  %(prog)s --dataset A1_Video

  # Process all datasets
  %(prog)s --all

  # Custom visualization frequency
  %(prog)s --all --viz-every 5

  # No visualizations (faster)
  %(prog)s --all --viz-every 0
        """
    )

    default_input = "~/Documents/Capstone/dice-automation-tools/processed_datasets"
    default_output = "~/Documents/Capstone/dice-automation-tools/ml_training_data"

    parser.add_argument('--input', default=default_input,
                        help=f'Input directory with processed datasets (default: {default_input})')
    parser.add_argument('--output', default=default_output,
                        help=f'Output directory for ML training data (default: {default_output})')
    parser.add_argument('--dataset', help='Process single dataset by name (e.g., A1_Video)')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    parser.add_argument('--viz-every', type=int, default=10,
                        help='Create visualization every N frames (0 to disable, default: 10)')
    parser.add_argument('--method', default='cubic', choices=['linear', 'cubic', 'nearest'],
                        help='Interpolation method (default: cubic)')

    args = parser.parse_args()

    input_root = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()

    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        return 1

    # Determine which datasets to process
    if args.dataset:
        datasets = [input_root / args.dataset]
        if not datasets[0].exists():
            print(f"Error: Dataset not found: {datasets[0]}")
            return 1
    elif args.all:
        datasets = sorted([d for d in input_root.iterdir() if d.is_dir()])
    else:
        print("Error: Specify --dataset NAME or --all")
        parser.print_help()
        return 1

    print("=" * 60)
    print("Batch DICe to Dense Flow Converter")
    print("=" * 60)
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Datasets: {len(datasets)}")
    print(f"Visualization: every {args.viz_every} frames" if args.viz_every > 0 else "Visualization: disabled")

    # Process each dataset
    all_stats = []
    total_start = time.time()

    for dataset_dir in datasets:
        dataset_output = output_root / dataset_dir.name
        stats = process_dataset(
            dataset_dir,
            dataset_output,
            viz_every=args.viz_every,
            method=args.method
        )
        all_stats.append(stats)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    print()

    total_files = sum(s.get('processed', 0) for s in all_stats)
    total_errors = sum(s.get('errors', 0) for s in all_stats)

    print(f"{'Dataset':<20} {'Files':>8} {'Errors':>8} {'Max Mag':>10}")
    print("-" * 50)
    for stats in all_stats:
        if 'error' in stats:
            print(f"{'ERROR':<20} {stats['error']}")
        else:
            print(f"{stats['dataset']:<20} {stats['processed']:>8} {stats['errors']:>8} "
                  f"{stats['magnitude_max']:>10.4f}")
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_files:>8} {total_errors:>8}")

    # Save summary JSON
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "conversion_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_elapsed_seconds': total_elapsed,
        'total_files_processed': total_files,
        'total_errors': total_errors,
        'datasets': all_stats
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    exit(main())
