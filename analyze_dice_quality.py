#!/usr/bin/env python3
"""
analyze_dice_quality.py - Analyze DICe output quality metrics for ML training data filtering

This script analyzes SIGMA, GAMMA, and MATCH values from DICe output files
to help identify high-quality displacement data suitable for ML model training.

Quality metrics (from DICe documentation):
- SIGMA: Predicted displacement variation (lower = better, less uncertainty)
- GAMMA: Template matching quality (lower = better, 0.0 = perfect match)
- MATCH: 0 = success, -1 = failure
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def parse_dice_file(filepath):
    """Parse a single DICe solution file and return DataFrame."""
    # Skip header lines (start with ***)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (contains SUBSET_ID)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('SUBSET_ID'):
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError(f"Could not find header in {filepath}")
    
    # Read data starting from header
    df = pd.read_csv(filepath, skiprows=header_idx, header=0)
    return df


def analyze_single_file(filepath, verbose=False):
    """Analyze quality metrics for a single DICe output file."""
    df = parse_dice_file(filepath)
    
    stats = {
        'file': os.path.basename(filepath),
        'total_points': len(df),
        'match_success': (df['MATCH'] == 0).sum(),
        'match_failure': (df['MATCH'] == -1).sum(),
        'match_success_rate': (df['MATCH'] == 0).mean() * 100,
        'sigma_mean': df['SIGMA'].mean(),
        'sigma_std': df['SIGMA'].std(),
        'sigma_max': df['SIGMA'].max(),
        'gamma_mean': df['GAMMA'].mean(),
        'gamma_std': df['GAMMA'].std(),
        'gamma_min': df['GAMMA'].min(),
        'gamma_max': df['GAMMA'].max(),
    }
    
    if verbose:
        print(f"\n=== {stats['file']} ===")
        print(f"Total points: {stats['total_points']}")
        print(f"Match success rate: {stats['match_success_rate']:.1f}%")
        print(f"SIGMA - mean: {stats['sigma_mean']:.4f}, std: {stats['sigma_std']:.4f}")
        print(f"GAMMA - mean: {stats['gamma_mean']:.4f}, range: [{stats['gamma_min']:.4f}, {stats['gamma_max']:.4f}]")
    
    return stats, df


def analyze_sequence(dice_dir, output_file=None):
    """Analyze all DICe solution files in a directory."""
    pattern = os.path.join(dice_dir, "DICe_solution_*.txt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No DICe solution files found in {dice_dir}")
        return None
    
    print(f"Found {len(files)} DICe solution files")
    
    all_stats = []
    all_data = []
    
    for f in files:
        try:
            stats, df = analyze_single_file(f)
            df['frame'] = os.path.basename(f)
            all_stats.append(stats)
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    stats_df = pd.DataFrame(all_stats)
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SEQUENCE QUALITY SUMMARY")
    print("="*60)
    print(f"Total frames analyzed: {len(stats_df)}")
    print(f"Total data points: {len(combined_df)}")
    print(f"\nMATCH Statistics:")
    print(f"  Overall success rate: {stats_df['match_success_rate'].mean():.1f}%")
    print(f"  Frames with 100% success: {(stats_df['match_success_rate'] == 100).sum()}")
    print(f"  Frames with 0% success: {(stats_df['match_success_rate'] == 0).sum()}")
    
    print(f"\nSIGMA (displacement uncertainty):")
    print(f"  Mean across sequence: {combined_df['SIGMA'].mean():.6f}")
    print(f"  Std across sequence: {combined_df['SIGMA'].std():.6f}")
    print(f"  Range: [{combined_df['SIGMA'].min():.6f}, {combined_df['SIGMA'].max():.6f}]")
    
    print(f"\nGAMMA (matching quality, lower=better):")
    print(f"  Mean across sequence: {combined_df['GAMMA'].mean():.4f}")
    print(f"  Std across sequence: {combined_df['GAMMA'].std():.4f}")
    print(f"  Range: [{combined_df['GAMMA'].min():.4f}, {combined_df['GAMMA'].max():.4f}]")
    
    # Quality filtering recommendations
    print("\n" + "="*60)
    print("ML TRAINING DATA RECOMMENDATIONS")

    # Alternative quality filter (ignoring MATCH, which may be unreliable for SEM images)
    alt_quality = combined_df[
        (combined_df['GAMMA'] < 0.5) & 
        (combined_df['SIGMA'] < 0.02)
    ]
    print(f"Alternative filter (GAMMA<0.5, SIGMA<0.02, ignoring MATCH): {len(alt_quality)} ({len(alt_quality)/len(combined_df)*100:.1f}%)")
    
    # Points with good GAMMA only
    good_gamma = combined_df[combined_df['GAMMA'] < 0.3]
    print(f"Good matching quality (GAMMA<0.3): {len(good_gamma)} ({len(good_gamma)/len(combined_df)*100:.1f}%)")
    
    print("="*60)
    
    # Count high-quality points (MATCH=0, GAMMA < 0.5, SIGMA < 0.02)
    high_quality = combined_df[
        (combined_df['MATCH'] == 0) & 
        (combined_df['GAMMA'] < 0.5) & 
        (combined_df['SIGMA'] < 0.02)
    ]
    print(f"High-quality points (MATCH=0, GAMMA<0.5, SIGMA<0.02): {len(high_quality)} ({len(high_quality)/len(combined_df)*100:.1f}%)")
    
    # Save summary if output file specified
    if output_file:
        stats_df.to_csv(output_file, index=False)
        print(f"\nFrame-by-frame statistics saved to: {output_file}")
    
    return stats_df, combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze DICe output quality metrics for ML training data filtering'
    )
    parser.add_argument('dice_dir', help='Directory containing DICe solution files')
    parser.add_argument('-o', '--output', help='Output CSV file for statistics')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show per-file details')
    parser.add_argument('--single', help='Analyze a single file instead of directory')
    
    args = parser.parse_args()
    
    if args.single:
        analyze_single_file(args.single, verbose=True)
    else:
        analyze_sequence(args.dice_dir, args.output)


if __name__ == '__main__':
    main()