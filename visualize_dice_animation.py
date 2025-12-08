#!/usr/bin/env python3
"""
visualize_dice_animation.py
Generate animated GIF showing DICe displacement field evolution over time.

Usage:
    python visualize_dice_animation.py --image-dir <path> --dice-dir <path> --output animation.gif
    python visualize_dice_animation.py --image-dir ./images --dice-dir ./dice_output -o result.gif --fps 10 --scale 5
"""

import argparse
import os
import glob
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio


def load_dice_output(filepath):
    """
    Load DICe output file, skipping header comments.
    
    Returns numpy array with columns:
    [SUBSET_ID, COORDINATE_X, COORDINATE_Y, DISPLACEMENT_X, DISPLACEMENT_Y, ...]
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Skip comment lines and header
            if line.startswith('***') or line.startswith('SUBSET_ID'):
                continue
            if line.strip():
                values = line.strip().split(',')
                try:
                    data.append([float(v) for v in values])
                except ValueError:
                    continue  # Skip malformed lines
    return np.array(data)


def fig_to_array(fig, width, height):
    """Convert matplotlib figure to numpy array with fixed size."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB').resize((width, height), Image.LANCZOS)
    return np.asarray(img)


def create_displacement_gif(image_dir, dice_output_dir, output_gif, 
                            scale=5, fps=5, max_frames=None):
    """
    Create animated GIF of displacement vectors over image sequence.
    
    Args:
        image_dir: Directory containing SEM images (.tif)
        dice_output_dir: Directory containing DICe output files (DICe_solution_*.txt)
        output_gif: Output GIF path
        scale: Arrow scale multiplier for visualization
        fps: Frames per second in output GIF
        max_frames: Maximum number of frames to process (None for all)
    """
    # Find files
    dice_files = sorted(glob.glob(os.path.join(dice_output_dir, "DICe_solution_*.txt")))
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    
    if not dice_files:
        print(f"Error: No DICe output files found in {dice_output_dir}")
        return
    if not image_files:
        print(f"Error: No .tif images found in {image_dir}")
        return
    
    print(f"Found {len(dice_files)} DICe files")
    print(f"Found {len(image_files)} images")
    
    # Limit frames if specified
    if max_frames:
        dice_files = dice_files[:max_frames]
    
    frames = []
    frame_width = 1200
    frame_height = 1000
    
    # Get global displacement range for consistent colorbar
    print("Calculating displacement range...")
    all_magnitudes = []
    for dice_file in dice_files[:min(10, len(dice_files))]:  # Sample first 10
        data = load_dice_output(dice_file)
        if len(data) > 0:
            dx, dy = data[:, 3], data[:, 4]
            all_magnitudes.extend(np.sqrt(dx**2 + dy**2))
    
    vmin, vmax = 0, np.percentile(all_magnitudes, 95) if all_magnitudes else 10
    
    print(f"Processing frames...")
    for i, dice_file in enumerate(dice_files):
        # Extract frame number from filename
        basename = os.path.basename(dice_file)
        try:
            frame_num = int(basename.split('_')[2].split('.')[0])
        except (IndexError, ValueError):
            frame_num = i + 1
        
        # Get corresponding image
        if frame_num > len(image_files):
            continue
        img_path = image_files[frame_num - 1]
        
        # Load image
        img = Image.open(img_path)
        img_array = np.asarray(img)
        
        # Load DICe data
        data = load_dice_output(dice_file)
        if len(data) == 0:
            continue
        
        x = data[:, 1]   # COORDINATE_X
        y = data[:, 2]   # COORDINATE_Y
        dx = data[:, 3]  # DISPLACEMENT_X
        dy = data[:, 4]  # DISPLACEMENT_Y
        
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Show background image
        ax.imshow(img_array, cmap='gray', alpha=1.0)
        
        # Draw displacement vectors
        q = ax.quiver(x, y, dx * scale, dy * scale,
                      magnitude,
                      cmap='jet',
                      clim=(vmin, vmax),
                      angles='xy',
                      scale_units='xy',
                      scale=1,
                      width=0.003,
                      headwidth=3,
                      headlength=4)
        
        ax.set_title(f'Frame {frame_num} - Displacement Vectors (x{scale})')
        ax.set_xlim(0, img_array.shape[1])
        ax.set_ylim(img_array.shape[0], 0)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(q, ax=ax, label='Displacement (pixels)')
        plt.tight_layout()
        
        # Convert to array
        frame = fig_to_array(fig, frame_width, frame_height)
        frames.append(frame)
        
        plt.close(fig)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(dice_files):
            print(f"  Processed {i + 1}/{len(dice_files)} frames")
    
    if not frames:
        print("Error: No frames generated")
        return
    
    # Save GIF
    print(f"Saving GIF to {output_gif}...")
    imageio.mimsave(output_gif, frames, fps=fps, loop=0)
    
    # Report file size
    size_mb = os.path.getsize(output_gif) / (1024 * 1024)
    print(f"Done! Saved {len(frames)} frames ({size_mb:.1f} MB)")
    
    return output_gif


def main():
    parser = argparse.ArgumentParser(
        description='Generate animated GIF of DICe displacement field evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image-dir ./images --dice-dir ./dice_output -o animation.gif
  %(prog)s -i ./ESWG006_gray -d ./test_config -o result.gif --fps 10 --scale 3
        """
    )
    
    parser.add_argument('--image-dir', '-i', required=True,
                        help='Directory containing SEM images (.tif)')
    parser.add_argument('--dice-dir', '-d', required=True,
                        help='Directory containing DICe output files')
    parser.add_argument('--output', '-o', default='displacement_animation.gif',
                        help='Output GIF path (default: displacement_animation.gif)')
    parser.add_argument('--scale', '-s', type=float, default=5,
                        help='Arrow scale multiplier (default: 5)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second (default: 5)')
    parser.add_argument('--max-frames', '-n', type=int, default=None,
                        help='Maximum frames to process (default: all)')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    if not os.path.isdir(args.dice_dir):
        print(f"Error: DICe output directory not found: {args.dice_dir}")
        return 1
    
    create_displacement_gif(
        image_dir=args.image_dir,
        dice_output_dir=args.dice_dir,
        output_gif=args.output,
        scale=args.scale,
        fps=args.fps,
        max_frames=args.max_frames
    )
    
    return 0


if __name__ == "__main__":
    exit(main())