#!/usr/bin/env python3
"""
SEM Image Preprocessor for DICe
Prepares SEM image sequences for DICe processing by:
- Detecting and cropping SEM info bars
- Converting to grayscale
- Standardizing file naming
"""
import argparse
import warnings
from pathlib import Path
from PIL import Image
import numpy as np

# Suppress PIL/numpy deprecation warning
warnings.filterwarnings('ignore', message='.*__array__.*copy.*')


class SEMImagePreprocessor:
    """Preprocesses SEM images for DICe analysis."""

    def __init__(self, input_dir, output_dir, info_bar_rows=60, force_grayscale=True):
        self.input_dir = Path(input_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.info_bar_rows = info_bar_rows
        self.force_grayscale = force_grayscale

    def detect_info_bar(self, img_path, threshold=10):
        """
        Detect if image has an SEM info bar at the bottom.
        Returns the number of rows to crop, or 0 if no info bar detected.
        """
        img = Image.open(img_path)
        arr = np.array(img)

        # Handle RGB images
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]

        # Check for black rows at the bottom (typical SEM info bar pattern)
        # Look for a black separator line in the bottom 70 rows
        for i in range(1, 70):
            row_mean = np.mean(arr[-i])
            if row_mean < threshold:
                # Found a black row, check if there's a pattern
                # SEM info bars typically have black borders
                if i <= 10:
                    # Check for another black region around row -55 to -60
                    for j in range(55, 65):
                        if j < arr.shape[0] and np.mean(arr[-j]) < threshold:
                            return 60  # Standard SEM info bar height
                    return i + 5  # Small border only
        return 0

    def process_image(self, img_path, output_path, crop_rows):
        """Process a single image: crop info bar and convert to grayscale."""
        img = Image.open(img_path)

        # Crop info bar if needed
        if crop_rows > 0:
            width, height = img.size
            img = img.crop((0, 0, width, height - crop_rows))

        # Convert to grayscale if needed
        if self.force_grayscale and img.mode != 'L':
            img = img.convert('L')

        # Save as TIFF
        img.save(output_path, format='TIFF')
        return img.size

    def get_output_filename(self, index, num_digits=4):
        """Generate standardized output filename."""
        return f"frame_{index:0{num_digits}d}.tif"

    def process_sequence(self):
        """Process all images in the input directory."""
        # Find all TIFF images
        tif_files = sorted(
            list(self.input_dir.glob("*.tif")) +
            list(self.input_dir.glob("*.tiff")) +
            list(self.input_dir.glob("*.TIF")) +
            list(self.input_dir.glob("*.TIFF"))
        )

        if not tif_files:
            raise ValueError(f"No TIFF images found in {self.input_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect info bar from first image
        crop_rows = self.detect_info_bar(tif_files[0])
        if crop_rows > 0:
            print(f"Detected SEM info bar: {crop_rows} rows will be cropped")
        else:
            print("No SEM info bar detected")

        # Get original image info
        first_img = Image.open(tif_files[0])
        orig_size = first_img.size
        orig_mode = first_img.mode

        # Process images
        print(f"\nProcessing {len(tif_files)} images...")
        num_digits = len(str(len(tif_files)))
        num_digits = max(num_digits, 4)  # At least 4 digits

        processed = []
        for i, img_path in enumerate(tif_files, start=1):
            output_name = self.get_output_filename(i, num_digits)
            output_path = self.output_dir / output_name
            new_size = self.process_image(img_path, output_path, crop_rows)

            if i == 1:
                print(f"  Original: {orig_size[0]}x{orig_size[1]} ({orig_mode})")
                print(f"  Output:   {new_size[0]}x{new_size[1]} (L)")

            processed.append(output_name)

            # Progress indicator
            if i % 20 == 0 or i == len(tif_files):
                print(f"  Processed {i}/{len(tif_files)} images")

        return {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'num_images': len(processed),
            'original_size': orig_size,
            'new_size': new_size,
            'crop_rows': crop_rows,
            'converted_to_grayscale': orig_mode != 'L'
        }


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess SEM images for DICe analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/data/A1\\ Video ~/data/A1_preprocessed
  %(prog)s --info-bar-rows 70 ~/data/ESWG007 ~/data/ESWG007_prep
  %(prog)s --no-crop ~/data/ESWG006 ~/data/ESWG006_gray
        """
    )
    parser.add_argument('input_dir', help='Directory containing raw SEM images')
    parser.add_argument('output_dir', help='Directory for preprocessed images')
    parser.add_argument('--info-bar-rows', type=int, default=60,
                        help='Number of rows to crop from bottom (default: 60)')
    parser.add_argument('--no-crop', action='store_true',
                        help='Skip info bar cropping (auto-detection)')
    parser.add_argument('--keep-color', action='store_true',
                        help='Keep original color mode (do not convert to grayscale)')

    args = parser.parse_args()

    preprocessor = SEMImagePreprocessor(
        args.input_dir,
        args.output_dir,
        info_bar_rows=0 if args.no_crop else args.info_bar_rows,
        force_grayscale=not args.keep_color
    )

    try:
        print(f"SEM Image Preprocessor")
        print(f"=" * 50)
        print(f"Input:  {args.input_dir}")
        print(f"Output: {args.output_dir}")
        print()

        result = preprocessor.process_sequence()

        print()
        print(f"Preprocessing complete!")
        print(f"  Images processed: {result['num_images']}")
        print(f"  Size: {result['original_size'][0]}x{result['original_size'][1]} -> "
              f"{result['new_size'][0]}x{result['new_size'][1]}")
        if result['crop_rows'] > 0:
            print(f"  Cropped: {result['crop_rows']} rows (info bar)")
        if result['converted_to_grayscale']:
            print(f"  Converted to grayscale")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
