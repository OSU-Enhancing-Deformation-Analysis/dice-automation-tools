#!/usr/bin/env python3
"""
Batch Process All Datasets
Complete pipeline: Preprocess -> Generate Config -> Run DICe -> Analyze

Processes all SEM image sequences in Graphite_image_sequences folder.
"""
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

from preprocess_sem_images import SEMImagePreprocessor
from dice_config_generator import DICeConfigGenerator


class BatchProcessor:
    """Orchestrates the complete DICe processing pipeline for multiple datasets."""

    # Folders to skip (already processed, no images, or special cases)
    SKIP_FOLDERS = {
        'ESWG006',       # Already processed
        'ESWG006_gray',  # Duplicate/grayscale version
        '6 Video',       # No images, only plots
    }

    def __init__(self, source_root, output_root, dice_binary):
        self.source_root = Path(source_root).expanduser()
        self.output_root = Path(output_root).expanduser()
        self.dice_binary = Path(dice_binary).expanduser()

        if not self.dice_binary.exists():
            raise FileNotFoundError(f"DICe binary not found: {self.dice_binary}")

        if not self.source_root.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_root}")

    def sanitize_name(self, folder_name):
        """Convert folder name to safe directory name."""
        return folder_name.replace(' ', '_')

    def find_datasets(self):
        """Find all valid image sequence folders."""
        datasets = []

        for folder in sorted(self.source_root.iterdir()):
            if not folder.is_dir():
                continue

            if folder.name in self.SKIP_FOLDERS:
                print(f"  Skipping: {folder.name} (in skip list)")
                continue

            # Count TIF files
            tif_files = list(folder.glob("*.tif")) + list(folder.glob("*.TIF"))
            tif_files += list(folder.glob("*.tiff")) + list(folder.glob("*.TIFF"))

            if len(tif_files) < 10:
                print(f"  Skipping: {folder.name} (only {len(tif_files)} images)")
                continue

            datasets.append({
                'name': folder.name,
                'safe_name': self.sanitize_name(folder.name),
                'path': folder,
                'image_count': len(tif_files)
            })

        return datasets

    def process_dataset(self, dataset, subset_size=41, step_size=50):
        """Run complete pipeline for one dataset."""
        name = dataset['name']
        safe_name = dataset['safe_name']
        source_path = dataset['path']

        # Output directories
        dataset_dir = self.output_root / safe_name
        preprocess_dir = dataset_dir / "preprocessed"
        dice_dir = dataset_dir / "dice_output"

        print(f"\n{'='*60}")
        print(f"PROCESSING: {name}")
        print(f"{'='*60}")
        print(f"  Source: {source_path}")
        print(f"  Images: {dataset['image_count']}")
        print(f"  Output: {dataset_dir}")

        result = {
            'name': name,
            'safe_name': safe_name,
            'image_count': dataset['image_count'],
            'preprocess': False,
            'config': False,
            'dice': False,
            'error': None
        }

        # Step 1: Preprocess images
        print(f"\n  [1/3] Preprocessing images...")
        try:
            preprocessor = SEMImagePreprocessor(
                source_path,
                preprocess_dir,
                info_bar_rows=60,
                force_grayscale=True
            )
            prep_result = preprocessor.process_sequence()
            result['preprocess'] = True
            result['preprocessed_size'] = prep_result['new_size']
            result['crop_rows'] = prep_result['crop_rows']
            print(f"        Done: {prep_result['num_images']} images")
            print(f"        Size: {prep_result['new_size'][0]}x{prep_result['new_size'][1]}")
            if prep_result['crop_rows'] > 0:
                print(f"        Cropped: {prep_result['crop_rows']} rows")
        except Exception as e:
            result['error'] = f"Preprocessing failed: {e}"
            print(f"        ERROR: {e}")
            return result

        # Step 2: Generate DICe configuration
        print(f"\n  [2/3] Generating DICe configuration...")
        try:
            generator = DICeConfigGenerator(
                preprocess_dir,
                dice_dir,
                subset_size=subset_size,
                step_size=step_size
            )
            config_result = generator.generate()
            result['config'] = True
            result['tracking_points'] = config_result['sequence_info']['width'] // step_size * \
                                        config_result['sequence_info']['height'] // step_size
            print(f"        Done: ~{result['tracking_points']} tracking points")
        except Exception as e:
            result['error'] = f"Config generation failed: {e}"
            print(f"        ERROR: {e}")
            return result

        # Step 3: Run DICe
        print(f"\n  [3/3] Running DICe (this may take a while)...")
        input_xml = dice_dir / "input.xml"
        start_time = time.time()

        try:
            proc_result = subprocess.run(
                [str(self.dice_binary), '-i', str(input_xml)],
                cwd=dice_dir,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per dataset
            )

            elapsed = time.time() - start_time
            result['dice_time'] = elapsed

            if proc_result.returncode == 0:
                # Count output files
                solution_files = list(dice_dir.glob("DICe_solution_*.txt"))
                result['dice'] = True
                result['solution_files'] = len(solution_files)
                print(f"        Done: {len(solution_files)} solution files")
                print(f"        Time: {elapsed/60:.1f} minutes")
            else:
                result['error'] = f"DICe failed: {proc_result.stderr[:200]}"
                print(f"        ERROR: DICe returned non-zero exit code")
                print(f"        {proc_result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            result['error'] = "DICe timed out (>2 hours)"
            print(f"        ERROR: DICe timed out")
        except Exception as e:
            result['error'] = f"DICe execution failed: {e}"
            print(f"        ERROR: {e}")

        return result

    def run(self, subset_size=41, step_size=50, limit=None, skip_existing=True):
        """Run the complete batch processing pipeline."""
        print("="*60)
        print("BATCH PROCESSING PIPELINE")
        print("="*60)
        print(f"Source: {self.source_root}")
        print(f"Output: {self.output_root}")
        print(f"DICe:   {self.dice_binary}")
        print()

        # Find datasets
        print("Scanning for datasets...")
        datasets = self.find_datasets()

        if not datasets:
            print("No valid datasets found!")
            return

        print(f"\nFound {len(datasets)} datasets to process:")
        for ds in datasets:
            print(f"  - {ds['name']} ({ds['image_count']} images)")

        # Apply limit if specified
        if limit:
            datasets = datasets[:limit]
            print(f"\nLimiting to first {limit} datasets")

        # Skip existing if requested
        if skip_existing:
            filtered = []
            for ds in datasets:
                output_dir = self.output_root / ds['safe_name'] / "dice_output"
                solution_files = list(output_dir.glob("DICe_solution_*.txt")) if output_dir.exists() else []
                if len(solution_files) > 0:
                    print(f"\nSkipping {ds['name']} (already has {len(solution_files)} solution files)")
                else:
                    filtered.append(ds)
            datasets = filtered

        if not datasets:
            print("\nNo new datasets to process!")
            return

        # Create output root
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Process each dataset
        results = []
        start_time = datetime.now()

        for i, ds in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}] ", end="")
            result = self.process_dataset(ds, subset_size, step_size)
            results.append(result)

        # Print summary
        elapsed = datetime.now() - start_time
        self.print_summary(results, elapsed)

        # Save results to JSON
        results_file = self.output_root / "batch_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': elapsed.total_seconds(),
                'results': results
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

    def print_summary(self, results, elapsed):
        """Print final summary table."""
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {elapsed}")
        print()

        # Count successes
        preprocess_ok = sum(1 for r in results if r['preprocess'])
        config_ok = sum(1 for r in results if r['config'])
        dice_ok = sum(1 for r in results if r['dice'])

        print(f"{'Dataset':<20} {'Images':>7} {'Preprocess':>11} {'Config':>8} {'DICe':>8}")
        print("-" * 60)

        for r in results:
            prep = "OK" if r['preprocess'] else "FAIL"
            conf = "OK" if r['config'] else "FAIL"
            dice = "OK" if r['dice'] else "FAIL"
            print(f"{r['name']:<20} {r['image_count']:>7} {prep:>11} {conf:>8} {dice:>8}")

        print("-" * 60)
        print(f"{'TOTALS':<20} {'':<7} {preprocess_ok}/{len(results):>8} "
              f"{config_ok}/{len(results):>5} {dice_ok}/{len(results):>5}")

        # List any errors
        errors = [r for r in results if r['error']]
        if errors:
            print(f"\nERRORS ({len(errors)}):")
            for r in errors:
                print(f"  {r['name']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch process all SEM image sequences through DICe pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets
  %(prog)s

  # Process with custom parameters
  %(prog)s --subset-size 51 --step-size 40

  # Process only first 2 datasets (for testing)
  %(prog)s --limit 2

  # Reprocess even if output exists
  %(prog)s --no-skip-existing
        """
    )

    default_source = "~/Documents/Capstone/Graphite_image_sequences"
    default_output = "~/Documents/Capstone/dice-automation-tools/processed_datasets"
    default_dice = "~/Documents/Capstone/dice/build/bin/dice"

    parser.add_argument('--source', default=default_source,
                        help=f'Source directory (default: {default_source})')
    parser.add_argument('--output', default=default_output,
                        help=f'Output directory (default: {default_output})')
    parser.add_argument('--dice', default=default_dice,
                        help=f'DICe binary path (default: {default_dice})')
    parser.add_argument('--subset-size', type=int, default=41,
                        help='DICe subset size (default: 41)')
    parser.add_argument('--step-size', type=int, default=50,
                        help='DICe step size (default: 50)')
    parser.add_argument('--limit', type=int,
                        help='Limit number of datasets to process')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Reprocess datasets that already have output')

    args = parser.parse_args()

    try:
        processor = BatchProcessor(
            args.source,
            args.output,
            args.dice
        )

        processor.run(
            subset_size=args.subset_size,
            step_size=args.step_size,
            limit=args.limit,
            skip_existing=not args.no_skip_existing
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
