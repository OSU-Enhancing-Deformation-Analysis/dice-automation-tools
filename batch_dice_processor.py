#!/usr/bin/env python3
"""
Batch DICe Processor
Process multiple image sequences with DICe in batch mode
"""
import argparse
import subprocess
import json
from pathlib import Path
from dice_config_generator import DICeConfigGenerator

class BatchDICeProcessor:
    def __init__(self, dice_binary, sequences_root, output_root):
        """Initialize paths and verify DICe binary exists."""
        self.dice_binary = Path(dice_binary)
        self.sequences_root = Path(sequences_root)
        self.output_root = Path(output_root)
        
        # Ensure the executable exists before starting
        if not self.dice_binary.exists():
            raise FileNotFoundError(f"DICe binary not found: {self.dice_binary}")
    
    def find_sequences(self):
        """Scan for image sequences in root directory."""
        sequences = []
        # Recursively search for all .tif files
        for item in self.sequences_root.rglob("*.tif"):
            parent = item.parent
            # Check if this folder is already in our list
            if parent not in [s['path'] for s in sequences]:
                # Count tif files in this folder
                tif_count = len(list(parent.glob("*.tif")))
                if tif_count >= 10:  # Only process folders with enough images
                    sequences.append({
                        'name': parent.name,
                        'path': parent,
                        'image_count': tif_count
                    })
        return sequences
    
    def process_sequence(self, seq_info, subset_size=41, step_size=50):
        """Generate config and run DICe for one sequence."""
        seq_name = seq_info['name']
        config_dir = self.output_root / seq_name
        
        print(f"\n{'='*60}")
        print(f"Processing: {seq_name}")
        print(f"  Images: {seq_info['image_count']}")
        print(f"  Config output: {config_dir}")
        
        # Step 1: Generate configuration files using the Generator class
        generator = DICeConfigGenerator(
            seq_info['path'],
            config_dir,
            subset_size=subset_size,
            step_size=step_size
        )
        
        try:
            config_info = generator.generate()
            print(f"  Configuration generated successfully")
        except Exception as e:
            print(f"  ERROR: Failed to generate config: {e}")
            return False
        
        # Step 2: Run DICe executable using subprocess
        input_xml = Path(config_info['input_xml'])
        print(f"  Running DICe...")
        
        try:
            result = subprocess.run(
                [str(self.dice_binary), '-i', str(input_xml)],
                cwd=config_dir,       # Run inside the output directory
                capture_output=True,  # Capture stdout/stderr
                text=True,            # Return strings instead of bytes
                timeout=3600          # Kill if runs longer than 1 hour
            )
            
            if result.returncode == 0:
                print(f"  SUCCESS: DICe completed")
                return True
            else:
                print(f"  ERROR: DICe failed")
                print(f"  {result.stderr[:500]}") # Print start of error message
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ERROR: DICe timed out (>1 hour)")
            return False
        except Exception as e:
            print(f"  ERROR: {e}")
            return False
    
    def run(self, subset_size=41, step_size=50, limit=None):
        """Process all found sequences."""
        sequences = self.find_sequences()
        
        if not sequences:
            print("No image sequences found")
            return
        
        print(f"Found {len(sequences)} sequences")
        
        # Optional limit for testing (e.g., only process first 3)
        if limit:
            sequences = sequences[:limit]
            print(f"Processing first {limit} sequences")
        
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        results = []
        # Main processing loop
        for seq in sequences:
            success = self.process_sequence(seq, subset_size, step_size)
            results.append({
                'sequence': seq['name'],
                'success': success
            })
        
        # Print final summary table
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        success_count = sum(1 for r in results if r['success'])
        print(f"Total: {len(results)}")
        print(f"Success: {success_count}")
        print(f"Failed: {len(results) - success_count}")
        
        for r in results:
            status = "OK" if r['success'] else "FAILED"
            print(f"  {r['sequence']}: {status}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch process sequences with DICe')
    parser.add_argument('dice_binary', help='Path to DICe executable')
    parser.add_argument('sequences_root', help='Root directory containing sequences')
    parser.add_argument('output_root', help='Output directory for configs and results')
    parser.add_argument('--subset-size', type=int, default=41)
    parser.add_argument('--step-size', type=int, default=50)
    parser.add_argument('--limit', type=int, help='Limit number of sequences')
    
    args = parser.parse_args()
    
    processor = BatchDICeProcessor(
        args.dice_binary,
        args.sequences_root,
        args.output_root
    )
    
    processor.run(
        subset_size=args.subset_size,
        step_size=args.step_size,
        limit=args.limit
    )

if __name__ == "__main__":
    main()