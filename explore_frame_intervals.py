#!/usr/bin/env python3
"""
DICe Parameter Exploration: step_size=1 with varying frame intervals.

Creates DICe configs for image pairs at different intervals and runs them in parallel.
Purpose: Determine optimal frame separation for meaningful displacement data.

Usage:
    python3 explore_frame_intervals.py --image_dir /path/to/preprocessed/ --output_dir ./exploration_results
    python3 explore_frame_intervals.py --image_dir /path/to/preprocessed/ --output_dir ./exploration_results --dry_run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# Dr. Chen's DICe parameters (from reference DIC files)
PARAMS_XML = """\
<ParameterList>
<Parameter name="interpolation_method" type="string" value="KEYS_FOURTH" />
<Parameter name="sssig_threshold" type="double" value="144" />
<Parameter name="optimization_method" type="string" value="GRADIENT_BASED" />
<Parameter name="initialization_method" type="string" value="USE_FEATURE_MATCHING" />
<Parameter name="enable_translation" type="bool" value="true" />
<Parameter name="enable_rotation" type="bool" value="true" />
<Parameter name="enable_normal_strain" type="bool" value="true" />
<Parameter name="enable_shear_strain" type="bool" value="true" />
<ParameterList name="post_process_vsg_strain">
<Parameter name="strain_window_size_in_pixels" type="int" value="60" />
</ParameterList>
<Parameter name="output_delimiter" type="string" value="," />
<ParameterList name="output_spec">
<Parameter name="COORDINATE_X" type="bool" value="true" />
<Parameter name="COORDINATE_Y" type="bool" value="true" />
<Parameter name="DISPLACEMENT_X" type="bool" value="true" />
<Parameter name="DISPLACEMENT_Y" type="bool" value="true" />
<Parameter name="SIGMA" type="bool" value="true" />
<Parameter name="GAMMA" type="bool" value="true" />
<Parameter name="BETA" type="bool" value="true" />
<Parameter name="STATUS_FLAG" type="bool" value="true" />
<Parameter name="UNCERTAINTY" type="bool" value="true" />
<Parameter name="VSG_STRAIN_XX" type="bool" value="true" />
<Parameter name="VSG_STRAIN_YY" type="bool" value="true" />
<Parameter name="VSG_STRAIN_XY" type="bool" value="true" />
</ParameterList>
</ParameterList>
"""

# ROI covering most of the 1024x943 image with margin for subset_size=35
SUBSETS_TXT = """\
begin region_of_interest
  begin boundary
    begin rectangle
      center 512 441
      width 940
      height 800
    end rectangle
  end boundary
end region_of_interest
"""


def generate_input_xml(image_dir, output_dir, ref_index, def_index,
                       subset_size=35, step_size=1, num_digits=4,
                       prefix="frame_", extension=".tif"):
    """Generate input.xml for a single image pair."""
    return f"""\
<ParameterList>
<Parameter name="output_folder" type="string" value="{output_dir}/" />
<Parameter name="correlation_parameters_file" type="string" value="{output_dir}/params.xml" />
<Parameter name="subset_file" type="string" value="{output_dir}/subsets.txt" />
<Parameter name="subset_size" type="int" value="{subset_size}" />
<Parameter name="step_size" type="int" value="{step_size}" />
<Parameter name="separate_output_file_for_each_subset" type="bool" value="false" />
<Parameter name="create_separate_run_info_file" type="bool" value="true" />
<Parameter name="image_folder" type="string" value="{image_dir}/" />
<Parameter name="reference_image_index" type="int" value="{ref_index}" />
<Parameter name="start_image_index" type="int" value="{def_index}" />
<Parameter name="end_image_index" type="int" value="{def_index}" />
<Parameter name="num_file_suffix_digits" type="int" value="{num_digits}" />
<Parameter name="image_file_extension" type="string" value="{extension}" />
<Parameter name="image_file_prefix" type="string" value="{prefix}" />
</ParameterList>
"""


def create_pair_config(image_dir, output_base, ref_frame, def_frame, step_size=1):
    """Create a config directory for one image pair."""
    pair_name = f"pair_f{ref_frame:04d}_vs_f{def_frame:04d}"
    pair_dir = os.path.join(output_base, pair_name)
    os.makedirs(pair_dir, exist_ok=True)

    # Write configs
    with open(os.path.join(pair_dir, "input.xml"), "w") as f:
        f.write(generate_input_xml(image_dir, pair_dir, ref_frame, def_frame,
                                   step_size=step_size))

    with open(os.path.join(pair_dir, "params.xml"), "w") as f:
        f.write(PARAMS_XML)

    with open(os.path.join(pair_dir, "subsets.txt"), "w") as f:
        f.write(SUBSETS_TXT)

    return pair_name, pair_dir


def run_dice(pair_name, pair_dir, dice_exe):
    """Run DICe on a single image pair. Returns (pair_name, elapsed_seconds, success)."""
    input_xml = os.path.join(pair_dir, "input.xml")
    start = time.time()
    try:
        result = subprocess.run(
            [dice_exe, "-i", input_xml, "-v", "-t"],
            capture_output=True, text=True, timeout=7200  # 2 hour timeout
        )
        elapsed = time.time() - start
        success = "Successful Completion" in result.stdout
        if not success:
            # Save error log
            with open(os.path.join(pair_dir, "error.log"), "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
        return pair_name, elapsed, success
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return pair_name, elapsed, False
    except Exception as e:
        elapsed = time.time() - start
        with open(os.path.join(pair_dir, "error.log"), "w") as f:
            f.write(f"Exception: {e}\n")
        return pair_name, elapsed, False


def main():
    parser = argparse.ArgumentParser(
        description="Run DICe exploration on image pairs at different frame intervals")
    parser.add_argument("--image_dir", required=True,
                        help="Path to preprocessed images (e.g., processed_datasets/A6_Video/preprocessed/)")
    parser.add_argument("--output_dir", default="./exploration_results",
                        help="Base output directory")
    parser.add_argument("--dice_exe", default=os.path.expanduser(
                        "~/Documents/Capstone/dice/build/bin/dice"),
                        help="Path to DICe executable")
    parser.add_argument("--ref_frame", type=int, default=1,
                        help="Reference frame number (default: 1)")
    parser.add_argument("--intervals", type=int, nargs="+",
                        default=[1, 5, 10, 20, 30],
                        help="Frame intervals to test (default: 1 5 10 20 30)")
    parser.add_argument("--include_last", action="store_true", default=True,
                        help="Also test ref vs last frame")
    parser.add_argument("--last_frame", type=int, default=None,
                        help="Last frame number (auto-detected if not set)")
    parser.add_argument("--step_size", type=int, default=1,
                        help="DICe step_size (default: 1)")
    parser.add_argument("--max_parallel", type=int, default=None,
                        help="Max parallel DICe processes (default: number of pairs)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only generate configs, don't run DICe")
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)
    dice_exe = os.path.abspath(args.dice_exe)

    # Validate
    if not os.path.isdir(image_dir):
        print(f"Error: image_dir not found: {image_dir}")
        sys.exit(1)
    if not os.path.isfile(dice_exe):
        print(f"Error: DICe executable not found: {dice_exe}")
        sys.exit(1)

    # Auto-detect last frame
    if args.last_frame is None:
        tif_files = sorted(Path(image_dir).glob("frame_*.tif"))
        if not tif_files:
            print(f"Error: no frame_*.tif files found in {image_dir}")
            sys.exit(1)
        last_frame = int(tif_files[-1].stem.split("_")[-1])
    else:
        last_frame = args.last_frame

    # Build list of deformed frame numbers
    def_frames = []
    for interval in args.intervals:
        f = args.ref_frame + interval
        if f <= last_frame:
            def_frames.append(f)
    if args.include_last and last_frame not in def_frames:
        def_frames.append(last_frame)

    def_frames = sorted(set(def_frames))

    print(f"{'='*60}")
    print(f"DICe Frame Interval Exploration")
    print(f"{'='*60}")
    print(f"Image directory : {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"DICe executable : {dice_exe}")
    print(f"Reference frame : {args.ref_frame}")
    print(f"Last frame      : {last_frame}")
    print(f"Step size       : {args.step_size}")
    print(f"Pairs to run    : {len(def_frames)}")
    for f in def_frames:
        interval = f - args.ref_frame
        print(f"  frame {args.ref_frame:04d} vs {f:04d} (interval={interval})")
    print(f"{'='*60}")

    # Create configs
    pairs = []
    for def_frame in def_frames:
        pair_name, pair_dir = create_pair_config(
            image_dir, output_dir, args.ref_frame, def_frame,
            step_size=args.step_size)
        pairs.append((pair_name, pair_dir))
        print(f"Created config: {pair_name}")

    if args.dry_run:
        print(f"\n--dry_run: configs created in {output_dir}, not running DICe.")
        return

    # Run DICe in parallel
    max_workers = args.max_parallel or len(pairs)
    print(f"\nStarting {len(pairs)} DICe processes (max {max_workers} parallel)...")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_dice, name, pdir, dice_exe): name
            for name, pdir in pairs
        }
        for future in as_completed(futures):
            pair_name, elapsed, success = future.result()
            status = "OK" if success else "FAILED"
            minutes = elapsed / 60
            print(f"  [{status}] {pair_name} — {minutes:.1f} min")
            results.append((pair_name, elapsed, success))

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All done! Total wall time: {total_elapsed/60:.1f} min")
    print(f"{'='*60}")

    # Summary
    print(f"\n{'Pair':<35} {'Time':>8} {'Status':>8}")
    print(f"{'-'*55}")
    for pair_name, elapsed, success in sorted(results):
        status = "OK" if success else "FAIL"
        print(f"{pair_name:<35} {elapsed/60:>7.1f}m {status:>8}")

    # Check for output files
    print(f"\nOutput files:")
    for pair_name, pair_dir in sorted(pairs):
        txt_files = list(Path(pair_dir).glob("DICe_solution_*.txt"))
        print(f"  {pair_name}: {len(txt_files)} solution file(s)")


if __name__ == "__main__":
    main()