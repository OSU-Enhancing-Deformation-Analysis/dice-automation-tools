# DICe Automation Tools

Automation scripts for batch processing SEM image sequences with the Digital Image Correlation Engine (DICe), and for generating training data for ML-based deformation analysis.

## Purpose

This toolkit automates the DICe workflow for generating ground-truth displacement fields from SEM image sequences, then converts those fields into training tiles for the RAFT optical-flow model in the Enhancing Deformation Analysis project.

## Pipeline Overview

Per experimental dataset:

1. **Preprocess** raw SEM images (strip metadata bars, standardize format).
2. **Explore frame intervals** with a middle-frame reference to find usable displacement ranges.
3. **Run DICe** with `step_size=1`, `subset_size=35`, and a middle-frame reference to produce dense displacement fields.
4. **Trim the unreliable border zone** (~17 px, half the subset size).
5. **Generate training tiles** with systematic grid coverage and 8-way D4 augmentation.
6. **(Optional) Stitch tiles back** into per-pair absolute-coordinate fields for visualization or RAFT input.

## Tools

### Configuration Generation
- **dice_config_generator.py** — Generates DICe configuration files (`input.xml`, `params.xml`, `subsets.txt`) for an image sequence.

### Data Processing Pipeline
- **preprocess_sem_images.py** — Removes SEM metadata bars and standardizes image format for DICe.
- **explore_frame_intervals.py** — Runs DICe across multiple frame intervals (with middle-frame reference) to identify usable displacement ranges before generating data at scale.
- **batch_dice_processor.py** — Runs DICe in batch over multiple image sequences.
- **batch_process_all.py** — End-to-end pipeline: preprocess → generate config → run DICe → analyze.

### Training Tile Generation
- **generate_tiles_systematic.py** — *(primary)* Generates 128×128 training tiles using systematic grid coverage with anchored last-tile placement (full reliable-zone coverage) and 8-way D4 augmentation. This is the script used for the final training data.
- **generate_tiles.py** — Earlier random-crop variant; kept for reference and ablation comparison.
- **real_dataset.py** — PyTorch `Dataset` wrapping the generated tile triplets for RAFT training.
- **npy_to_dice_txt.py** — Converts tile `flow.npy` files back to DICe-native `.txt` format for downstream consumers that expect SIGMA/GAMMA fields.

### Stitching and Full-Field Reconstruction
- **merge_pair_absolute.py** — Merges all tiles of one pair into a single DICe-format `.txt` with absolute image-space coordinates.
- **merge_pair_absolute_grid.py** — Same as above but emits a complete grid (failed pixels marked `SIGMA=-1`) for fast heatmap visualization.
- **make_systematic_gif.py** — Renders the merged absolute `.txt` files as an animated GIF showing displacement-field progression across pairs.

### Quality Analysis and Visualization
- **analyze_dice_quality.py** — Analyzes DICe output quality via SIGMA, GAMMA, and MATCH metrics.
- **visualize_dice_output.py** — Visualizes displacement vectors and strain fields from DICe output.
- **visualize_dice_animation.py** — Animated GIF of displacement-field evolution across frames.
- **visualize_exploration.py** — Plots from `explore_frame_intervals.py` results (success/fail spatial map, displacement field, interval comparison).
- **visualize_tiles.py** — Summary plots for a `training_tiles_128_*` output root.

## Key Parameters

The defaults below were validated against the ESWG007 dataset and are documented in `DICE_PARAMETERS.md`:

- `subset_size=35`, `step_size=1`
- `sssig_threshold=144`, `strain_window=60`
- `GRADIENT_BASED` initialization with `USE_FEATURE_MATCHING` and rotation enabled
- **Middle-frame reference** (not first-frame) — gives reliable results across all displacement intervals
- Border trim: ~17 px (half the subset size) before tile extraction

## Requirements

- Python 3.10+
- NumPy, SciPy, Pandas
- Pillow, Matplotlib, imageio
- DICe (compiled binary) — see https://github.com/dicengine/dice

## Installation

```bash
pip install -r requirements.txt
```

A working DICe binary must be on `PATH` or referenced explicitly in the generated configs.

## Project Context

Part of OSU Capstone CS.057: *Applying a Machine Learning-powered Localized Deformation Analyzer for Digital Twin Applications in Materials Testing.*

**Related Repositories (under [OSU-Enhancing-Deformation-Analysis](https://github.com/OSU-Enhancing-Deformation-Analysis)):**
- [Deban-API](https://github.com/OSU-Enhancing-Deformation-Analysis/Deban-API) — Deben mechanical-tester control panel and GUI.
- [ML-Model](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model) — RAFT model implementation (maintained by research partner Brock Cloutier).

## Contributors

- Yanghui Ren

## License

MIT [LICENSE](./LICENSE).
