# DICe Automation Tools

Automation scripts for batch processing SEM image sequences with Digital Image Correlation Engine (DICe).

## Purpose

This toolkit automates the DICe workflow for generating ground truth displacement data from SEM image sequences, intended for training ML models in the Enhancing Deformation Analysis project.

## Tools

### Configuration Generation

**dice_config_generator.py** - Automatically generates DICe configuration files (input.xml, params.xml, subsets.txt) for image sequences.

### Data Processing Pipeline

**preprocess_sem_images.py** - Removes SEM metadata bars from images and standardizes format for DICe processing.

**dice_to_dense.py** - Converts sparse DICe output (~340 points) to dense per-pixel displacement fields (~900k points) using interpolation.

**batch_dice_to_dense.py** - Batch version of dice_to_dense.py for processing multiple sequences.

**batch_process_all.py** - Complete end-to-end pipeline: preprocessing → DICe → dense conversion.

### Quality Analysis

**analyze_dice_quality.py** - Analyzes DICe output quality using SIGMA, GAMMA, and MATCH metrics.

### Visualization

**visualize_dice_output.py** - Visualizes displacement vectors and strain fields from DICe output.

**visualize_dice_animation.py** - Generates animated GIF showing displacement field evolution.

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- Pillow, Matplotlib, imageio
- DICe (compiled binary)

## Installation
```bash
pip install -r requirements.txt
```

## Project Context

Part of OSU Capstone CS.057: ML-Powered Digital Twin for Material Deformation Analysis.

**Related Repositories:**
- [EnhancingDeformationAnalysisUI](https://github.com/OSU-Enhancing-Deformation-Analysis/EnhancingDeformationAnalysisUI)
- [ML-Model](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model)

## Contributors

- Yanghui Ren

## License

MIT [LICENSE](./LICENSE).