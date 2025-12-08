# DICe Automation Tools

Automation scripts for batch processing SEM image sequences with Digital Image Correlation Engine (DICe).

## Purpose

This toolkit automates the DICe workflow for generating ground truth displacement data from SEM image sequences, intended for training ML models in the Enhancing Deformation Analysis project.

## Tools

### dice_config_generator.py

Automatically generates DICe configuration files for image sequences.

**Usage:**
```bash
python3 dice_config_generator.py <image_dir> <output_dir> [--subset-size SIZE] [--step-size SIZE]
```

**Example:**
```bash
python3 dice_config_generator.py \
    "/path/to/image/sequence" \
    "./output_config" \
    --subset-size 41 \
    --step-size 50
```

**Output:**
- `input.xml` - DICe input configuration
- `params.xml` - Correlation parameters  
- `subsets.txt` - Region of interest definition
- `sequence_info.json` - Sequence metadata

### visualize_dice_output.py

Visualize displacement vectors and strain fields from a single DICe output file.

**Usage:**
```bash
# Displacement vectors
python3 visualize_dice_output.py DICe_solution_050.txt --scale 5 --summary

# Strain field
python3 visualize_dice_output.py DICe_solution_050.txt --strain --strain-component VSG_STRAIN_XX

# Without background image
python3 visualize_dice_output.py DICe_solution_050.txt --no-background -o output.png
```

### visualize_dice_animation.py

Generate animated GIF showing displacement field evolution over an image sequence.

**Usage:**
```bash
python3 visualize_dice_animation.py \
    --image-dir /path/to/images \
    --dice-dir ./dice_output \
    --output animation.gif \
    --scale 5 --fps 5
```

## Requirements

- Python 3.8+
- Pillow (PIL)
- NumPy
- Pandas
- Matplotlib
- imageio
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
- [dice-model-comparison](https://github.com/OSU-Enhancing-Deformation-Analysis/dice-model-comparison)

## Contributors

- Yanghui Ren

## License

MIT [LICENSE](./LICENSE).