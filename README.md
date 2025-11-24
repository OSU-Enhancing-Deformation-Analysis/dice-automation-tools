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

## Requirements

- Python 3.8+
- Pillow (PIL)
- DICe (compiled binary)

## Installation
```bash
pip install Pillow
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