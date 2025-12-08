# DICe Parameters Documentation

This document explains key DICe parameters and their impact on displacement/strain data generation for ML training.

## Key Parameters for ML Training Data

### Image Processing Parameters (input.xml)

| Parameter | Current Value | Description | Impact on Training Data |
|-----------|---------------|-------------|------------------------|
| `subset_size` | 41 | Size of correlation subset (pixels) | Smaller = more detail but more noise; Larger = smoother but less detail |
| `step_size` | 50 | Spacing between tracking points | Smaller = denser displacement field; Larger = sparser but faster |

### Correlation Parameters (params.xml)

| Parameter | Current Value | Description | Impact on Training Data |
|-----------|---------------|-------------|------------------------|
| `optimization_method` | SIMPLEX | Nelder-Mead simplex optimization | Sub-pixel accuracy for displacement |
| `interpolation_method` | KEYS_FOURTH | Keys fourth-order interpolation | Higher accuracy than bilinear |
| `enable_translation` | true | Track X/Y displacement | Required for displacement output |
| `enable_rotation` | false | Track rotation | Disabled - assumes no rotation |
| `enable_normal_strain` | true | Calculate normal strain | Outputs VSG_STRAIN_XX, VSG_STRAIN_YY |
| `enable_shear_strain` | true | Calculate shear strain | Outputs VSG_STRAIN_XY |
| `strain_window_size_in_pixels` | 51 | Window for strain calculation | Larger = smoother strain field |

### Output Fields

| Field | Description | Use for ML |
|-------|-------------|------------|
| `COORDINATE_X/Y` | Point location (pixels) | Input coordinates |
| `DISPLACEMENT_X/Y` | Motion vectors (pixels) | **Primary training labels** |
| `VSG_STRAIN_XX/YY/XY` | Strain tensor components | Secondary labels |
| `SIGMA` | Correlation confidence | Filter low-quality points |
| `GAMMA` | Gradient quality metric | Filter low-contrast regions |
| `MATCH` | Match quality (0-1) | Filter failed correlations |

## Recommended Settings for Training Data Generation

For high-quality ground truth data:
- `subset_size`: 31-51 (balance detail vs. noise)
- `step_size`: 25-50 (denser is better for training)
- Use SIGMA > 0.1 to filter unreliable points
- Use MATCH > 0.5 to filter failed correlations

## References

- [DICe Documentation](https://github.com/dicengine/dice)
- [DICe User Guide](https://dicengine.github.io/dice/)
