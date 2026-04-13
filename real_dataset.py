#!/usr/bin/env python3
"""
real_dataset.py
PyTorch Dataset wrapping the tile triplets produced by generate_tiles.py.

Output is format-compatible with Synthetic-Data/m4.py CustomDataset so the
same model can be trained/evaluated on synthetic vs real data:

    input      : float32 (2, H, W)  ch0=reference tile, ch1=deformed tile, 0-255
    target     : float32 (2, H, W)  ch0=dU (dx), ch1=dV (dy), scaled by 1/10
    valid_mask : bool    (H, W)     True where DICe produced a finite value

The 1/10 scale mirrors m4.py, which warps with `pos - dU * 10`, so the model
internally learns displacements in units of pixel/10 rather than raw pixels.
The mask covers the ~65% NaN pixels DICe leaves behind; expected usage:

    loss = L1(pred * mask, target * mask)
"""

from pathlib import Path
from typing import List, Tuple

import cv2  # pylint: disable=import-error,no-name-in-module
import numpy as np
from torch.utils.data import Dataset

# cv2 is a C extension and pylint cannot introspect its members.
# pylint: disable=no-member

FLOW_SCALE = 10.0


class RealDiceDataset(Dataset):
    """Load (ref, def, flow) tile triplets written by generate_tiles.py."""

    def __init__(self, tiles_dir: str, augment: bool = False) -> None:
        self.tiles_dir = Path(tiles_dir)
        if not self.tiles_dir.is_dir():
            raise FileNotFoundError(f"tiles_dir does not exist: {self.tiles_dir}")

        self.augment = augment
        self.samples: List[Tuple[Path, Path, Path]] = []

        for pair_dir in sorted(self.tiles_dir.iterdir()):
            if not pair_dir.is_dir() or not pair_dir.name.startswith("pair_"):
                continue
            ref_dir = pair_dir / "ref"
            def_dir = pair_dir / "def"
            flow_dir = pair_dir / "flow"
            if not (ref_dir.is_dir() and def_dir.is_dir() and flow_dir.is_dir()):
                continue

            for flow_path in sorted(flow_dir.glob("*.npy")):
                tile_name = flow_path.stem
                ref_path = ref_dir / f"{tile_name}.tif"
                def_path = def_dir / f"{tile_name}.tif"
                if ref_path.is_file() and def_path.is_file():
                    self.samples.append((ref_path, def_path, flow_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref_path, def_path, flow_path = self.samples[index]

        ref = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
        deformed = cv2.imread(str(def_path), cv2.IMREAD_UNCHANGED)
        if ref is None or deformed is None:
            raise IOError(f"Failed to read tile pair: {ref_path}, {def_path}")

        ref = ref.astype(np.float32)
        deformed = deformed.astype(np.float32)

        flow = np.load(flow_path).astype(np.float32)  # (H, W, 2), channels [dx, dy]
        valid_mask = ~np.isnan(flow[..., 0])
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            ref, deformed, flow, valid_mask = _apply_augmentation(
                ref, deformed, flow, valid_mask
            )

        image_input = np.stack([ref, deformed], axis=0)  # (2, H, W)
        target = flow.transpose(2, 0, 1) / FLOW_SCALE  # (2, H, W)

        return (
            image_input.astype(np.float32),
            target.astype(np.float32),
            valid_mask.astype(bool),
        )


def _apply_augmentation(
    ref: np.ndarray,
    deformed: np.ndarray,
    flow: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random flips and 90-degree rotations applied jointly to images + flow.

    Flow channels [dx, dy] follow image coordinates (x=col, y=row). Sign and
    axis swaps are chosen so that after the transform the vectors still point
    from reference to deformed in the new frame. The valid_mask is spatially
    transformed alongside (no sign handling, it is a boolean).
    """
    if np.random.rand() < 0.5:
        # Horizontal flip (flip columns): dx -> -dx, dy unchanged
        ref = np.fliplr(ref).copy()
        deformed = np.fliplr(deformed).copy()
        flow = np.fliplr(flow).copy()
        valid_mask = np.fliplr(valid_mask).copy()
        flow[..., 0] = -flow[..., 0]

    if np.random.rand() < 0.5:
        # Vertical flip (flip rows): dy -> -dy, dx unchanged
        ref = np.flipud(ref).copy()
        deformed = np.flipud(deformed).copy()
        flow = np.flipud(flow).copy()
        valid_mask = np.flipud(valid_mask).copy()
        flow[..., 1] = -flow[..., 1]

    k = int(np.random.randint(0, 4))
    if k > 0:
        ref = np.rot90(ref, k=k).copy()
        deformed = np.rot90(deformed, k=k).copy()
        flow = np.rot90(flow, k=k).copy()
        valid_mask = np.rot90(valid_mask, k=k).copy()
        flow = _rotate_flow_vectors(flow, k)

    return ref, deformed, flow, valid_mask


def _rotate_flow_vectors(flow: np.ndarray, k: int) -> np.ndarray:
    """Rotate the (dx, dy) vector stored at each pixel to match np.rot90(k).

    np.rot90 with the default axes moves pixel (r, c) on an (H, W) image to
    (W-1-c, r) for k=1. Applying the same linear map to (dx, dy):
        k=1: (dx, dy) -> ( dy, -dx)
        k=2: (dx, dy) -> (-dx, -dy)
        k=3: (dx, dy) -> (-dy,  dx)
    """
    k = k % 4
    if k == 0:
        return flow
    dx = flow[..., 0]
    dy = flow[..., 1]
    if k == 1:
        new_dx, new_dy = dy, -dx
    elif k == 2:
        new_dx, new_dy = -dx, -dy
    else:  # k == 3
        new_dx, new_dy = -dy, dx
    return np.stack([new_dx, new_dy], axis=-1)


def _main() -> int:
    tiles_dir = Path(__file__).parent / "training_tiles_128"
    print(f"Loading dataset from: {tiles_dir}")

    dataset = RealDiceDataset(str(tiles_dir), augment=False)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        print("No tiles found.")
        return 1

    image_input, target, valid_mask = dataset[0]
    print(f"input      shape={image_input.shape} dtype={image_input.dtype} "
          f"min={image_input.min():.2f} max={image_input.max():.2f}")
    print(f"target     shape={target.shape} dtype={target.dtype} "
          f"min={target.min():.4f} max={target.max():.4f}")
    valid_ratio = valid_mask.mean()
    print(f"valid_mask shape={valid_mask.shape} dtype={valid_mask.dtype} "
          f"valid_ratio={valid_ratio:.3f}")

    assert image_input.shape == (2, 128, 128)
    assert target.shape == (2, 128, 128)
    assert valid_mask.shape == (128, 128)
    assert image_input.dtype == np.float32
    assert target.dtype == np.float32
    assert valid_mask.dtype == np.bool_
    assert not np.isnan(target).any(), "target contains NaN"
    assert not np.isnan(image_input).any(), "input contains NaN"

    augmented = RealDiceDataset(str(tiles_dir), augment=True)
    aug_input, aug_target, aug_mask = augmented[0]
    assert aug_input.shape == (2, 128, 128)
    assert aug_target.shape == (2, 128, 128)
    assert aug_mask.shape == (128, 128)
    assert aug_mask.dtype == np.bool_
    print("Augmented sample shapes OK.")

    print("Format matches m4.py CustomDataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
