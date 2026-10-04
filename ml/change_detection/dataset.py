from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRCDDataset(Dataset):
    """
    PyTorch dataset for LEVIR-CD+.

    Directory structure:

        LEVIR-CD+/
        ├── train/
        │   ├── A/
        │   ├── B/
        │   └── label/
        └── test/
            ├── A/
            ├── B/
            └── label/

    A     -> before image
    B     -> after image
    label -> binary change mask
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
    ) -> None:

        self.root = Path(root)
        self.split = split

        self.split_dir = (
            self.root / split
        )

        self.a_dir = (
            self.split_dir / "A"
        )

        self.b_dir = (
            self.split_dir / "B"
        )

        self.label_dir = (
            self.split_dir / "label"
        )

        if not self.a_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {self.a_dir}"
            )

        if not self.b_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {self.b_dir}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {self.label_dir}"
            )

        self.samples = []

        for a_path in sorted(
            self.a_dir.glob("*.png")
        ):

            filename = a_path.name

            b_path = self.b_dir / filename
            label_path = self.label_dir / filename

            if not b_path.exists():
                raise FileNotFoundError(
                    f"Missing B image for {filename}"
                )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label for {filename}"
                )

            self.samples.append(
                (
                    a_path,
                    b_path,
                    label_path,
                )
            )

        if not self.samples:
            raise RuntimeError(
                f"No PNG samples found in {self.a_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(
        path: Path,
    ) -> torch.Tensor:

        image = Image.open(path).convert("RGB")

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        # Convert HWC -> CHW
        array = np.transpose(
            array,
            (2, 0, 1),
        )

        # Normalize uint8 RGB to [0, 1]
        array /= 255.0

        return torch.from_numpy(
            array
        )

    @staticmethod
    def _load_label(
        path: Path,
    ) -> torch.Tensor:

        label = Image.open(path).convert("L")

        array = np.asarray(
            label,
            dtype=np.uint8,
        )

        # Convert all non-zero pixels to 1.
        array = (
            array > 0
        ).astype(np.float32)

        return torch.from_numpy(
            array
        ).unsqueeze(0)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:

        a_path, b_path, label_path = (
            self.samples[index]
        )

        before = self._load_rgb(
            a_path
        )

        after = self._load_rgb(
            b_path
        )

        label = self._load_label(
            label_path
        )

        return {
            "before": before,
            "after": after,
            "label": label,
            "filename": a_path.name,
        }