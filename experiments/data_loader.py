"""Unified crack dataset loader for CrackForest and DeepCrack datasets."""

import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import scipy.io

logger = logging.getLogger(__name__)


class CrackDataset:
    """Unified crack dataset loader.

    Supports "crackforest" and "deepcrack" datasets, returning image-mask pairs
    in a consistent format: (H,W,3) RGB uint8 image and (H,W) binary uint8 mask (0/255).
    """

    SUPPORTED_DATASETS = ("crackforest", "deepcrack")

    def __init__(self, dataset_name: str, root_dir: str) -> None:
        """Initialize dataset loader.

        Args:
            dataset_name: "crackforest" or "deepcrack"
            root_dir: Path to dataset root directory
        Raises:
            ValueError: If dataset_name is not recognized
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)
        self._samples: List[Tuple[str, str]] = []  # (image_path, gt_path)

        if dataset_name == "crackforest":
            self._scan_crackforest()
        else:
            self._scan_deepcrack()

    def _scan_crackforest(self) -> None:
        """Scan CrackForest dataset and pair image/GT files."""
        img_dir = self.root_dir / "image"
        gt_dir = self.root_dir / "groundTruth"

        if not img_dir.exists() or not gt_dir.exists():
            logger.warning("CrackForest image or groundTruth directory not found at %s", self.root_dir)
            return

        for img_file in sorted(img_dir.glob("*.jpg")):
            stem = img_file.stem
            gt_file = gt_dir / f"{stem}.mat"
            if gt_file.exists():
                self._samples.append((str(img_file), str(gt_file)))
            else:
                logger.warning("Missing GT for CrackForest image: %s", img_file.name)

        logger.info("CrackForest: loaded %d samples from %s", len(self._samples), self.root_dir)

    def _scan_deepcrack(self) -> None:
        """Scan DeepCrack dataset and pair image/GT files from train+test dirs."""
        base = self.root_dir / "dataset" / "DeepCrack"

        for split in ("train", "test"):
            img_dir = base / f"{split}_img"
            lab_dir = base / f"{split}_lab"
            if not img_dir.exists() or not lab_dir.exists():
                logger.warning("DeepCrack %s directory not found at %s", split, base)
                continue

            for img_file in sorted(img_dir.glob("*.jpg")):
                stem = img_file.stem
                lab_file = lab_dir / f"{stem}.png"
                if lab_file.exists():
                    self._samples.append((str(img_file), str(lab_file)))
                else:
                    logger.warning("Missing label for DeepCrack image: %s", img_file.name)

        logger.info("DeepCrack: loaded %d samples from %s", len(self._samples), self.root_dir)

    def __len__(self) -> int:
        """Return number of loaded samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (image, gt_mask) for the given index.

        Returns:
            image: (H, W, 3) uint8 RGB numpy array
            gt_mask: (H, W) uint8 binary mask, 0=background, 255=crack
        """
        img_path, gt_path = self._samples[idx]

        # Load image as RGB
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load ground truth
        if self.dataset_name == "crackforest":
            gt_mask = self._load_crackforest_gt(gt_path)
        else:
            gt_mask = self._load_deepcrack_gt(gt_path)

        return image, gt_mask

    @staticmethod
    def _load_crackforest_gt(gt_path: str) -> np.ndarray:
        """Load CrackForest .mat GT and convert to binary mask.

        .mat structure: mat['groundTruth'][0, 0]['Segmentation'] -> (H, W) uint8
        Values: 1 = background, 2 = crack
        """
        mat = scipy.io.loadmat(gt_path)
        seg = mat["groundTruth"][0, 0]["Segmentation"]
        return (seg == 2).astype(np.uint8) * 255

    @staticmethod
    def _load_deepcrack_gt(gt_path: str) -> np.ndarray:
        """Load DeepCrack .png GT and binarize at threshold 127."""
        label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        return ((label > 127).astype(np.uint8)) * 255

    def get_split(self, split: str) -> "CrackDataset":
        """Return train or test subset.

        CrackForest: 80/20 random split with fixed seed=42.
        DeepCrack: Uses existing train/test directory structure.

        Args:
            split: "train" or "test"
        Returns:
            A new CrackDataset containing only the split's samples.
        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        new_ds = object.__new__(CrackDataset)
        new_ds.dataset_name = self.dataset_name
        new_ds.root_dir = self.root_dir

        if self.dataset_name == "crackforest":
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(self._samples))
            split_idx = int(len(self._samples) * 0.8)
            if split == "train":
                new_ds._samples = [self._samples[i] for i in indices[:split_idx]]
            else:
                new_ds._samples = [self._samples[i] for i in indices[split_idx:]]
        else:
            # DeepCrack: filter by directory name
            base = self.root_dir / "dataset" / "DeepCrack"
            split_dir = str(base / f"{split}_img")
            new_ds._samples = [
                (img, gt) for img, gt in self._samples
                if img.startswith(split_dir)
            ]

        return new_ds
