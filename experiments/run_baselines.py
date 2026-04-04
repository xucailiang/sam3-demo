"""Supervised baseline training and evaluation for crack segmentation.

Trains U-Net, DeepLabV3+, and YOLOv8n-seg on CrackForest / DeepCrack datasets
and evaluates them using the same metrics as the AMPF experiments.

Dependencies:
  - segmentation_models_pytorch (smp) — for U-Net and DeepLabV3+
  - ultralytics — for YOLOv8-seg (already in requirements.txt)
  - torch, torchvision — PyTorch ecosystem
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_loader import CrackDataset
from evaluation import evaluate_single

logger = logging.getLogger(__name__)

# Output directory — same pattern as run_experiments.py
_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Optional smp import — handle gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import segmentation_models_pytorch as smp

    _SMP_AVAILABLE = True
except ImportError:
    smp = None  # type: ignore[assignment]
    _SMP_AVAILABLE = False
    logger.warning(
        "segmentation_models_pytorch not installed. "
        "U-Net and DeepLabV3+ baselines will be unavailable."
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper for smp models
# ---------------------------------------------------------------------------

_IMG_SIZE = 512  # Resize target for U-Net / DeepLabV3+


class _CrackTorchDataset(Dataset):
    """Wraps a CrackDataset for PyTorch DataLoader.

    Resizes images to 512×512, normalises to [0, 1], and converts masks to
    float tensors suitable for BCE loss.
    """

    def __init__(self, crack_dataset: CrackDataset) -> None:
        self._ds = crack_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, gt_mask = self._ds[idx]

        # Resize to 512×512
        image = cv2.resize(image, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Image: (H, W, 3) uint8 → (3, H, W) float32 [0, 1]
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Mask: (H, W) uint8 {0, 255} → (1, H, W) float32 {0, 1}
        mask_t = torch.from_numpy(gt_mask).unsqueeze(0).float() / 255.0

        return img_t, mask_t


# ---------------------------------------------------------------------------
# BCE + Dice loss
# ---------------------------------------------------------------------------

class _BCEDiceLoss(nn.Module):
    """Combined BCE-with-logits and Dice loss."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + 1e-7) / (probs.sum() + targets.sum() + 1e-7)
        dice_loss = 1.0 - dice

        return bce_loss + dice_loss


# ---------------------------------------------------------------------------
# YOLO format conversion helpers
# ---------------------------------------------------------------------------

def _mask_to_yolo_polygons(mask: np.ndarray) -> List[List[float]]:
    """Convert a binary mask to YOLO polygon format.

    Finds contours, normalises coordinates to [0, 1], and returns a list of
    polygon coordinate lists (one per contour).
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: List[List[float]] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        coords: List[float] = []
        for pt in cnt.squeeze():
            if pt.ndim == 0:
                continue
            coords.append(float(pt[0]) / w)  # normalised x
            coords.append(float(pt[1]) / h)  # normalised y
        if len(coords) >= 6:  # at least 3 points
            polygons.append(coords)
    return polygons


def _prepare_yolo_dataset(
    train_ds: CrackDataset, val_ds: CrackDataset, tmp_dir: str
) -> str:
    """Create a temporary YOLO-format dataset directory.

    Structure:
        tmp_dir/
            images/train/  images/val/
            labels/train/  labels/val/
            data.yaml

    Returns path to data.yaml.
    """
    base = Path(tmp_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (base / sub).mkdir(parents=True, exist_ok=True)

    # Write train split
    for idx in range(len(train_ds)):
        image, gt_mask = train_ds[idx]
        fname = f"img_{idx:05d}"
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(base / "images" / "train" / f"{fname}.jpg"), img_bgr)

        polygons = _mask_to_yolo_polygons(gt_mask)
        with open(base / "labels" / "train" / f"{fname}.txt", "w") as f:
            for poly in polygons:
                f.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")

    # Write val split
    for idx in range(len(val_ds)):
        image, gt_mask = val_ds[idx]
        fname = f"img_{idx:05d}"
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(base / "images" / "val" / f"{fname}.jpg"), img_bgr)

        polygons = _mask_to_yolo_polygons(gt_mask)
        with open(base / "labels" / "val" / f"{fname}.txt", "w") as f:
            for poly in polygons:
                f.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")

    # data.yaml
    yaml_path = base / "data.yaml"
    yaml_path.write_text(
        f"path: {base}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['crack']\n"
    )
    return str(yaml_path)


# ---------------------------------------------------------------------------
# BaselineTrainer
# ---------------------------------------------------------------------------

class BaselineTrainer:
    """Trains and evaluates supervised segmentation baselines.

    Supports U-Net (smp), DeepLabV3+ (smp), and YOLOv8n-seg (ultralytics).
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info("BaselineTrainer using device: %s", self.device)

    # ------------------------------------------------------------------
    # Internal: generic smp training loop
    # ------------------------------------------------------------------

    def _train_smp_model(
        self,
        model: nn.Module,
        dataset: CrackDataset,
        model_name: str,
        epochs: int = 50,
        lr: float = 1e-4,
        batch_size: int = 4,
    ) -> nn.Module:
        """Train an smp model with BCE+Dice loss and Adam optimiser.

        Uses dataset.get_split() for train/val. Saves best model by val loss.
        """
        train_ds = dataset.get_split("train")
        val_ds = dataset.get_split("test")

        train_loader = DataLoader(
            _CrackTorchDataset(train_ds), batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            _CrackTorchDataset(val_ds), batch_size=batch_size, shuffle=False, num_workers=0
        )

        model = model.to(self.device)
        criterion = _BCEDiceLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None

        for epoch in range(1, epochs + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= max(len(train_loader.dataset), 1)  # type: ignore[arg-type]

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    logits = model(images)
                    loss = criterion(logits, masks)
                    val_loss += loss.item() * images.size(0)
            val_loss /= max(len(val_loader.dataset), 1)  # type: ignore[arg-type]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "[%s] Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  best=%.4f",
                    model_name, epoch, epochs, train_loss, val_loss, best_val_loss,
                )

            scheduler.step()

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.to(self.device)
        model.eval()
        logger.info("[%s] Training complete. Best val loss: %.4f", model_name, best_val_loss)
        return model

    # ------------------------------------------------------------------
    # Public training methods
    # ------------------------------------------------------------------

    def train_unet(
        self, dataset: CrackDataset, epochs: int = 50, lr: float = 1e-4
    ) -> nn.Module:
        """Train U-Net with ResNet-34 encoder, ImageNet pretrained, BCE+Dice loss, Adam."""
        if not _SMP_AVAILABLE:
            raise RuntimeError(
                "segmentation_models_pytorch is required for U-Net training. "
                "Install with: pip install segmentation-models-pytorch"
            )
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return self._train_smp_model(model, dataset, "U-Net", epochs=epochs, lr=lr)

    def train_deeplabv3plus(
        self, dataset: CrackDataset, epochs: int = 50, lr: float = 1e-4
    ) -> nn.Module:
        """Train DeepLabV3+ with ResNet-34 encoder, same config as U-Net."""
        if not _SMP_AVAILABLE:
            raise RuntimeError(
                "segmentation_models_pytorch is required for DeepLabV3+ training. "
                "Install with: pip install segmentation-models-pytorch"
            )
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return self._train_smp_model(model, dataset, "DeepLabV3+", epochs=epochs, lr=lr)

    def train_yolov8seg(
        self, dataset: CrackDataset, epochs: int = 100
    ) -> Any:
        """Train YOLOv8n-seg after converting masks to YOLO polygon format.

        Creates a temporary YOLO dataset directory, trains, and cleans up.
        """
        from ultralytics import YOLO

        train_ds = dataset.get_split("train")
        val_ds = dataset.get_split("test")

        tmp_dir = tempfile.mkdtemp(prefix="yolo_crack_")
        try:
            yaml_path = _prepare_yolo_dataset(train_ds, val_ds, tmp_dir)
            logger.info("[YOLOv8-seg] Starting training for %d epochs", epochs)

            model = YOLO("yolov8n-seg.pt")
            model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=512,
                batch=8,
                project=tmp_dir,
                name="crack_seg",
                verbose=False,
            )
            logger.info("[YOLOv8-seg] Training complete.")
            return model
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(tmp_dir)
                logger.info("[YOLOv8-seg] Cleaned up temp directory: %s", tmp_dir)
            except OSError as exc:
                logger.warning("Failed to clean up temp dir %s: %s", tmp_dir, exc)


    # ------------------------------------------------------------------
    # Evaluation (Task 8.2)
    # ------------------------------------------------------------------

    def evaluate_model(
        self, model: Any, dataset: CrackDataset, model_name: str
    ) -> pd.DataFrame:
        """Evaluate a trained model on the test split.

        For smp models (U-Net, DeepLabV3+): resize to 512×512, run inference,
        threshold at 0.5, resize prediction back to original size, convert to
        binary mask (0/255).

        For YOLOv8: use model.predict(), extract masks from results.

        Returns:
            DataFrame with columns: image_id, method, dataset, iou, dice,
            precision, recall, f1.
        """
        test_ds = dataset.get_split("test")
        rows: List[Dict[str, Any]] = []

        is_yolo = hasattr(model, "predict") and hasattr(model, "task")

        for idx in range(len(test_ds)):
            image_id = f"{dataset.dataset_name}_{idx:04d}"
            try:
                image, gt_mask = test_ds[idx]
                if is_yolo:
                    pred_mask = self._predict_yolo(model, image, gt_mask.shape)
                else:
                    pred_mask = self._predict_smp(model, image, gt_mask.shape)

                metrics = evaluate_single(pred_mask, gt_mask)
                rows.append(
                    {
                        "image_id": image_id,
                        "method": model_name,
                        "dataset": dataset.dataset_name,
                        **metrics,
                    }
                )
            except Exception:
                logger.exception(
                    "Evaluation of '%s' failed on image %s", model_name, image_id
                )

        df = pd.DataFrame(rows)
        logger.info(
            "[%s] Evaluated %d images on %s",
            model_name, len(df), dataset.dataset_name,
        )
        return df

    # ------------------------------------------------------------------
    # Internal prediction helpers
    # ------------------------------------------------------------------

    def _predict_smp(
        self, model: nn.Module, image: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Run smp model inference on a single image.

        Returns (H, W) uint8 binary mask with values 0/255 at original resolution.
        """
        h_orig, w_orig = original_shape[:2]

        # Preprocess: same as training
        resized = cv2.resize(image, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(img_t)
            probs = torch.sigmoid(logits)

        # Threshold at 0.5
        pred = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        # Resize back to original resolution
        if pred.shape != (h_orig, w_orig):
            pred = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        return pred

    def _predict_yolo(
        self, model: Any, image: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Run YOLOv8-seg inference on a single image.

        Returns (H, W) uint8 binary mask with values 0/255.
        """
        h_orig, w_orig = original_shape[:2]

        # YOLOv8 expects BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = model.predict(img_bgr, verbose=False, imgsz=512)

        combined = np.zeros((h_orig, w_orig), dtype=np.uint8)
        if results and results[0].masks is not None:
            for mask_data in results[0].masks.data:
                mask_np = mask_data.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
                )
                combined = np.maximum(
                    combined, (mask_resized > 0.5).astype(np.uint8) * 255
                )

        return combined


# ---------------------------------------------------------------------------
# Convenience: run all baselines
# ---------------------------------------------------------------------------

def run_all_baselines(
    datasets: Dict[str, CrackDataset], device: Optional[str] = None
) -> pd.DataFrame:
    """Train and evaluate all 3 baseline models on all provided datasets.

    Args:
        datasets: Mapping of dataset name → CrackDataset instance.
        device: PyTorch device string (auto-detected if None).

    Returns:
        Combined DataFrame with per-image metrics for every model × dataset.
    """
    trainer = BaselineTrainer(device=device)
    all_frames: List[pd.DataFrame] = []

    for ds_name, dataset in datasets.items():
        logger.info("=" * 60)
        logger.info("Running baselines on dataset: %s", ds_name)
        logger.info("=" * 60)

        # --- U-Net ---
        try:
            unet_model = trainer.train_unet(dataset)
            unet_df = trainer.evaluate_model(unet_model, dataset, "unet")
            all_frames.append(unet_df)
        except Exception:
            logger.exception("U-Net baseline failed on %s", ds_name)

        # --- DeepLabV3+ ---
        try:
            dlv3_model = trainer.train_deeplabv3plus(dataset)
            dlv3_df = trainer.evaluate_model(dlv3_model, dataset, "deeplabv3plus")
            all_frames.append(dlv3_df)
        except Exception:
            logger.exception("DeepLabV3+ baseline failed on %s", ds_name)

        # --- YOLOv8-seg ---
        try:
            yolo_model = trainer.train_yolov8seg(dataset)
            yolo_df = trainer.evaluate_model(yolo_model, dataset, "yolov8seg")
            all_frames.append(yolo_df)
        except Exception:
            logger.exception("YOLOv8-seg baseline failed on %s", ds_name)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=["image_id", "method", "dataset", "iou", "dice", "precision", "recall", "f1"]
        )

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baselines.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Baseline results saved to %s (%d rows)", out_path, len(combined))

    return combined


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train and evaluate supervised baselines")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=str(_SCRIPT_DIR / "datasets"),
        help="Root directory containing CrackForest-dataset/ and DeepCrack/",
    )
    parser.add_argument("--device", type=str, default=None, help="PyTorch device")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["crackforest", "deepcrack"],
        help="Run on a single dataset (default: both)",
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    datasets: Dict[str, CrackDataset] = {}

    if args.dataset is None or args.dataset == "crackforest":
        cf_root = datasets_dir / "CrackForest-dataset"
        if cf_root.exists():
            datasets["crackforest"] = CrackDataset("crackforest", str(cf_root))
        else:
            logger.warning("CrackForest dataset not found at %s", cf_root)

    if args.dataset is None or args.dataset == "deepcrack":
        dc_root = datasets_dir / "DeepCrack"
        if dc_root.exists():
            datasets["deepcrack"] = CrackDataset("deepcrack", str(dc_root))
        else:
            logger.warning("DeepCrack dataset not found at %s", dc_root)

    if not datasets:
        logger.error("No datasets found. Exiting.")
        raise SystemExit(1)

    results = run_all_baselines(datasets, device=args.device)

    # Print summary
    if not results.empty:
        metrics = ["iou", "dice", "precision", "recall", "f1"]
        summary = results.groupby(["method", "dataset"])[metrics].mean()
        print("\n" + "=" * 72)
        print("  Baseline Summary (mean metrics)")
        print("=" * 72)
        print(summary.to_string(float_format="{:.4f}".format))
        print("=" * 72 + "\n")
