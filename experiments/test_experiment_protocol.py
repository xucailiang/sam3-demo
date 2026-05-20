"""Tests for experiment protocol choices that affect paper validity."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(SCRIPT_DIR))

from app.services.ampf_engine import AMPFConfig, AMPFEngine
from data_loader import CrackDataset
from evaluation import evaluate_single
from prompt_generator import PromptGenerator


def _sample_paths(dataset: CrackDataset) -> set[str]:
    return {img for img, _ in dataset._samples}


@pytest.mark.parametrize(
    ("name", "root"),
    [
        ("crackforest", "datasets/CrackForest-dataset"),
        ("deepcrack", "datasets/DeepCrack"),
    ],
)
def test_train_val_test_splits_are_disjoint(name: str, root: str) -> None:
    dataset = CrackDataset(name, str(SCRIPT_DIR / root))

    train = dataset.get_split("train")
    val = dataset.get_split("val")
    test = dataset.get_split("test")

    train_paths = _sample_paths(train)
    val_paths = _sample_paths(val)
    test_paths = _sample_paths(test)

    assert train_paths
    assert val_paths
    assert test_paths
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    assert len(train_paths | val_paths | test_paths) == len(dataset)


def test_prompt_generator_is_oracle_gt_prompt_protocol() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:8, 5:10] = 255
    mask[12:15, 20:25] = 255

    prompts = PromptGenerator.generate_all_prompts(mask)

    assert prompts["text"] == "crack"
    assert prompts["box"] == [5.0, 4.0, 25.0, 15.0]
    assert len(prompts["points"]) == 2
    assert prompts["labels"] == [1, 1]


def test_evaluation_metrics_toy_case() -> None:
    pred = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    gt = np.array([[255, 0], [255, 0]], dtype=np.uint8)

    metrics = evaluate_single(pred, gt)

    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_ampf_perturbations_are_reproducible_with_seed() -> None:
    cfg = AMPFConfig(random_seed=123)
    engine_a = AMPFEngine(segmentation_service=None, config=cfg)  # type: ignore[arg-type]
    engine_b = AMPFEngine(segmentation_service=None, config=cfg)  # type: ignore[arg-type]

    bbox = [10.0, 20.0, 30.0, 50.0]
    point = (12.0, 24.0)

    assert engine_a._perturb_bbox(bbox, 0.05) == engine_b._perturb_bbox(bbox, 0.05)
    assert engine_a._perturb_point(point, 3) == engine_b._perturb_point(point, 3)


def test_boundary_score_prefers_image_edges() -> None:
    engine = AMPFEngine(segmentation_service=None)  # type: ignore[arg-type]

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, 20:] = 255

    edge_mask = np.zeros((40, 40), dtype=np.uint8)
    edge_mask[:, 18:22] = 255

    off_edge_mask = np.zeros((40, 40), dtype=np.uint8)
    off_edge_mask[:, 2:6] = 255

    edge_score = engine.compute_boundary_score(edge_mask, image)
    off_edge_score = engine.compute_boundary_score(off_edge_mask, image)

    assert edge_score > off_edge_score
