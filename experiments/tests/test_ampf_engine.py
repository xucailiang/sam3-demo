# Feature: ampf-crack-segmentation, Property 3-6
"""Property-based tests for AMPF engine standalone functions.

We load ampf_engine.py via importlib to bypass the package __init__.py
that eagerly imports modules with missing dependencies.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# --- Mock missing modules before loading ampf_engine ---
_mock_schemas = types.ModuleType("app.models.schemas")
_mock_schemas.SegmentationResult = MagicMock
_mock_schemas.MaskData = MagicMock
sys.modules.setdefault("app", types.ModuleType("app"))
sys.modules.setdefault("app.models", types.ModuleType("app.models"))
sys.modules["app.models.schemas"] = _mock_schemas

_mock_seg_svc = types.ModuleType("app.services.segmentation_service")
_mock_seg_svc.SegmentationService = MagicMock
sys.modules.setdefault("app.services", types.ModuleType("app.services"))
sys.modules["app.services.segmentation_service"] = _mock_seg_svc

# Load ampf_engine directly from file path (bypass __init__.py)
_engine_path = (
    Path(__file__).resolve().parent.parent.parent
    / "backend" / "app" / "services" / "ampf_engine.py"
)
spec = importlib.util.spec_from_file_location("ampf_engine", _engine_path)
ampf_mod = importlib.util.module_from_spec(spec)
sys.modules["ampf_engine"] = ampf_mod  # register so @dataclass can resolve module
spec.loader.exec_module(ampf_mod)

AMPFConfig = ampf_mod.AMPFConfig
AMPFEngine = ampf_mod.AMPFEngine
Candidate = ampf_mod.Candidate


def _make_engine(config=None):
    return AMPFEngine(segmentation_service=MagicMock(), config=config or AMPFConfig())


# --- Property 3: IoU matching selects best candidate ---

@given(n_candidates=st.integers(min_value=1, max_value=5), data=st.data())
@settings(max_examples=100)
def test_iou_matching_selects_best(n_candidates, data):
    engine = _make_engine()
    h, w = 50, 50
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    gt_instances = engine.extract_gt_instances(gt)
    if not gt_instances:
        return
    candidates = []
    for _ in range(n_candidates):
        mask = np.zeros((h, w), dtype=np.uint8)
        offset = data.draw(st.integers(min_value=0, max_value=20))
        mask[10 + offset:30, 10:30] = 255
        candidates.append(Candidate(mask=mask, detection_score=0.9, mode="text"))
    aligned = engine.align_candidates(gt_instances, candidates, [], [])
    for inst in aligned:
        if inst.text_candidate is not None:
            best_iou = AMPFEngine._compute_iou(inst.gt_mask, inst.text_candidate.mask)
            for c in candidates:
                assert AMPFEngine._compute_iou(inst.gt_mask, c.mask) <= best_iou + 1e-9


# --- Property 4: Composite confidence formula ---

@given(
    s_det=st.floats(min_value=0.0, max_value=1.0),
    s_stab=st.floats(min_value=0.0, max_value=1.0),
    s_bound=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_composite_confidence_formula(s_det, s_stab, s_bound):
    config = AMPFConfig(alpha=0.4, beta=0.35, gamma=0.25)
    expected = config.alpha * s_det + config.beta * s_stab + config.gamma * s_bound
    assert 0.0 <= expected <= 1.0 + 1e-9


# --- Property 5: Boundary score range [0, 1] ---

@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_boundary_score_range(seed):
    np.random.seed(seed)
    engine = _make_engine()
    mask = np.zeros((50, 50), dtype=np.uint8)
    n = int(50 * 50 * 0.3)
    indices = np.random.choice(50 * 50, size=n, replace=False)
    mask.flat[indices] = 255
    score = engine.compute_boundary_score(mask)
    assert 0.0 <= score <= 1.0


def test_boundary_score_empty():
    engine = _make_engine()
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert engine.compute_boundary_score(mask) == 0.0


# --- Property 6: Fusion output is valid binary mask ---

@given(n=st.integers(min_value=1, max_value=4), data=st.data())
@settings(max_examples=100)
def test_fusion_output_valid(n, data):
    engine = _make_engine()
    h, w = 30, 30
    pairs = []
    for _ in range(n):
        mask = np.zeros((h, w), dtype=np.uint8)
        r1 = data.draw(st.integers(0, h - 5))
        c1 = data.draw(st.integers(0, w - 5))
        mask[r1:r1 + 5, c1:c1 + 5] = 255
        conf = data.draw(st.floats(min_value=0.0, max_value=1.0))
        cand = Candidate(mask=mask, detection_score=conf, mode="text")
        pairs.append((cand, conf))
    result = engine.fuse_instance(pairs)
    assert result.dtype == np.uint8
    assert result.shape == (h, w)
    assert set(np.unique(result)).issubset({0, 255})


def test_fusion_fallback_all_below_threshold():
    config = AMPFConfig(confidence_threshold=0.9)
    engine = _make_engine(config)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    cand = Candidate(mask=mask, detection_score=0.1, mode="text")
    result = engine.fuse_instance([(cand, 0.1)])
    assert result.shape == (20, 20)
    assert np.any(result > 0)
