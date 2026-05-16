"""Model: split integrity (no train/test overlap), discrimination,
explainability shape, and per-segment aggregation."""

import numpy as np

from vista.config import TEST_FRACTION, VAL_FRACTION
from vista.data_gen import generate_fleet
from vista.model import (
    explain_pole,
    fit,
    predict_proba,
    segment_risk,
    split_indices,
)


def _fitted():
    fd = generate_fleet()
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    fr = fit(fd.X, fd.y, fd.feature_names, tr, va, te)
    return fd, fr


def test_split_is_disjoint_and_complete():
    fd = generate_fleet()
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == len(fd.y)


def test_split_is_stratified():
    fd = generate_fleet()
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    base = fd.y.mean()
    for idx in (tr, va, te):
        assert abs(fd.y[idx].mean() - base) < 0.05


def test_model_discriminates_on_heldout():
    from sklearn.metrics import roc_auc_score
    fd, fr = _fitted()
    p = predict_proba(fr, fd.X[fr.idx_test])
    auc = roc_auc_score(fd.y[fr.idx_test], p)
    assert auc > 0.78, f"held-out AUC unexpectedly low: {auc:.3f}"


def test_predict_proba_in_unit_interval():
    fd, fr = _fitted()
    p = predict_proba(fr, fd.X)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_explain_returns_ranked_drivers():
    fd, fr = _fitted()
    drivers = explain_pole(fr, fd.X[0], top_k=4)
    assert len(drivers) == 4
    mags = [abs(c) for _, c in drivers]
    assert mags == sorted(mags, reverse=True), "drivers must be rank-ordered"


def test_segment_risk_covers_all_segments():
    fd, fr = _fitted()
    p = predict_proba(fr, fd.X)
    seg = segment_risk(fd.segment_id, p)
    assert set(seg.keys()) == set(fd.segment_id)
    assert all(0.0 <= v <= 1.0 for v in seg.values())


def test_explanations_are_deterministic():
    fd, fr = _fitted()
    assert explain_pole(fr, fd.X[5], 3) == explain_pole(fr, fd.X[5], 3)
