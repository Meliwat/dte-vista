"""Validation integrity - the Effectiveness axis.

These tests assert the claims VISTA makes are real:
  * a genuine held-out split is scored,
  * calibration is measured (reliability table + Brier + gap),
  * spatial AND temporal backtests produce DISTRIBUTIONS (not single splits),
  * VISTA beats the REAL incumbent age-cycle baseline (positive lift),
  * the imagery modality contributes POSITIVE marginal value (ablation) -
    this is VISTA's distinct axis and must be demonstrably real.
"""

import numpy as np
import pytest

from vista.config import N_SPATIAL_FOLDS, TEST_FRACTION, VAL_FRACTION
from vista.data_gen import generate_fleet
from vista.model import fit, split_indices
from vista.validation import run_validation


@pytest.fixture(scope="module")
def report():
    fd = generate_fleet()
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    fr = fit(fd.X, fd.y, fd.feature_names, tr, va, te)
    return fd, fr, run_validation(fd, fr)


def test_heldout_metrics_sane(report):
    _, _, vr = report
    h = vr.heldout
    assert 0.78 < h["roc_auc"] <= 1.0
    assert 0.0 < h["pr_auc"] <= 1.0
    assert 0.0 < h["brier"] < 0.25
    assert h["n_test"] == 280


def test_calibration_table_present_and_reasonable(report):
    _, _, vr = report
    assert len(vr.reliability) >= 4, "need a populated reliability table"
    assert 0.0 <= vr.calib_gap < 0.15, "weighted calibration gap must be small"
    for mp, ef, c in vr.reliability:
        assert 0.0 <= mp <= 1.0 and 0.0 <= ef <= 1.0 and c > 0


def test_spatial_backtest_is_a_distribution(report):
    _, _, vr = report
    assert len(vr.spatial_auc) >= N_SPATIAL_FOLDS - 1
    assert len(set(vr.spatial_auc)) > 1, "must be a DISTRIBUTION, not one value"
    mean, sd = vr.spatial_summary
    assert mean > 0.72, f"spatial generalization weak: {mean}"
    assert sd > 0.0


def test_temporal_backtest_is_a_distribution(report):
    _, _, vr = report
    assert len(vr.temporal_capture) >= 6
    assert len(set(vr.temporal_capture)) > 1, "must vary across storms"
    mean, sd = vr.temporal_summary
    assert 0.0 < mean <= 1.0
    assert sd > 0.0


def test_beats_real_incumbent_age_cycle(report):
    """The headline claim: VISTA's risk-ranked worklist catches strictly
    more failures than the fixed age-based inspection cycle at equal budget."""
    _, _, vr = report
    inc = vr.incumbent
    assert inc["capture_vista"] > inc["capture_age_cycle"], \
        "VISTA must beat the incumbent age-cycle"
    assert inc["capture_vista"] > inc["capture_fault_reactive"], \
        "VISTA must beat reactive run-to-failure"
    assert inc["capture_vista"] > inc["capture_random"], \
        "VISTA must beat random selection"
    assert inc["lift_vs_age_cycle_x"] > 1.3, "lift over incumbent too small"


def test_imagery_ablation_shows_positive_contribution(report):
    """VISTA's DISTINCT axis: removing the image-derived features must
    measurably degrade held-out AUC and capture. This proves the imagery
    modality (not just a tabular classifier) is doing the work."""
    _, _, vr = report
    abl = vr.ablation
    assert abl["auc_with_imagery"] > abl["auc_without_imagery"], \
        "imagery features must improve discrimination"
    assert abl["auc_gain_from_imagery"] > 0.03, \
        "imagery contribution must be material, not marginal"
    assert abl["capture20_gain_from_imagery"] > 0.0


def test_validation_is_deterministic():
    fd = generate_fleet()
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    fr = fit(fd.X, fd.y, fd.feature_names, tr, va, te)
    a = run_validation(fd, fr)
    b = run_validation(fd, fr)
    assert a.heldout == b.heldout
    assert a.spatial_auc == b.spatial_auc
    assert a.temporal_capture == b.temporal_capture
    assert a.ablation == b.ablation
