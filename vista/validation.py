"""Honest validation - the Effectiveness axis.

Provides, all deterministic:
  1. Held-out test metrics (ROC-AUC, PR-AUC, Brier) on the fold seen once.
  2. A calibration reliability table (10 bins) + Brier + calibration gap.
  3. A SPATIAL backtest *distribution*: leave-county-group-out (6 folds) -
     "deploy where we never inspected". Reports per-fold AUC -> mean +/- sd.
  4. A TEMPORAL backtest *distribution*: 8 independent storm replays with
     storm-dependent stress; per-storm top-k capture -> mean +/- sd.
  5. Lift vs a REAL incumbent baseline: the fixed time-based inspection cycle
     utilities run today (oldest-pole-first), measured as failures caught in
     the same inspection budget, plus an ablation showing the imagery stack's
     marginal contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import (
    CALIBRATION_BINS,
    INSPECTION_BUDGET_FRACTION,
    MASTER_SEED,
    N_SPATIAL_FOLDS,
    N_TEMPORAL_STORMS,
)
from .data_gen import FleetData
from .model import FitResult, predict_proba


@dataclass
class ValidationReport:
    heldout: Dict[str, float]
    reliability: List[Tuple[float, float, int]]  # (mean_pred, emp_freq, count)
    brier: float
    calib_gap: float
    spatial_auc: List[float]
    spatial_summary: Tuple[float, float]
    temporal_capture: List[float]
    temporal_summary: Tuple[float, float]
    incumbent: Dict[str, float]
    ablation: Dict[str, float] = field(default_factory=dict)


def _capture_at_budget(y: np.ndarray, score: np.ndarray,
                       budget_frac: float) -> float:
    """Fraction of true failures caught when inspecting the top `budget_frac`
    of poles ranked by `score` (precision/recall@k style)."""
    n = len(y)
    k = max(1, int(round(n * budget_frac)))
    order = np.argsort(-score, kind="stable")
    top = order[:k]
    return float(y[top].sum()) / float(max(1, y.sum()))


def heldout_metrics(fr: FitResult, fd: FleetData) -> Dict[str, float]:
    Xte, yte = fd.X[fr.idx_test], fd.y[fr.idx_test]
    p = predict_proba(fr, Xte)
    return {
        "n_test": int(len(yte)),
        "pos_rate": round(float(yte.mean()), 4),
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "brier": round(float(brier_score_loss(yte, p)), 4),
        "capture@20": round(_capture_at_budget(yte, p, INSPECTION_BUDGET_FRACTION), 4),
    }


def reliability_table(fr: FitResult, fd: FleetData,
                      bins: int = CALIBRATION_BINS
                      ) -> Tuple[List[Tuple[float, float, int]], float, float]:
    Xte, yte = fd.X[fr.idx_test], fd.y[fr.idx_test]
    p = predict_proba(fr, Xte)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Tuple[float, float, int]] = []
    gaps, weights = [], []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        c = int(m.sum())
        if c == 0:
            continue
        mp = float(p[m].mean())
        ef = float(yte[m].mean())
        rows.append((round(mp, 4), round(ef, 4), c))
        gaps.append(abs(mp - ef))
        weights.append(c)
    brier = float(brier_score_loss(yte, p))
    calib_gap = float(np.average(gaps, weights=weights)) if gaps else 0.0
    return rows, round(brier, 4), round(calib_gap, 4)


def spatial_backtest(fd: FleetData, seed: int = MASTER_SEED) -> List[float]:
    """Leave-county-group-out: hold whole county groups out, predict cold.

    This is the honest "deploy where we have never inspected" test that a
    random split silently leaks. Returns a DISTRIBUTION of fold AUCs.
    """
    counties = np.unique(fd.county_idx)
    rng = np.random.default_rng(seed)
    perm = counties.copy()
    rng.shuffle(perm)
    folds = np.array_split(perm, N_SPATIAL_FOLDS)
    aucs: List[float] = []
    for held in folds:
        te = np.isin(fd.county_idx, held)
        tr = ~te
        if fd.y[tr].sum() < 5 or fd.y[te].sum() < 2:
            continue
        sc = StandardScaler().fit(fd.X[tr])
        gb = GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=seed)
        gb.fit(sc.transform(fd.X[tr]), fd.y[tr])
        p = gb.predict_proba(sc.transform(fd.X[te]))[:, 1]
        aucs.append(round(float(roc_auc_score(fd.y[te], p)), 4))
    return aucs


def temporal_backtest(fr: FitResult, fd: FleetData,
                      seed: int = MASTER_SEED) -> List[float]:
    """Replay N independent storms. Each storm perturbs which latent-risky
    poles actually fail (wind/flood-driven), then we measure top-budget
    capture using the FROZEN model. Returns a DISTRIBUTION across storms.
    """
    rng = np.random.default_rng(seed + 101)
    Xte = fd.X[fr.idx_test]
    base_p = fd.p_latent[fr.idx_test]
    fi = {n: j for j, n in enumerate(fd.feature_names)}
    wind = Xte[:, fi["wind_norm_mph"]]
    flood = Xte[:, fi["flood_zone_risk"]]
    rowc = Xte[:, fi["img_row_canopy"]]
    score = predict_proba(fr, Xte)

    def zsc(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    caps: List[float] = []
    for _ in range(N_TEMPORAL_STORMS):
        sev = rng.uniform(0.5, 1.6)
        stress = 1.0 / (1.0 + np.exp(-(
            np.log(base_p / (1 - base_p) + 1e-9)
            + sev * (0.6 * zsc(wind) + 0.5 * zsc(flood) + 0.5 * zsc(rowc)))))
        y_storm = (rng.random(len(stress)) < stress).astype(int)
        if y_storm.sum() < 2:
            continue
        caps.append(round(_capture_at_budget(
            y_storm, score, INSPECTION_BUDGET_FRACTION), 4))
    return caps


def incumbent_comparison(fr: FitResult, fd: FleetData) -> Dict[str, float]:
    """Lift over the REAL incumbent policy utilities run today.

    Incumbent A: fixed time-based cycle == oldest-pole-first ranking.
    Incumbent B: prior-fault reactive (run-to-failure proxy).
    All evaluated at the SAME inspection budget on the held-out fold.
    """
    Xte, yte = fd.X[fr.idx_test], fd.y[fr.idx_test]
    fi = {n: j for j, n in enumerate(fd.feature_names)}
    age = Xte[:, fi["age_years"]]
    faults = Xte[:, fi["prior_faults"]]
    model_score = predict_proba(fr, Xte)
    b = INSPECTION_BUDGET_FRACTION

    cap_model = _capture_at_budget(yte, model_score, b)
    cap_age = _capture_at_budget(yte, age, b)
    cap_fault = _capture_at_budget(yte, faults, b)
    cap_random = float(b)  # expectation of random selection
    return {
        "budget_frac": b,
        "capture_vista": round(cap_model, 4),
        "capture_age_cycle": round(cap_age, 4),
        "capture_fault_reactive": round(cap_fault, 4),
        "capture_random": round(cap_random, 4),
        "lift_vs_age_cycle_pts": round(cap_model - cap_age, 4),
        "lift_vs_age_cycle_x": round(cap_model / max(1e-6, cap_age), 2),
        "lift_vs_reactive_x": round(cap_model / max(1e-6, cap_fault), 2),
    }


def imagery_ablation(fd: FleetData, fr: FitResult,
                     seed: int = MASTER_SEED) -> Dict[str, float]:
    """Quantify the imagery modality's marginal value: refit WITHOUT the
    image-derived features and compare held-out AUC + capture. This is the
    measured payoff of VISTA's distinct axis."""
    img_cols = [j for j, n in enumerate(fd.feature_names) if n.startswith("img_")]
    keep = [j for j in range(len(fd.feature_names)) if j not in img_cols]
    Xtr, ytr = fd.X[fr.idx_train][:, keep], fd.y[fr.idx_train]
    Xte, yte = fd.X[fr.idx_test][:, keep], fd.y[fr.idx_test]
    sc = StandardScaler().fit(Xtr)
    gb = GradientBoostingClassifier(
        n_estimators=220, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=seed)
    gb.fit(sc.transform(Xtr), ytr)
    p = gb.predict_proba(sc.transform(Xte))[:, 1]
    auc_no_img = float(roc_auc_score(yte, p))
    cap_no_img = _capture_at_budget(yte, p, INSPECTION_BUDGET_FRACTION)

    full = predict_proba(fr, fd.X[fr.idx_test])
    auc_full = float(roc_auc_score(yte, full))
    cap_full = _capture_at_budget(yte, full, INSPECTION_BUDGET_FRACTION)
    return {
        "auc_without_imagery": round(auc_no_img, 4),
        "auc_with_imagery": round(auc_full, 4),
        "auc_gain_from_imagery": round(auc_full - auc_no_img, 4),
        "capture20_without_imagery": round(cap_no_img, 4),
        "capture20_with_imagery": round(cap_full, 4),
        "capture20_gain_from_imagery": round(cap_full - cap_no_img, 4),
    }


def run_validation(fd: FleetData, fr: FitResult) -> ValidationReport:
    heldout = heldout_metrics(fr, fd)
    rel, brier, gap = reliability_table(fr, fd)
    sp = spatial_backtest(fd)
    sp_sum = (round(float(np.mean(sp)), 4), round(float(np.std(sp)), 4)) if sp else (0.0, 0.0)
    tp = temporal_backtest(fr, fd)
    tp_sum = (round(float(np.mean(tp)), 4), round(float(np.std(tp)), 4)) if tp else (0.0, 0.0)
    inc = incumbent_comparison(fr, fd)
    abl = imagery_ablation(fd, fr)
    return ValidationReport(
        heldout=heldout, reliability=rel, brier=brier, calib_gap=gap,
        spatial_auc=sp, spatial_summary=sp_sum,
        temporal_capture=tp, temporal_summary=tp_sum,
        incumbent=inc, ablation=abl,
    )
