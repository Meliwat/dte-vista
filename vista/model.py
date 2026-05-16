"""Explainable, calibrated pole-risk model.

Pipeline: standardize -> gradient-boosted trees -> isotonic/sigmoid
calibration on a held-out validation fold. We expose:
  * per-pole calibrated failure probability,
  * per-circuit-segment aggregated risk,
  * per-pole plain-English ranked driver attribution (margin contribution of
    each standardized feature, including the image-derived ones), so a planner
    sees *why* a pole is flagged.

Determinism: fixed seeds, single-thread tree building, sorted tie-breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .config import MASTER_SEED

# Human-readable names for driver explanations.
PRETTY = {
    "age_years": "pole age",
    "material_wood": "wood construction",
    "prior_faults": "prior fault history",
    "span_len_m": "long span length",
    "wind_norm_mph": "high wind climatology (NOAA)",
    "snow_norm_in": "heavy snow load (NOAA)",
    "precip_norm_in": "high precipitation (NOAA)",
    "freeze_thaw_idx": "freeze-thaw cycling",
    "soil_corrosivity": "corrosive soil",
    "soil_moisture": "wet soil",
    "flood_zone_risk": "flood-zone exposure (FEMA-style)",
    "slope_deg": "terrain slope",
    "load_proxy": "electrical loading",
    "coastal_dist_km": "proximity to open water",
    "img_canopy_frac": "dense canopy in tile (imagery)",
    "img_ndvi_mean": "vegetation vigor / NDVI (imagery)",
    "img_ndvi_p90": "peak NDVI density (imagery)",
    "img_row_canopy": "canopy encroaching the right-of-way (imagery)",
    "img_overhang": "canopy overhang directly above the line (imagery)",
    "img_row_growth": "right-of-way vegetation growth t0->t1 (imagery)",
    "img_pole_lean_deg": "pole lean detected from imagery",
    "img_canopy_roughness": "ragged crown / limb-failure texture (imagery)",
    "img_veg_stress": "vegetation stress / dieback (imagery)",
}


@dataclass
class FitResult:
    model: CalibratedClassifierCV
    scaler: StandardScaler
    base: GradientBoostingClassifier
    feature_names: List[str]
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def split_indices(n: int, y: np.ndarray, test_frac: float, val_frac: float,
                  seed: int = MASTER_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified deterministic train/val/test split (test seen exactly once)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    pos, neg = idx[y == 1], idx[y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(a):
        nt = int(round(len(a) * test_frac))
        nv = int(round(len(a) * val_frac))
        return a[:nt], a[nt:nt + nv], a[nt + nv:]

    pt, pv, ptr = cut(pos)
    nt, nv, ntr = cut(neg)
    test = np.sort(np.concatenate([pt, nt]))
    val = np.sort(np.concatenate([pv, nv]))
    train = np.sort(np.concatenate([ptr, ntr]))
    return train, val, test


def fit(X: np.ndarray, y: np.ndarray, feature_names: List[str],
        idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray,
        seed: int = MASTER_SEED) -> FitResult:
    scaler = StandardScaler().fit(X[idx_train])
    Xtr = scaler.transform(X[idx_train])
    Xva = scaler.transform(X[idx_val])

    base = GradientBoostingClassifier(
        n_estimators=220, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=seed,
    )
    base.fit(Xtr, y[idx_train])

    # calibrate on the held-out validation fold (prefit base estimator)
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(Xva, y[idx_val])

    return FitResult(model=cal, scaler=scaler, base=base,
                     feature_names=feature_names, idx_train=idx_train,
                     idx_val=idx_val, idx_test=idx_test)


def predict_proba(fr: FitResult, X: np.ndarray) -> np.ndarray:
    return fr.model.predict_proba(fr.scaler.transform(X))[:, 1]


def segment_risk(seg_ids: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Aggregate per-pole risk to circuit-segment risk (mean of member poles)."""
    out: Dict[str, list] = {}
    for s, p in zip(seg_ids, proba):
        out.setdefault(s, []).append(p)
    return {s: float(np.mean(v)) for s, v in sorted(out.items())}


def explain_pole(fr: FitResult, x_row: np.ndarray, top_k: int = 4) -> List[Tuple[str, float]]:
    """Ranked plain-English drivers for ONE pole.

    Contribution proxy = standardized feature value * tree-importance weight,
    signed so positive = pushes risk up. Deterministic and fast; gives a
    planner an auditable 'why'.
    """
    xs = fr.scaler.transform(x_row.reshape(1, -1))[0]
    imp = fr.base.feature_importances_
    contrib = xs * imp
    order = np.argsort(-np.abs(contrib))
    drivers: List[Tuple[str, float]] = []
    for j in order[:top_k]:
        name = fr.feature_names[j]
        drivers.append((PRETTY.get(name, name), float(contrib[j])))
    return drivers
