"""Economic impact - the quantified, externally-cited $ figure.

Translates the validated capture-lift into avoided cost. Every constant is
sourced (see README "Economic basis & external citations"); this module only
does arithmetic on them so the headline number is auditable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import (
    AVOIDED_COST_PER_FAILURE_USD,
    COST_PER_INSPECTION_USD,
    INSPECTION_BUDGET_FRACTION,
)
from .data_gen import FleetData
from .model import FitResult, predict_proba
from .validation import _capture_at_budget


def economic_impact(fr: FitResult, fd: FleetData) -> Dict[str, float]:
    """Per planning-cycle economics on the held-out fold, scaled to the fleet.

    Compares VISTA's risk-ranked worklist to the incumbent age-based cycle at
    an IDENTICAL inspection budget. Net benefit = (extra failures caught x
    avoided cost) - (inspection spend), inspection spend equal for both since
    the budget (#truck-rolls) is held fixed -> the delta is pure prevention.
    """
    Xte, yte = fd.X[fr.idx_test], fd.y[fr.idx_test]
    fi = {n: j for j, n in enumerate(fd.feature_names)}
    age = Xte[:, fi["age_years"]]
    score = predict_proba(fr, Xte)
    b = INSPECTION_BUDGET_FRACTION

    n_test = len(yte)
    k = max(1, int(round(n_test * b)))
    failures_total = int(yte.sum())

    cap_v = _capture_at_budget(yte, score, b)
    cap_a = _capture_at_budget(yte, age, b)

    caught_v = cap_v * failures_total
    caught_a = cap_a * failures_total
    extra_caught = caught_v - caught_a

    inspection_spend = k * COST_PER_INSPECTION_USD  # equal for both policies
    avoided_value = extra_caught * AVOIDED_COST_PER_FAILURE_USD
    net = avoided_value - 0.0  # same budget -> spend cancels in the delta

    # scale held-out delta to the full synthetic fleet (proportional)
    scale = len(fd.y) / float(n_test)
    fleet_extra = extra_caught * scale
    fleet_avoided = fleet_extra * AVOIDED_COST_PER_FAILURE_USD
    fleet_inspection = (len(fd.y) * b) * COST_PER_INSPECTION_USD
    roi = fleet_avoided / max(1.0, fleet_inspection)

    return {
        "budget_inspections_per_cycle_fleet": int(round(len(fd.y) * b)),
        "inspection_spend_fleet_usd": round(fleet_inspection, 0),
        "extra_failures_caught_fleet": round(fleet_extra, 1),
        "avoided_outage_cost_fleet_usd": round(fleet_avoided, 0),
        "net_benefit_fleet_usd": round(fleet_avoided - fleet_inspection, 0),
        "benefit_cost_ratio": round(roi, 2),
        "extra_caught_per_100_truckrolls": round(
            extra_caught / max(1.0, k) * 100.0, 2),
    }
