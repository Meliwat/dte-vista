"""One-command entrypoint:  python -m vista

Runs the entire offline deterministic pipeline end-to-end:
  1. load REAL bundled NOAA climate normals + synthesize the DTE-style fleet
     (rendering a 2-band imagery tile per pole and recovering CV features),
  2. stratified train/val/test split (test seen once),
  3. fit + calibrate the explainable risk model,
  4. honest validation: held-out, calibration, spatial + temporal backtest
     distributions, lift vs the real incumbent age-cycle, imagery ablation,
  5. economic impact (externally-cited constants),
  6. render the single map-based dashboard PNG,
  7. print a deterministic text report (the demo's captured stdout).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import OUTPUT_DIR, TEST_FRACTION, VAL_FRACTION
from .data_gen import generate_fleet
from .impact import economic_impact
from .llm import generate_brief
from .model import explain_pole, fit, predict_proba, split_indices
from .noaa import load_noaa_normals, normals_summary
from .validation import run_validation
from .viz import render_dashboard


def main() -> None:
    np.random.seed(0)  # belt-and-suspenders global seed

    print("=" * 74)
    print("VISTA - Imagery-Led Predictive Utility-Pole Risk Profiling (DTE)")
    print("=" * 74)

    ns = normals_summary(load_noaa_normals())
    print(f"\n[1] REAL public artifact wired: NOAA 1991-2020 Climate Normals")
    print(f"    stations={ns['n_stations']}  mean ann precip="
          f"{ns['mean_ann_prcp_in']}in  mean ann snow={ns['mean_ann_snow_in']}in"
          f"  mean ann T={ns['mean_ann_tavg_f']}F")
    print(f"    counties: {', '.join(ns['counties'])}")

    print("\n[2] Synthesizing DTE-style fleet + rendering imagery tiles ...")
    fd = generate_fleet()
    print(f"    poles={len(fd.y):,}  failures={int(fd.y.sum()):,} "
          f"({fd.y.mean():.1%})  features={len(fd.feature_names)} "
          f"(tabular+image)  segments={len(set(fd.segment_id))}")
    n_img = sum(1 for f in fd.feature_names if f.startswith('img_'))
    print(f"    image-derived features per pole: {n_img} "
          f"(NDVI / encroachment / overhang / lean / change-detection / texture)")

    print("\n[3] Stratified split (test fold seen exactly once) ...")
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    print(f"    train={len(tr)}  val={len(va)}  test={len(te)}")

    print("\n[4] Fitting + calibrating explainable model ...")
    fr = fit(fd.X, fd.y, fd.feature_names, tr, va, te)

    print("\n[5] Validation (held-out + calibration + backtests + lift) ...")
    vr = run_validation(fd, fr)
    h = vr.heldout
    print(f"    HELD-OUT : ROC-AUC={h['roc_auc']}  PR-AUC={h['pr_auc']}  "
          f"Brier={h['brier']}  capture@20%={h['capture@20']}")
    print(f"    CALIB    : Brier={vr.brier}  weighted gap={vr.calib_gap}  "
          f"({len(vr.reliability)} populated bins)")
    print(f"    SPATIAL  : leave-county-group AUC = "
          f"{vr.spatial_summary[0]:.3f} +/- {vr.spatial_summary[1]:.3f}  "
          f"(n={len(vr.spatial_auc)} folds)  {vr.spatial_auc}")
    print(f"    TEMPORAL : storm-replay capture@20% = "
          f"{vr.temporal_summary[0]:.3f} +/- {vr.temporal_summary[1]:.3f}  "
          f"(n={len(vr.temporal_capture)} storms)")
    inc = vr.incumbent
    print(f"    LIFT     : VISTA caught {inc['capture_vista']:.0%} vs "
          f"age-cycle {inc['capture_age_cycle']:.0%} vs reactive "
          f"{inc['capture_fault_reactive']:.0%}  -> "
          f"{inc['lift_vs_age_cycle_x']:.2f}x age-cycle, "
          f"{inc['lift_vs_reactive_x']:.2f}x reactive")
    abl = vr.ablation
    print(f"    ABLATION : imagery adds +{abl['auc_gain_from_imagery']:.3f} "
          f"ROC-AUC and +{abl['capture20_gain_from_imagery']:.0%} capture "
          f"(with={abl['auc_with_imagery']:.3f} "
          f"without={abl['auc_without_imagery']:.3f})")

    print("\n[6] Economic impact (externally-cited constants) ...")
    econ = economic_impact(fr, fd)
    print(f"    budget={econ['budget_inspections_per_cycle_fleet']:,} "
          f"truck-rolls/cycle  extra failures caught="
          f"+{econ['extra_failures_caught_fleet']:.0f}")
    print(f"    avoided outage cost=${econ['avoided_outage_cost_fleet_usd']:,.0f}"
          f"  net=${econ['net_benefit_fleet_usd']:,.0f}  "
          f"benefit:cost={econ['benefit_cost_ratio']:.1f}x")

    print("\n[7] Top-5 prioritized poles (explainable drivers + LLM brief) ...")
    proba = predict_proba(fr, fd.X)
    for rank, j in enumerate(np.argsort(-proba, kind="stable")[:5], 1):
        drv = explain_pole(fr, fd.X[j], top_k=3)
        names = ", ".join(d for d, _ in drv)
        print(f"    #{rank} {fd.pole_id[j]} [{fd.segment_id[j]}] "
              f"{fd.county[j]}  risk={proba[j]:.0%}")
        print(f"        drivers: {names}")
        print(f"        brief: {generate_brief(fd.pole_id[j], proba[j], drv)}")

    print("\n[8] Rendering single map-based dashboard ...")
    out = render_dashboard(fd, fr, vr, econ)
    sz = os.path.getsize(out)
    print(f"    wrote {out}  ({sz:,} bytes)")

    # machine-readable artifact for tests / provenance
    summary = {
        "noaa": ns,
        "heldout": h,
        "calibration": {"brier": vr.brier, "gap": vr.calib_gap},
        "spatial_summary": vr.spatial_summary,
        "temporal_summary": vr.temporal_summary,
        "incumbent": inc,
        "ablation": abl,
        "economics": econ,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"    wrote {os.path.join(OUTPUT_DIR, 'summary.json')}")

    print("\n" + "=" * 74)
    print("DONE - reactive -> predictive, imagery-led, validated, in one figure.")
    print("=" * 74)


if __name__ == "__main__":
    main()
