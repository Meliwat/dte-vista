"""The single cohesive MAP-BASED dashboard.

One figure, the visualization the DTE brief implies: a real WGS84 map of the
DTE Southeast-Michigan territory with every pole plotted and colored by
calibrated risk, plus the panels a planner actually needs - imagery-forward
because that is VISTA's distinct axis:

  [A] MAP: geographic pole risk over the DTE territory (county overlay,
      NOAA stations, top-risk poles ringed) - the prioritization surface.
  [B] PRIORITIZED WORKLIST: ranked critical poles with drivers + action.
  [C] IMAGERY DRILL-DOWN: NDVI tile chips (t0->t1) for the #1 risk pole
      with the recovered image features - proves the imagery pipeline.
  [D] CIRCUIT-SEGMENT RISK: top segments (the per-segment deliverable).
  [E] CALIBRATION: reliability diagram (trust the probabilities).
  [F] LIFT: VISTA vs incumbent age-cycle vs reactive at equal budget,
      plus the imagery-ablation bar (the modality's measured payoff).

Pure matplotlib (Agg), deterministic, one PNG.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import COUNTIES, OUTPUT_DIR, TERRITORY_BBOX  # noqa: E402
from .data_gen import FleetData  # noqa: E402
from .imagery import _ndvi  # noqa: E402
from .model import FitResult, explain_pole, predict_proba, segment_risk  # noqa: E402
from .noaa import load_noaa_normals  # noqa: E402
from .validation import ValidationReport  # noqa: E402

RISK_CMAP = "RdYlGn_r"


def _tier(p: float) -> str:
    return ("CRITICAL" if p >= 0.65 else "HIGH" if p >= 0.40
            else "ELEVATED" if p >= 0.22 else "ROUTINE")


def render_dashboard(fd: FleetData, fr: FitResult, vr: ValidationReport,
                     econ: dict, out_path: str | None = None) -> str:
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "dashboard.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    proba = predict_proba(fr, fd.X)
    order = np.argsort(-proba, kind="stable")
    stations = load_noaa_normals()

    plt.rcParams.update({
        "font.size": 8.5, "axes.titleweight": "bold",
        "axes.edgecolor": "#444", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.35, 1.0, 1.0],
                           width_ratios=[1.55, 1.0, 1.0, 1.0],
                           hspace=0.34, wspace=0.30)

    fig.suptitle(
        "VISTA  -  Imagery-Led Predictive Utility-Pole Risk  |  DTE Energy "
        "Southeast-Michigan Territory  (offline - deterministic - public data)",
        fontsize=16, fontweight="bold", y=0.985)

    # ---- [A] MAP -------------------------------------------------------
    axm = fig.add_subplot(gs[0, 0:2])
    sc = axm.scatter(fd.lon, fd.lat, c=proba, cmap=RISK_CMAP, s=14,
                     vmin=0, vmax=1, edgecolors="none", alpha=0.85)
    topN = order[:25]
    axm.scatter(fd.lon[topN], fd.lat[topN], s=95, facecolors="none",
                edgecolors="black", linewidths=1.4, label="Top-25 priority poles")
    axm.scatter([s.lon for s in stations], [s.lat for s in stations],
                marker="^", s=85, c="#1f3b8c", edgecolors="white",
                linewidths=0.7, label="Real NOAA stations", zorder=5)
    # county longitude bands (the geographic context overlay)
    for i in range(1, len(COUNTIES)):
        x = (TERRITORY_BBOX["lon_min"]
             + i * (TERRITORY_BBOX["lon_max"] - TERRITORY_BBOX["lon_min"])
             / len(COUNTIES))
        axm.axvline(x, color="#999", lw=0.5, ls=":", alpha=0.6)
    for i, cname in enumerate(COUNTIES):
        x = (TERRITORY_BBOX["lon_min"]
             + (i + 0.5) * (TERRITORY_BBOX["lon_max"] - TERRITORY_BBOX["lon_min"])
             / len(COUNTIES))
        axm.text(x, TERRITORY_BBOX["lat_max"] - 0.04, cname, fontsize=6.3,
                 ha="center", va="top", rotation=90, color="#555")
    axm.set_xlim(TERRITORY_BBOX["lon_min"], TERRITORY_BBOX["lon_max"])
    axm.set_ylim(TERRITORY_BBOX["lat_min"], TERRITORY_BBOX["lat_max"])
    axm.set_title("[A] Geographic pole-risk map - DTE territory "
                  "(color = calibrated failure risk)")
    axm.set_xlabel("Longitude (WGS84)")
    axm.set_ylabel("Latitude (WGS84)")
    axm.legend(loc="lower left", fontsize=7, framealpha=0.9)
    cb = fig.colorbar(sc, ax=axm, fraction=0.035, pad=0.01)
    cb.set_label("Calibrated failure risk", fontsize=7.5)

    # ---- [B] PRIORITIZED WORKLIST -------------------------------------
    axw = fig.add_subplot(gs[0, 2:4])
    axw.set_xlim(0, 1)
    axw.set_ylim(0, 1)
    axw.axis("off")
    axw.set_title("[B] Prioritized inspection worklist  "
                  "(CRITICAL/HIGH, ranked - the worklist a planner runs)",
                  loc="left", fontsize=9.5)
    n_rows = 15
    hdr = f"{'POLE':<7}{'SEGMENT':<9}{'COUNTY':<11}{'RISK':<6}{'TIER':<10}TOP DRIVER"
    axw.text(0.01, 0.955, hdr, family="monospace", fontsize=7.8,
             fontweight="bold", va="top")
    axw.axhline(0.915, 0.0, 1.0, color="#888", lw=0.7)
    y_top, y_bot = 0.88, 0.10
    step = (y_top - y_bot) / (n_rows - 1)
    shown = 0
    for j in order:
        if proba[j] < 0.40 or shown >= n_rows:
            break
        drv = explain_pole(fr, fd.X[j], top_k=1)[0][0]
        yln = y_top - shown * step
        row = (f"{fd.pole_id[j]:<7}{fd.segment_id[j]:<9}"
               f"{fd.county[j][:10]:<11}{proba[j]:<6.2f}"
               f"{_tier(proba[j]):<10}{drv[:26]}")
        col = "#a11" if proba[j] >= 0.65 else "#b5651d"
        if shown % 2 == 0:
            axw.add_patch(plt.Rectangle((0.0, yln - step * 0.5), 1.0, step,
                                        color="#f4f4f4", zorder=0))
        axw.text(0.01, yln, row, family="monospace", fontsize=7.4,
                 va="center", color=col, zorder=2)
        shown += 1
    axw.text(0.01, 0.025,
             f"+ {max(0, int((proba >= 0.40).sum()) - shown)} more >= HIGH  |  "
             f"{int((proba >= 0.65).sum())} CRITICAL / "
             f"{int((proba >= 0.40).sum())} HIGH+ of {len(proba)} poles "
             f"({(proba >= 0.40).mean():.0%} of fleet flagged)",
             family="monospace", fontsize=7.2, style="italic", color="#555",
             va="center")

    # ---- [C] IMAGERY DRILL-DOWN (VISTA's distinct axis) ---------------
    top1 = int(order[0])
    gC = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, 0:2], wspace=0.18, width_ratios=[1, 1, 1.25])
    t0, t1 = fd.tiles_t0[top1], fd.tiles_t1[top1]
    nd0, nd1 = _ndvi(t0.astype(np.float64)), _ndvi(t1.astype(np.float64))
    axc0 = fig.add_subplot(gC[0, 0])
    axc0.imshow(nd0, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axc0.set_title(f"[C] NDVI tile t0\n{fd.pole_id[top1]}", fontsize=8.5)
    axc0.axvline(t0.shape[2] / 2, color="black", lw=0.8, ls="--")
    axc0.set_xticks([]); axc0.set_yticks([])
    axc1 = fig.add_subplot(gC[0, 1])
    im = axc1.imshow(nd1, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axc1.set_title("NDVI tile t1\n(current)", fontsize=8.5)
    axc1.axvline(t1.shape[2] / 2, color="black", lw=0.8, ls="--")
    axc1.set_xticks([]); axc1.set_yticks([])
    fig.colorbar(im, ax=axc1, fraction=0.046, pad=0.03).set_label(
        "NDVI", fontsize=6.5)
    axcf = fig.add_subplot(gC[0, 2])
    axcf.axis("off")
    fi = {n: k for k, n in enumerate(fd.feature_names)}
    img_feat_lines = [
        ("RoW canopy", fd.X[top1, fi["img_row_canopy"]]),
        ("Overhang", fd.X[top1, fi["img_overhang"]]),
        ("RoW growth t0->t1", fd.X[top1, fi["img_row_growth"]]),
        ("Pole lean (deg)", fd.X[top1, fi["img_pole_lean_deg"]]),
        ("NDVI mean", fd.X[top1, fi["img_ndvi_mean"]]),
        ("Crown roughness", fd.X[top1, fi["img_canopy_roughness"]]),
        ("Veg stress", fd.X[top1, fi["img_veg_stress"]]),
    ]
    axcf.text(0.0, 1.0, "Image-derived features\n(recovered from pixels)",
              fontsize=8.0, fontweight="bold", va="top",
              transform=axcf.transAxes)
    yy = 0.78
    for nm, val in img_feat_lines:
        axcf.text(0.0, yy, f"{nm:<19}{val:>7.3f}", family="monospace",
                  fontsize=7.6, va="top", transform=axcf.transAxes)
        yy -= 0.108
    axcf.text(0.0, yy - 0.02,
              f"-> calibrated risk {proba[top1]:.0%}  ({_tier(proba[top1])})",
              fontsize=8.0, fontweight="bold", color="#a11", va="top",
              transform=axcf.transAxes)

    # ---- [D] CIRCUIT-SEGMENT RISK -------------------------------------
    axd = fig.add_subplot(gs[1, 2])
    seg = segment_risk(fd.segment_id, proba)
    items = sorted(seg.items(), key=lambda kv: -kv[1])[:12]
    names = [k for k, _ in items][::-1]
    vals = np.array([v for _, v in items][::-1])
    # normalize colors across the displayed range so the ranking is legible
    cnorm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    bars = axd.barh(names, vals, color=plt.cm.RdYlGn_r(0.25 + 0.7 * cnorm))
    axd.set_title("[D] Top circuit-segment risk\n(per-segment deliverable)",
                  fontsize=9)
    axd.set_xlabel("Mean segment failure risk")
    axd.set_xlim(0, max(vals) * 1.18)
    axd.tick_params(axis="y", labelsize=6.5)
    for b, v in zip(bars, vals):
        axd.text(v + max(vals) * 0.012, b.get_y() + b.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=6.4, fontweight="bold")

    # ---- [E] CALIBRATION ----------------------------------------------
    axe = fig.add_subplot(gs[1, 3])
    if vr.reliability:
        mp = [r[0] for r in vr.reliability]
        ef = [r[1] for r in vr.reliability]
        axe.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        axe.plot(mp, ef, "o-", color="#1f3b8c", ms=4, label="VISTA")
        axe.set_xlim(0, 1); axe.set_ylim(0, 1)
    axe.set_title("[E] Calibration reliability", fontsize=9)
    axe.set_xlabel("Predicted")
    axe.set_ylabel("Observed")
    axe.legend(fontsize=6.5, loc="upper left")
    axe.text(0.97, 0.05,
             f"Brier {vr.brier:.3f}\ngap {vr.calib_gap:.3f}\n"
             f"AUC {vr.heldout['roc_auc']:.3f}",
             transform=axe.transAxes, ha="right", va="bottom", fontsize=7,
             bbox=dict(boxstyle="round", fc="#eef", ec="#88a"))

    # ---- [F] LIFT vs incumbent + imagery ablation ---------------------
    axf = fig.add_subplot(gs[2, 0])
    inc = vr.incumbent
    labels = ["VISTA", "Age\ncycle", "Reactive\n(faults)", "Random"]
    caps = [inc["capture_vista"], inc["capture_age_cycle"],
            inc["capture_fault_reactive"], inc["capture_random"]]
    colb = ["#1a7d3c", "#b5651d", "#9a3b3b", "#888"]
    bb = axf.bar(labels, caps, color=colb)
    axf.set_ylim(0, 1)
    axf.set_title("[F] Failures caught @ equal budget\n(20% of fleet inspected)",
                  fontsize=9)
    axf.set_ylabel("Frac. true failures caught")
    for b, v in zip(bb, caps):
        axf.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                 ha="center", fontsize=7.5, fontweight="bold")

    # ---- [G] imagery ablation -----------------------------------------
    axg = fig.add_subplot(gs[2, 1])
    abl = vr.ablation
    axg.bar(["with\nimagery", "without\nimagery"],
            [abl["auc_with_imagery"], abl["auc_without_imagery"]],
            color=["#1a7d3c", "#aaa"])
    axg.set_ylim(0.5, 1.0)
    axg.set_title("[G] Imagery modality payoff\n(held-out ROC-AUC)", fontsize=9)
    for i, v in enumerate([abl["auc_with_imagery"], abl["auc_without_imagery"]]):
        axg.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8,
                 fontweight="bold")
    axg.text(0.5, 0.52, f"+{abl['auc_gain_from_imagery']:.3f} AUC\n"
             f"+{abl['capture20_gain_from_imagery']:.0%} capture",
             transform=axg.transAxes, ha="center", fontsize=7.5,
             bbox=dict(boxstyle="round", fc="#dfe", ec="#7a7"))

    # ---- [H] backtest distributions (twin axis: different scales) ------
    axh = fig.add_subplot(gs[2, 2])
    bp1 = axh.boxplot([vr.spatial_auc], positions=[1], widths=0.5,
                      showmeans=True, patch_artist=True)
    for b in bp1["boxes"]:
        b.set_facecolor("#cfe3ff")
    axh.set_ylim(min(vr.spatial_auc) - 0.03, max(vr.spatial_auc) + 0.03)
    axh.set_ylabel("Spatial ROC-AUC", color="#1f3b8c", fontsize=8)
    axh.tick_params(axis="y", labelcolor="#1f3b8c")
    axh2 = axh.twinx()
    bp2 = axh2.boxplot([vr.temporal_capture], positions=[2], widths=0.5,
                       showmeans=True, patch_artist=True)
    for b in bp2["boxes"]:
        b.set_facecolor("#ffe0cc")
    axh2.set_ylim(min(vr.temporal_capture) - 0.05,
                  max(vr.temporal_capture) + 0.05)
    axh2.set_ylabel("Temporal capture@20%", color="#b5651d", fontsize=8)
    axh2.tick_params(axis="y", labelcolor="#b5651d")
    axh.set_xticks([1, 2])
    axh.set_xticklabels(["Spatial\n(leave-county,\nn=%d)" % len(vr.spatial_auc),
                         "Temporal\n(storm replays,\nn=%d)"
                         % len(vr.temporal_capture)], fontsize=7)
    axh.set_xlim(0.5, 2.5)
    axh.set_title("[H] Backtest DISTRIBUTIONS\n(spatial + temporal, not 1 split)",
                  fontsize=9)
    axh.text(0.5, -0.02,
             f"spatial {vr.spatial_summary[0]:.3f}±{vr.spatial_summary[1]:.3f}  |  "
             f"temporal {vr.temporal_summary[0]:.2f}±{vr.temporal_summary[1]:.2f}",
             transform=axh.transAxes, ha="center", va="top", fontsize=6.6,
             bbox=dict(boxstyle="round", fc="#eef", ec="#88a"))

    # ---- [I] economic headline ----------------------------------------
    axi = fig.add_subplot(gs[2, 3])
    axi.axis("off")
    axi.set_title("[I] Reactive -> predictive, quantified", fontsize=9,
                  loc="left")
    lines = [
        f"Budget: {econ['budget_inspections_per_cycle_fleet']:,} truck-rolls/cycle",
        f"Extra failures caught: +{econ['extra_failures_caught_fleet']:.0f}",
        f"Avoided outage cost: ${econ['avoided_outage_cost_fleet_usd']:,.0f}",
        f"Net benefit: ${econ['net_benefit_fleet_usd']:,.0f}",
        f"Benefit : cost = {econ['benefit_cost_ratio']:.1f}x",
        f"Lift vs age-cycle: {inc['lift_vs_age_cycle_x']:.1f}x "
        f"(+{inc['lift_vs_age_cycle_pts']:.0%} pts)",
    ]
    yy = 0.86
    for ln in lines:
        axi.text(0.0, yy, ln, fontsize=9.2, va="top",
                 transform=axi.transAxes,
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        yy -= 0.155
    axi.add_patch(plt.Rectangle((-0.02, -0.03), 1.05, 1.0, fill=False,
                                edgecolor="#1a7d3c", lw=2,
                                transform=axi.transAxes, clip_on=False))

    fig.text(0.5, 0.005,
             "Data: synthetic DTE-style fleet anchored to REAL NOAA 1991-2020 "
             "climate normals (bundled) + image-derived NDVI/structure features "
             "from rendered raster tiles. No proprietary inputs.",
             ha="center", fontsize=7.5, style="italic", color="#555")

    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out_path
