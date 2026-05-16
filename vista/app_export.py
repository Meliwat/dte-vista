"""Interactive offline web app exporter — the demoable DTE deliverable.

`build_app(fd, fr, vr, econ)` turns the already-computed pipeline state into:

  * ``output/app_data.json`` — a deterministic, sorted JSON snapshot of the
    fleet (every pole's calibrated risk, ranked drivers, recommended action),
    the validated KPIs, and provenance metadata.
  * ``output/app.html`` — ONE fully self-contained file. The JSON above is
    embedded inline in a ``<script type="application/json">`` block (never
    fetched), and the UI is pure vanilla JS + inline CSS + inline SVG. Zero
    external resources: no CDN, no web fonts, no map tiles, no network. It
    opens by double-click via ``file://`` and works fully offline.

The app is a "grid instrument console": a substation-HMI / oscilloscope
aesthetic over the real engine — an interactive territory field (poles
projected from real WGS84 lon/lat via ``TERRITORY_BBOX``, luminous nodes,
a one-time power-on energize sweep, a locked-target reticle on selection),
click-to-drill-down per-pole "why" dossiers, tier/county/min-risk filters,
a calibrated inspection-budget fader that shows the reactive→predictive
coverage story live, a sortable prioritized worklist, and a client-side CSV
export of the current worklist.

Determinism: this module only reads the (deterministic) pipeline state and
emits sorted JSON / a templated HTML string — no RNG, no clock, no I/O
beyond the two output files. Two runs are byte-identical.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .config import COUNTIES, OUTPUT_DIR, TERRITORY_BBOX
from .data_gen import FleetData
from .model import FitResult, explain_pole, predict_proba, segment_risk
from .noaa import load_noaa_normals
from .validation import ValidationReport

# Fixed build date string — keeps the embedded JSON byte-identical across runs
# and machines (the app is provenance-stamped, not wall-clock-stamped).
GENERATED_DATE = "2026-05-16"

# Tier thresholds — MUST match vista.viz._tier so the map, badges and the
# static PNG tell the identical story.
TIER_THRESHOLDS = [
    {"tier": "CRITICAL", "min": 0.65, "color": "#d4322c"},
    {"tier": "HIGH", "min": 0.40, "color": "#f08a24"},
    {"tier": "ELEVATED", "min": 0.22, "color": "#f3c613"},
    {"tier": "ROUTINE", "min": 0.0, "color": "#3a9d4e"},
]


def _tier(p: float) -> str:
    return ("CRITICAL" if p >= 0.65 else "HIGH" if p >= 0.40
            else "ELEVATED" if p >= 0.22 else "ROUTINE")


def _action_for_driver(driver_name: str) -> str:
    """Recommended field action derived from a pole's #1 risk driver."""
    n = driver_name.lower()
    if any(k in n for k in ("canopy", "ndvi", "overhang", "row",
                            "right-of-way", "veg", "growth")):
        return "TREE TRIM"
    if any(k in n for k in ("lean", "age", "material", "span", "structural",
                            "wood", "construction")):
        return "INSPECT/REPLACE"
    if any(k in n for k in ("flood", "soil", "corros", "moist")):
        return "INSPECT (ground)"
    return "INSPECT"


# --------------------------------------------------------------------------
# Citizen-corroboration overlay.
#
# The moderated, repo-committed ledger at community_reports/ledger.jsonl is
# read here and folded into the payload as an OVERLAY ONLY. It is reconciled
# against the model output as a priority / corroboration signal; it MUST NOT
# influence predict_proba, the model, drivers, tiers, or risk ordering, and it
# is never model training input. Ingestion is fully deterministic (a fixed
# committed file, normalized, sorted by report_id) so two pipeline runs over
# the same ledger produce a byte-identical app_data.json. No clock, no RNG,
# no env, no network — consistent with the rest of VISTA.
# --------------------------------------------------------------------------

# repo-root/community_reports/ledger.jsonl (this file lives in repo-root/vista)
COMMUNITY_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "community_reports", "ledger.jsonl")

# Statuses the pipeline ingests; "rejected" rows are dropped entirely.
_INGEST_STATUSES = ("verified", "pending")
_VALID_SEVERITY = ("low", "medium", "urgent")
_VALID_SOURCE = ("resident", "lineman", "sample")


def _norm_report(rec: dict) -> dict | None:
    """Normalize one raw ledger record into a fixed, deterministic shape.

    Returns None if the record is unusable or not in an ingestible status
    (so callers can simply skip falsy results). Pure: no clock/RNG/I/O.
    """
    if not isinstance(rec, dict):
        return None
    status = str(rec.get("status", "")).strip().lower()
    if status not in _INGEST_STATUSES:
        return None
    rid = rec.get("report_id")
    if rid is None or str(rid).strip() == "":
        return None
    pid = rec.get("pole_id")
    pole_id = None if pid is None else str(pid)
    try:
        lat = round(float(rec.get("lat")), 6)
        lon = round(float(rec.get("lon")), 6)
    except (TypeError, ValueError):
        return None
    conds = rec.get("conditions") or []
    if not isinstance(conds, list):
        conds = [str(conds)]
    conditions = [str(c) for c in conds]
    severity = str(rec.get("severity", "")).strip().lower()
    if severity not in _VALID_SEVERITY:
        severity = "medium"
    source = str(rec.get("source", "")).strip().lower()
    if source not in _VALID_SOURCE:
        source = "resident"
    return {
        "report_id": str(rid),
        "pole_id": pole_id,
        "lat": lat,
        "lon": lon,
        "county": str(rec.get("county", "")),
        "conditions": conditions,
        "severity": severity,
        "note": str(rec.get("note", "")),
        "reporter": str(rec.get("reporter", "")),
        "submitted": str(rec.get("submitted", "")),
        "status": status,
        "source": source,
    }


def _load_community_reports(path: str = COMMUNITY_LEDGER_PATH) -> List[dict]:
    """Read the moderated JSON-Lines ledger deterministically.

    Missing file -> empty list (never crash). Malformed lines are skipped.
    Only verified/pending rows survive; rejected rows are dropped. The result
    is sorted by report_id so the payload is byte-identical across runs.
    """
    reports: List[dict] = []
    if not os.path.isfile(path):
        return reports
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_lines = fh.read().splitlines()
    except OSError:
        return reports
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        norm = _norm_report(rec)
        if norm is not None:
            reports.append(norm)
    reports.sort(key=lambda r: r["report_id"])
    return reports


def _community_overlay(reports: List[dict]) -> dict:
    """Build the deterministic top-level community overlay block.

    Pure summary over the already-normalized, already-sorted reports. This is
    an OVERLAY: it carries corroboration/priority metadata only and never
    feeds the model.
    """
    n = len(reports)
    n_verified = sum(1 for r in reports if r["status"] == "verified")
    n_pending = sum(1 for r in reports if r["status"] == "pending")
    n_unmapped = sum(1 for r in reports if r["pole_id"] is None)
    # poles corroborated = distinct pole_ids with >=1 VERIFIED report
    corroborated = sorted({
        r["pole_id"] for r in reports
        if r["pole_id"] is not None and r["status"] == "verified"})
    by_sev = {s: sum(1 for r in reports if r["severity"] == s)
              for s in _VALID_SEVERITY}
    return {
        "reports": reports,  # already sorted by report_id
        "summary": {
            "n_reports": n,
            "n_verified": n_verified,
            "n_pending": n_pending,
            "n_unmapped": n_unmapped,
            "n_poles_corroborated": len(corroborated),
            "by_severity": by_sev,
        },
        "ledger_path": "community_reports/ledger.jsonl",
        "note": ("Citizen-corroboration overlay. Moderated, repo-committed "
                 "field reports reconciled against the model as a "
                 "priority/corroboration signal — never model input."),
    }


def _build_payload(fd: FleetData, fr: FitResult, vr: ValidationReport,
                    econ: dict) -> dict:
    """Assemble the deterministic data dict embedded in the app."""
    proba = predict_proba(fr, fd.X)
    seg = segment_risk(np.asarray(fd.segment_id), proba)
    stations = load_noaa_normals()

    # Citizen-corroboration overlay (read-only, deterministic, NOT model input)
    community_reports = _load_community_reports()
    # per-pole corroboration index: pole_id -> (n_total, has_verified)
    _comm_idx: Dict[str, list] = {}
    for r in community_reports:
        pid = r["pole_id"]
        if pid is None:
            continue
        slot = _comm_idx.setdefault(pid, [0, False])
        slot[0] += 1
        if r["status"] == "verified":
            slot[1] = True

    # poles, sorted by risk desc (stable tie-break by index → deterministic)
    order = np.argsort(-proba, kind="stable")
    poles: List[dict] = []
    for j in order:
        j = int(j)
        drv = explain_pole(fr, fd.X[j], top_k=5)
        drivers = [[str(name), round(float(c), 4)] for name, c in drv]
        top_driver = drivers[0][0] if drivers else ""
        pid = str(fd.pole_id[j])
        c_n, c_ver = _comm_idx.get(pid, (0, False))
        poles.append({
            "id": pid,
            "lat": round(float(fd.lat[j]), 6),
            "lon": round(float(fd.lon[j]), 6),
            "county": str(fd.county[j]),
            "segment": str(fd.segment_id[j]),
            "risk": round(float(proba[j]), 4),
            "tier": _tier(float(proba[j])),
            "drivers": drivers,
            "action": _action_for_driver(top_driver),
            # Citizen-corroboration overlay (NOT a model input / not in risk).
            "community_n": int(c_n),
            "community_status": "corroborated" if c_ver else (
                "reported" if c_n else "none"),
        })

    segments = [
        {"id": str(s), "risk": round(float(r), 4),
         "n_poles": int(np.sum(np.asarray(fd.segment_id) == s))}
        for s, r in sorted(seg.items(), key=lambda kv: -kv[1])
    ]

    h = vr.heldout
    inc = vr.incumbent
    abl = vr.ablation

    payload = {
        "meta": {
            "title": "VISTA — Utility Pole Risk Profiling",
            "subtitle": ("DTE Energy · Hack Michigan 2026 · offline · "
                         "public data · deterministic"),
            "generated": GENERATED_DATE,
            "territory_bbox": {
                "lon_min": float(TERRITORY_BBOX["lon_min"]),
                "lon_max": float(TERRITORY_BBOX["lon_max"]),
                "lat_min": float(TERRITORY_BBOX["lat_min"]),
                "lat_max": float(TERRITORY_BBOX["lat_max"]),
            },
            "counties": list(COUNTIES),
            "noaa_stations": [
                {"name": s.name, "lon": round(float(s.lon), 6),
                 "lat": round(float(s.lat), 6)}
                for s in stations
            ],
            "tier_thresholds": TIER_THRESHOLDS,
            "n_poles": int(len(fd.y)),
        },
        "kpis": {
            "roc_auc": round(float(h["roc_auc"]), 4),
            "pr_auc": round(float(h["pr_auc"]), 4),
            "brier": round(float(h["brier"]), 4),
            "capture_at_20": round(float(h["capture@20"]), 4),
            "calibration_gap": round(float(vr.calib_gap), 4),
            "spatial_summary": [round(float(vr.spatial_summary[0]), 4),
                                round(float(vr.spatial_summary[1]), 4)],
            "temporal_summary": [round(float(vr.temporal_summary[0]), 4),
                                 round(float(vr.temporal_summary[1]), 4)],
            "incumbent": {
                "capture_vista": round(float(inc["capture_vista"]), 4),
                "capture_age_cycle": round(float(inc["capture_age_cycle"]), 4),
                "capture_fault_reactive":
                    round(float(inc["capture_fault_reactive"]), 4),
                "capture_random": round(float(inc["capture_random"]), 4),
                "lift_vs_age_cycle_x":
                    round(float(inc["lift_vs_age_cycle_x"]), 2),
                "lift_vs_reactive_x":
                    round(float(inc["lift_vs_reactive_x"]), 2),
            },
            "ablation": {
                "auc_with_imagery": round(float(abl["auc_with_imagery"]), 4),
                "auc_without_imagery":
                    round(float(abl["auc_without_imagery"]), 4),
                "auc_gain_from_imagery":
                    round(float(abl["auc_gain_from_imagery"]), 4),
                "capture20_gain_from_imagery":
                    round(float(abl["capture20_gain_from_imagery"]), 4),
            },
            "economics": {
                "budget_inspections_per_cycle_fleet":
                    int(econ["budget_inspections_per_cycle_fleet"]),
                "extra_failures_caught_fleet":
                    round(float(econ["extra_failures_caught_fleet"]), 1),
                "avoided_outage_cost_fleet_usd":
                    round(float(econ["avoided_outage_cost_fleet_usd"]), 0),
                "net_benefit_fleet_usd":
                    round(float(econ["net_benefit_fleet_usd"]), 0),
                "benefit_cost_ratio":
                    round(float(econ["benefit_cost_ratio"]), 2),
            },
        },
        "poles": poles,
        "segments": segments,
        # Citizen-corroboration overlay — deterministic, sorted by report_id,
        # reconciled against the model as a priority/corroboration signal.
        # NOT model input; does not affect risk, tiers, drivers or ordering.
        "community": _community_overlay(community_reports),
    }
    return payload


# --------------------------------------------------------------------------
# The single-file app shell. The data JSON is injected into the marked script
# tag; the CSS and JS below are 100% inline and reference NO external URL.
# Aesthetic: a high-voltage grid instrument console (substation HMI /
# oscilloscope) — void field, blueprint grid, amber signal accent, luminous
# risk nodes, HUD registration brackets, a one-time power-on energize sweep,
# monospace telemetry, a locked-target reticle on selection.
# --------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VISTA — Utility Pole Risk Profiling · DTE · Hack Michigan 2026</title>
<style>
  :root{
    --void:#05070a; --void2:#070a0f; --panel:#0a0e14; --panel2:#0c1119;
    --line:#17202e; --line2:#202c3e; --ink:#eaf1f8; --dim:#7b8aa0;
    --mute:#54637a; --hv:#ffb01f; --hv2:#ff8a00; --cool:#36e0ff;
    --crit:#ff394e; --high:#ff8f1f; --elev:#ffd23f; --rout:#1fd98c;
    --grid:rgba(76,103,150,.10);
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--void);color:var(--ink);}
  body{
    min-height:100vh;
    font-family:ui-sans-serif,"SF Pro Text",-apple-system,BlinkMacSystemFont,
      "Helvetica Neue",Arial,sans-serif;
    -webkit-font-smoothing:antialiased; letter-spacing:.1px;
    /* atmosphere: blueprint grid + phosphor vignette over the void */
    background-image:
      radial-gradient(120% 90% at 50% -10%,rgba(255,176,31,.07),transparent 60%),
      radial-gradient(100% 100% at 50% 120%,rgba(54,224,255,.05),transparent 55%),
      linear-gradient(var(--grid) 1px,transparent 1px),
      linear-gradient(90deg,var(--grid) 1px,transparent 1px);
    background-size:auto,auto,42px 42px,42px 42px;
    background-position:center,center,center,center;
  }
  body::after{ /* faint sweeping scanline — the console is "live" */
    content:""; position:fixed; inset:0; pointer-events:none; z-index:90;
    background:linear-gradient(180deg,transparent,rgba(255,176,31,.045) 50%,
      transparent); height:42%; animation:scan 7.5s linear infinite;
    mix-blend-mode:screen; opacity:.5;
  }
  @keyframes scan{0%{transform:translateY(-110%)}100%{transform:translateY(320%)}}
  @keyframes power{0%{opacity:0;filter:brightness(2.4) blur(3px)}
    14%{opacity:1;filter:brightness(.4)}22%{filter:brightness(1.5)}
    40%{filter:brightness(.85)}100%{opacity:1;filter:none}}
  @keyframes rise{from{opacity:0;transform:translateY(14px)}
    to{opacity:1;transform:none}}
  @keyframes lamp{0%,100%{opacity:1}50%{opacity:.28}}
  .mono{font-family:ui-monospace,"SF Mono","JetBrains Mono","IBM Plex Mono",
    Menlo,Consolas,monospace; font-variant-numeric:tabular-nums;}
  a{color:var(--hv);text-decoration:none;} a:hover{text-decoration:underline;}

  body{animation:power .7s ease-out both;}
  header.app,.kpis,.layout>*{animation:rise .55s cubic-bezier(.2,.7,.2,1) both;}
  header.app{animation-delay:.10s;} .kpis{animation-delay:.18s;}
  .layout>aside:first-child{animation-delay:.26s;}
  .layout>main{animation-delay:.32s;}
  .layout>aside.right{animation-delay:.38s;}

  /* ---- header: instrument title block ---- */
  header.app{display:flex;align-items:center;justify-content:space-between;
    padding:15px 26px;border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,#0b1018,#06090e);position:relative;}
  header.app::before{content:"";position:absolute;left:0;right:0;bottom:-1px;
    height:1px;background:linear-gradient(90deg,transparent,var(--hv),
    transparent);opacity:.5;}
  .brand{display:flex;align-items:center;gap:16px;}
  .glyph{width:30px;height:30px;flex:none;position:relative;}
  .glyph::before,.glyph::after{content:"";position:absolute;}
  .glyph::before{inset:0;border:1.5px solid var(--hv);
    clip-path:polygon(50% 0,100% 50%,50% 100%,0 50%);}
  .glyph::after{inset:9px;background:var(--hv);
    box-shadow:0 0 12px var(--hv);
    clip-path:polygon(58% 0,42% 54%,70% 54%,40% 100%,52% 46%,28% 46%);}
  .brand h1{font-size:16px;margin:0;font-weight:800;letter-spacing:3.5px;}
  .brand .tag{display:block;color:var(--dim);font-size:10px;
    letter-spacing:3px;margin-top:3px;text-transform:uppercase;}
  .status{display:flex;align-items:center;gap:20px;}
  .lamp{display:flex;align-items:center;gap:8px;color:var(--dim);
    font-size:10px;letter-spacing:2px;text-transform:uppercase;}
  .lamp i{width:8px;height:8px;border-radius:50%;background:var(--rout);
    box-shadow:0 0 9px var(--rout);animation:lamp 2.4s ease-in-out infinite;}
  .badge{font-size:10px;font-weight:800;letter-spacing:2.5px;
    padding:7px 13px;color:var(--void);
    background:linear-gradient(95deg,var(--hv),var(--hv2));
    box-shadow:0 0 18px rgba(255,176,31,.35);
    clip-path:polygon(8px 0,100% 0,calc(100% - 8px) 100%,0 100%);}

  /* ---- KPI telemetry rail ---- */
  .kpis{display:flex;gap:1px;padding:0;border-bottom:1px solid var(--line);
    background:var(--line);}
  .kpi{background:var(--panel2);padding:14px 20px;min-width:172px;flex:1;
    position:relative;}
  .kpi .label{color:var(--mute);font-size:9.5px;text-transform:uppercase;
    letter-spacing:2px;}
  .kpi .value{font-size:25px;font-weight:800;margin-top:7px;
    font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;
    letter-spacing:-.5px;}
  .kpi .value.sm{font-size:15px;font-weight:700;letter-spacing:1px;}
  .kpi::before{content:"";position:absolute;left:0;top:14px;bottom:14px;
    width:2px;background:var(--line2);}
  .kpi.headline{background:linear-gradient(135deg,#1a1405,#0c0d12);}
  .kpi.headline::before{background:var(--hv);box-shadow:0 0 12px var(--hv);}
  .kpi.headline .label{color:var(--hv);}
  .kpi.headline .value{color:var(--hv);text-shadow:0 0 22px rgba(255,176,31,.5);}

  /* ---- layout + HUD panels ---- */
  .layout{display:grid;grid-template-columns:266px 1fr 396px;
    min-height:calc(100vh - 176px);}
  .panel{padding:20px;border-right:1px solid var(--line);position:relative;}
  .panel.right{border-right:none;border-left:1px solid var(--line);
    overflow-y:auto;max-height:calc(100vh - 176px);}
  .center{padding:16px 18px;display:flex;flex-direction:column;gap:14px;
    min-width:0;}
  .sec{font-size:10px;text-transform:uppercase;letter-spacing:2.5px;
    color:var(--dim);margin:0 0 13px;font-weight:700;
    display:flex;align-items:center;gap:9px;}
  .sec::after{content:"";flex:1;height:1px;
    background:linear-gradient(90deg,var(--line2),transparent);}
  .ctl{margin-bottom:26px;}

  /* breaker-style tier toggles */
  .chips{display:flex;flex-direction:column;gap:7px;}
  .chip{font-size:11px;padding:9px 12px;border:1px solid var(--line2);
    background:var(--panel);cursor:pointer;user-select:none;display:flex;
    align-items:center;gap:10px;letter-spacing:1.5px;font-weight:700;
    transition:all .14s;clip-path:polygon(0 0,100% 0,100% 100%,7px 100%);}
  .chip .dot{width:10px;height:10px;border-radius:50%;flex:none;
    box-shadow:0 0 8px currentColor;}
  .chip .ct{margin-left:auto;color:var(--mute);font-size:10px;
    font-family:ui-monospace,Menlo,monospace;}
  .chip.off{opacity:.34;filter:saturate(.2);}
  .chip.off .dot{box-shadow:none;}
  .chip:hover{border-color:var(--hv);background:#0e131c;}

  select{width:100%;background:var(--panel);color:var(--ink);
    border:1px solid var(--line2);padding:9px 10px;font-size:12px;
    letter-spacing:.5px;-webkit-appearance:none;cursor:pointer;}
  select:focus{outline:1px solid var(--hv);}

  input[type=range]{-webkit-appearance:none;width:100%;height:30px;
    background:transparent;cursor:pointer;margin-top:4px;}
  input[type=range]::-webkit-slider-runnable-track{height:4px;
    background:linear-gradient(90deg,var(--hv),#3a4761);}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;
    width:14px;height:22px;background:var(--hv);margin-top:-9px;
    box-shadow:0 0 10px rgba(255,176,31,.6);
    clip-path:polygon(50% 0,100% 30%,100% 100%,0 100%,0 30%);}
  .val{font-size:12px;color:var(--ink);margin-top:9px;letter-spacing:1px;}
  .hint{font-size:10.5px;color:var(--mute);margin-top:7px;line-height:1.6;}
  .gauge{margin-top:12px;border:1px solid var(--line2);background:var(--panel);
    padding:13px;position:relative;
    clip-path:polygon(0 0,100% 0,100% calc(100% - 9px),calc(100% - 9px) 100%,
      0 100%);}
  .gauge::before{content:"COVERAGE";position:absolute;top:-1px;right:10px;
    font-size:8px;letter-spacing:2px;color:var(--mute);
    background:var(--panel);padding:0 5px;transform:translateY(-50%);}
  .gauge b{color:var(--hv);font-family:ui-monospace,Menlo,monospace;
    font-size:16px;}
  .gauge .react{color:var(--mute);font-size:11px;}
  .gauge .big{font-size:13px;line-height:1.7;}

  /* HUD-bracketed map field */
  .mapwrap{border:1px solid var(--line2);background:var(--void2);
    padding:7px;position:relative;}
  .mapwrap::before,.mapwrap::after{content:"";position:absolute;width:16px;
    height:16px;border-color:var(--hv);pointer-events:none;}
  .mapwrap::before{top:-1px;left:-1px;border-top:2px solid;border-left:2px solid;}
  .mapwrap::after{bottom:-1px;right:-1px;border-bottom:2px solid;
    border-right:2px solid;}
  .fieldhdr{display:flex;justify-content:space-between;align-items:center;
    padding:3px 8px 9px;font-size:9.5px;letter-spacing:2.5px;
    color:var(--dim);text-transform:uppercase;}
  .fieldhdr .rt{color:var(--hv);}
  svg.map{width:100%;height:auto;display:block;}
  .legend{display:flex;gap:18px;flex-wrap:wrap;padding:10px 8px 2px;
    font-size:10px;color:var(--dim);letter-spacing:1px;text-transform:uppercase;}
  .legend span{display:flex;align-items:center;gap:7px;}
  .legend i{width:10px;height:10px;border-radius:50%;display:inline-block;
    box-shadow:0 0 7px currentColor;}
  #tooltip{position:fixed;pointer-events:none;background:#04070bee;
    border:1px solid var(--hv);color:var(--ink);font-size:11px;padding:8px 11px;
    display:none;z-index:120;white-space:nowrap;letter-spacing:.5px;
    box-shadow:0 0 22px rgba(255,176,31,.25);
    clip-path:polygon(0 0,100% 0,100% 100%,6px 100%);}
  #tooltip b{color:var(--hv);}

  /* worklist console */
  .tablewrap{border:1px solid var(--line2);background:var(--void2);flex:1;
    display:flex;flex-direction:column;min-height:226px;position:relative;}
  .tablebar{display:flex;align-items:center;justify-content:space-between;
    padding:12px 15px;border-bottom:1px solid var(--line);}
  .tablebar strong{font-size:11px;letter-spacing:2.5px;text-transform:uppercase;}
  .tablebar .count{color:var(--mute);font-size:10.5px;letter-spacing:1px;
    margin-left:8px;font-family:ui-monospace,Menlo,monospace;}
  button.exp{background:linear-gradient(95deg,var(--hv),var(--hv2));
    color:var(--void);border:none;padding:9px 16px;font-size:11px;
    font-weight:800;cursor:pointer;letter-spacing:1.5px;
    box-shadow:0 0 16px rgba(255,176,31,.3);
    clip-path:polygon(7px 0,100% 0,calc(100% - 7px) 100%,0 100%);}
  button.exp:hover{filter:brightness(1.12);}
  .tscroll{overflow:auto;flex:1;max-height:336px;}
  table{width:100%;border-collapse:collapse;font-size:12px;}
  th{position:sticky;top:0;background:#070a10;color:var(--mute);
    text-align:left;padding:10px 13px;font-weight:700;font-size:9.5px;
    text-transform:uppercase;letter-spacing:1.5px;
    border-bottom:1px solid var(--line2);white-space:nowrap;}
  th.sortable{cursor:pointer;} th.sortable:hover{color:var(--hv);}
  td{padding:9px 13px;border-bottom:1px solid #0f1622;white-space:nowrap;
    color:#cdd9e8;}
  tbody tr{cursor:pointer;transition:background .1s;}
  tbody tr:hover{background:#0e1828;}
  tbody tr.sel{background:#1a1405;box-shadow:inset 2px 0 0 var(--hv);}
  .pill{font-size:9.5px;font-weight:800;padding:3px 9px;letter-spacing:1px;
    color:var(--void);clip-path:polygon(4px 0,100% 0,calc(100% - 4px) 100%,
      0 100%);}

  /* target dossier */
  .detail .ttl{font-size:11px;letter-spacing:2.5px;text-transform:uppercase;
    color:var(--dim);margin-bottom:4px;}
  .detail .big{font-size:54px;font-weight:800;line-height:1;margin:6px 0 4px;
    font-family:ui-monospace,"SF Mono",Menlo,monospace;letter-spacing:-2px;}
  .detail .tierline{display:flex;align-items:center;gap:10px;margin-bottom:18px;}
  .detail .meta{color:var(--dim);font-size:12px;margin:14px 0 18px;
    line-height:1.9;border-top:1px solid var(--line);
    border-bottom:1px solid var(--line);padding:12px 0;}
  .detail .meta b{color:var(--ink);}
  .drv{margin:11px 0;}
  .drv .dl{display:flex;justify-content:space-between;font-size:11.5px;
    margin-bottom:5px;}
  .drv .dl .nm{color:#d6e1ef;} .drv .dl .cv{color:var(--mute);
    font-family:ui-monospace,Menlo,monospace;}
  .bar{height:7px;background:#0e1622;overflow:hidden;border:1px solid #142033;}
  .bar>i{display:block;height:100%;}
  .callout{margin-top:22px;border:1px solid var(--hv);
    background:linear-gradient(135deg,#1c1505,#0a0d12);padding:16px;
    position:relative;
    clip-path:polygon(0 0,100% 0,100% calc(100% - 11px),
      calc(100% - 11px) 100%,0 100%);}
  .callout .lab{font-size:9.5px;text-transform:uppercase;letter-spacing:3px;
    color:var(--hv);}
  .callout .act{font-size:25px;font-weight:800;color:var(--hv);margin-top:7px;
    letter-spacing:1px;text-shadow:0 0 20px rgba(255,176,31,.45);}
  .empty{color:var(--mute);font-size:12px;padding:46px 8px;text-align:center;
    line-height:1.9;letter-spacing:.5px;}
  .empty .ic{font-size:30px;color:var(--line2);display:block;margin-bottom:14px;}

  footer.app{padding:14px 26px;border-top:1px solid var(--line);
    color:var(--mute);font-size:10.5px;line-height:1.7;letter-spacing:.4px;}
  footer.app b{color:var(--dim);}
  /* ---- citizen-corroboration channel (cyan instrument sub-panel) ---- */
  .fieldhdr .tools{display:flex;align-items:center;gap:9px;}
  button.rtoggle{background:var(--panel);color:var(--cool);
    border:1px solid #1d3b44;font-size:9.5px;font-weight:800;
    letter-spacing:1.5px;padding:6px 11px;cursor:pointer;
    text-transform:uppercase;transition:all .14s;
    clip-path:polygon(6px 0,100% 0,calc(100% - 6px) 100%,0 100%);}
  button.rtoggle:hover{border-color:var(--cool);background:#08151a;}
  button.rtoggle.on{background:linear-gradient(95deg,var(--cool),#15a9c4);
    color:var(--void);border-color:var(--cool);
    box-shadow:0 0 14px rgba(54,224,255,.4);}
  .mapwrap.reporting{outline:1px dashed rgba(54,224,255,.5);
    outline-offset:3px;cursor:crosshair;}
  .mapwrap.reporting svg.map{cursor:crosshair;}
  .reportbox{margin-top:14px;border:1px solid #1d3b44;background:var(--void2);
    position:relative;display:none;
    clip-path:polygon(0 0,100% 0,100% calc(100% - 10px),
      calc(100% - 10px) 100%,0 100%);}
  .reportbox.show{display:block;animation:rise .4s ease-out both;}
  .reportbox .rhdr{display:flex;justify-content:space-between;
    align-items:center;padding:11px 15px;border-bottom:1px solid #14242b;
    background:linear-gradient(180deg,#08161b,#060b0e);}
  .reportbox .rhdr strong{font-size:10.5px;letter-spacing:2.5px;
    text-transform:uppercase;color:var(--cool);}
  .reportbox .rhdr .rx{color:var(--mute);font-size:9.5px;letter-spacing:1px;
    font-family:ui-monospace,Menlo,monospace;}
  .rgrid{padding:15px;display:grid;grid-template-columns:1fr 1fr;gap:14px 18px;}
  .rgrid .full{grid-column:1/3;}
  .rlab{font-size:9px;text-transform:uppercase;letter-spacing:2px;
    color:var(--mute);margin-bottom:7px;display:block;}
  .rconds{display:flex;flex-wrap:wrap;gap:7px;}
  .rcond{font-size:10.5px;letter-spacing:.5px;padding:6px 10px;
    border:1px solid var(--line2);background:var(--panel);cursor:pointer;
    user-select:none;color:var(--dim);transition:all .12s;
    display:flex;align-items:center;gap:7px;}
  .rcond input{accent-color:var(--cool);margin:0;cursor:pointer;}
  .rcond.on{border-color:var(--cool);color:var(--ink);background:#08151a;}
  .reportbox input[type=text],.reportbox textarea,.reportbox select{
    width:100%;background:var(--panel);color:var(--ink);
    border:1px solid var(--line2);padding:8px 10px;font-size:12px;
    letter-spacing:.4px;font-family:inherit;}
  .reportbox textarea{resize:vertical;min-height:52px;}
  .reportbox input[type=text]:focus,.reportbox textarea:focus,
  .reportbox select:focus{outline:1px solid var(--cool);}
  .sevseg{display:flex;gap:0;border:1px solid var(--line2);}
  .sevseg button{flex:1;background:var(--panel);color:var(--dim);
    border:none;border-right:1px solid var(--line2);padding:8px 0;
    font-size:10.5px;font-weight:700;letter-spacing:1px;cursor:pointer;
    text-transform:uppercase;transition:all .12s;}
  .sevseg button:last-child{border-right:none;}
  .sevseg button.on{background:var(--cool);color:var(--void);}
  .rcap{font-size:10.5px;color:var(--cool);letter-spacing:.5px;
    font-family:ui-monospace,Menlo,monospace;}
  .rcap .un{color:var(--mute);}
  .ractions{display:flex;gap:10px;align-items:center;
    padding:13px 15px;border-top:1px solid #14242b;flex-wrap:wrap;}
  button.rsub{background:linear-gradient(95deg,var(--cool),#15a9c4);
    color:var(--void);border:none;padding:9px 16px;font-size:11px;
    font-weight:800;cursor:pointer;letter-spacing:1.5px;
    box-shadow:0 0 14px rgba(54,224,255,.3);
    clip-path:polygon(6px 0,100% 0,calc(100% - 6px) 100%,0 100%);}
  button.rsub:hover{filter:brightness(1.12);}
  button.rghost{background:var(--panel);color:var(--dim);
    border:1px solid var(--line2);padding:9px 14px;font-size:10.5px;
    font-weight:700;cursor:pointer;letter-spacing:1px;
    text-transform:uppercase;}
  button.rghost:hover{border-color:var(--cool);color:var(--cool);}
  .rmsg{font-size:10.5px;letter-spacing:.5px;color:var(--cool);
    margin-left:auto;font-family:ui-monospace,Menlo,monospace;min-height:13px;}
  .rnote{padding:11px 15px;border-top:1px solid #14242b;
    font-size:10px;line-height:1.7;color:var(--mute);letter-spacing:.3px;
    background:#060b0e;}
  .rnote b{color:var(--dim);}
  .rnote code{font-family:ui-monospace,Menlo,monospace;color:var(--cool);
    background:#0a1417;padding:1px 5px;}
  /* dossier field-reports block */
  .freports{margin-top:22px;border:1px solid #1d3b44;
    background:linear-gradient(135deg,#08161b,#0a0d12);padding:16px;
    position:relative;
    clip-path:polygon(0 0,100% 0,100% calc(100% - 11px),
      calc(100% - 11px) 100%,0 100%);}
  .freports .lab{font-size:9.5px;text-transform:uppercase;letter-spacing:3px;
    color:var(--cool);}
  .freports .agree{font-size:12px;color:var(--ink);margin:9px 0 14px;
    line-height:1.6;letter-spacing:.3px;}
  .freports .agree b{color:var(--cool);}
  .freports .agree.un b{color:var(--mute);}
  .frow{border-top:1px solid #14242b;padding:10px 0 2px;font-size:11.5px;
    line-height:1.6;}
  .frow:first-of-type{border-top:none;}
  .frow .ft{display:flex;align-items:center;gap:9px;margin-bottom:4px;
    flex-wrap:wrap;}
  .fsev{font-size:9px;font-weight:800;padding:2px 8px;letter-spacing:1px;
    color:var(--void);clip-path:polygon(4px 0,100% 0,calc(100% - 4px) 100%,
      0 100%);}
  .fsrc{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
    color:var(--mute);font-family:ui-monospace,Menlo,monospace;}
  .fst{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
    font-family:ui-monospace,Menlo,monospace;}
  .fst.v{color:var(--cool);} .fst.p{color:var(--elev);}
  .frow .fc{color:#b9c7d8;} .frow .fn{color:var(--mute);font-size:11px;
    margin-top:3px;font-style:italic;}
  .legend i.diamond{border-radius:0;transform:rotate(45deg);
    width:9px;height:9px;background:none;border:1.5px solid var(--cool);
    box-shadow:none;}
  @media (max-width:1140px){.layout{grid-template-columns:1fr;}
    .panel,.panel.right{border:none;border-bottom:1px solid var(--line);
      max-height:none;}
    .rgrid{grid-template-columns:1fr;}}
</style>
</head>
<body>
<header class="app">
  <div class="brand">
    <span class="glyph"></span>
    <div>
      <h1>V I S T A</h1>
      <span class="tag" id="subtitle"></span>
    </div>
  </div>
  <div class="status">
    <span class="lamp"><i></i> Live · Deterministic</span>
    <span class="badge">DTE ENERGY · HACK MICHIGAN 2026</span>
  </div>
</header>

<div class="kpis" id="kpis"></div>

<div class="layout">
  <aside class="panel">
    <div class="ctl">
      <h2 class="sec">Risk tier</h2>
      <div class="chips" id="tierChips"></div>
    </div>
    <div class="ctl">
      <h2 class="sec">County</h2>
      <select id="countySel"></select>
    </div>
    <div class="ctl">
      <h2 class="sec">Minimum risk</h2>
      <input type="range" id="minRisk" min="0" max="100" value="0" step="1">
      <div class="val mono" id="minRiskVal">&ge; 0%</div>
    </div>
    <div class="ctl">
      <h2 class="sec">Inspection budget</h2>
      <input type="range" id="budget" min="1" max="100" value="20" step="1">
      <div class="val mono" id="budgetVal"></div>
      <div class="hint">Calibrate the crew's truck-roll budget. The worklist
        and the map priority reticles track the top-N by predicted risk.</div>
      <div class="gauge" id="coverage"></div>
    </div>
  </aside>

  <main class="center">
    <div class="mapwrap" id="mapwrap">
      <div class="fieldhdr">
        <span>FIG · DTE SE-MICHIGAN TERRITORY — POLE RISK FIELD</span>
        <span class="tools">
          <button class="rtoggle on" id="fieldReportsToggle"
            title="Show/hide moderated field-report markers">&#9826; FIELD
            REPORTS</button>
          <button class="rtoggle" id="reportToggle"
            title="Enter report mode, then click the map or a node">&#xFF0B;
            REPORT A POLE</button>
          <span class="rt mono">WGS84 · LIVE</span>
        </span>
      </div>
      <svg class="map" id="map" viewBox="0 0 1000 612"
           preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend">
        <span><i style="color:#ff394e;background:#ff394e"></i>Critical &ge;65%</span>
        <span><i style="color:#ff8f1f;background:#ff8f1f"></i>High &ge;40%</span>
        <span><i style="color:#ffd23f;background:#ffd23f"></i>Elevated &ge;22%</span>
        <span><i style="color:#1fd98c;background:#1fd98c"></i>Routine</span>
        <span><i style="color:#36e0ff;background:none;border:1px solid #36e0ff;
          box-shadow:none"></i>NOAA station</span>
        <span><i style="background:none;border:1px solid #fff;box-shadow:none;
          border-radius:0"></i>Priority (within budget)</span>
        <span><i class="diamond" style="color:#36e0ff"></i>Field reports
          (moderated / yours)</span>
      </div>

      <div class="reportbox" id="reportForm">
        <div class="rhdr">
          <strong>&#xFF0B; Report a pole — community field report</strong>
          <span class="rx" id="reportCap">No location — click the map or a
            node</span>
        </div>
        <div class="rgrid">
          <div>
            <span class="rlab">Location</span>
            <div class="rcap" id="reportLoc"><span class="un">awaiting map
              click…</span></div>
          </div>
          <div>
            <span class="rlab">Pole ID (auto / manual)</span>
            <input type="text" id="reportPoleId" placeholder="e.g. P00123 or
              blank if unmapped" autocomplete="off">
          </div>
          <div class="full">
            <span class="rlab">Observed conditions</span>
            <div class="rconds" id="reportConds"></div>
          </div>
          <div>
            <span class="rlab">Severity</span>
            <div class="sevseg" id="reportSev">
              <button type="button" data-s="low">Low</button>
              <button type="button" data-s="medium" class="on">Medium</button>
              <button type="button" data-s="urgent">Urgent</button>
            </div>
          </div>
          <div>
            <span class="rlab">Reporter handle (optional, no PII)</span>
            <input type="text" id="reportReporter" placeholder="optional"
              autocomplete="off" maxlength="40">
          </div>
          <div class="full">
            <span class="rlab">Short note</span>
            <textarea id="reportNote" maxlength="280"
              placeholder="What did you observe? (no personal info)"></textarea>
          </div>
        </div>
        <div class="ractions">
          <button class="rsub" id="reportSubmit">&#9826; SUBMIT REPORT</button>
          <button class="rghost" id="reportDownload">&#x2913; Download
            community reports (.jsonl)</button>
          <button class="rghost" id="reportClear">Clear my reports</button>
          <span class="rmsg" id="reportMsg"></span>
        </div>
        <div class="rnote" id="reportFlow">
          <b>Moderated flow.</b> Submitting stores the report locally in this
          browser only and draws it as a cyan diamond on the field. Use
          <b>Download community reports (.jsonl)</b>, then submit the
          downloaded file as a pull request adding lines to
          <code>community_reports/ledger.jsonl</code>; a maintainer
          reviews/verifies before it appears as corroboration. Reports are a
          corroboration/priority overlay — never model training input.
        </div>
      </div>
    </div>

    <div class="tablewrap">
      <div class="tablebar">
        <div><strong>Prioritized Worklist</strong>
          <span class="count" id="wcount"></span></div>
        <button class="exp" id="exportBtn">&#x2913; EXPORT INSPECTION PLAN</button>
      </div>
      <div class="tscroll">
        <table>
          <thead><tr>
            <th>#</th><th>Pole</th>
            <th class="sortable" data-k="county">County &#9662;</th>
            <th>Segment</th>
            <th class="sortable" data-k="risk">Risk &#9662;</th>
            <th>Tier</th><th>Top driver</th><th>Action</th>
          </tr></thead>
          <tbody id="wbody"></tbody>
        </table>
      </div>
    </div>
  </main>

  <aside class="panel right" id="detailPane">
    <div class="empty" id="detailEmpty">
      <span class="ic">&#9678;</span>
      Select a node on the field<br>or a worklist row to lock a target
      and read <strong>why</strong> it's flagged.
    </div>
    <div class="detail" id="detailBody" style="display:none"></div>
  </aside>
</div>

<footer class="app" id="footer"><span id="footerProv"></span>
  <span id="commStat"></span></footer>
<div id="tooltip"></div>

<script id="vista-data" type="application/json">__VISTA_DATA__</script>
<script>
"use strict";
(function(){
  var D = JSON.parse(document.getElementById("vista-data").textContent);
  var M = D.meta, K = D.kpis, POLES = D.poles;
  // Citizen-corroboration overlay (read-only; never a model input).
  var COMM = D.community || {reports:[],summary:{n_reports:0,
    n_poles_corroborated:0,n_verified:0,n_pending:0}};
  var COMM_REPORTS = COMM.reports || [];
  var COMM_SUM = COMM.summary || {};
  // ingested reports indexed by pole id (for the dossier block)
  var COMM_BY_POLE = {};
  COMM_REPORTS.forEach(function(r){
    if(r.pole_id==null) return;
    (COMM_BY_POLE[r.pole_id]=COMM_BY_POLE[r.pole_id]||[]).push(r); });
  var CY = "#36e0ff";
  var TIER_COLOR = {CRITICAL:"#ff394e",HIGH:"#ff8f1f",
                    ELEVATED:"#ffd23f",ROUTINE:"#1fd98c"};
  var TIERS = ["CRITICAL","HIGH","ELEVATED","ROUTINE"];
  var CONDITIONS = ["Leaning pole","Vegetation contact",
    "Damaged hardware/crossarm","Low/down wire","Cracked/rotted pole",
    "Other"];
  var LS_KEY = "vista_community_reports";

  var state = {
    tiers:{CRITICAL:true,HIGH:true,ELEVATED:true,ROUTINE:true},
    county:"ALL", minRisk:0, budget:20,
    sortKey:"risk", sortDir:-1, selected:null,
    reporting:false, showFieldReports:true,
    rLoc:null, rConds:{}, rSev:"medium"
  };

  function esc(s){ return String(s).replace(/[&<>"]/g,function(c){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]; }); }
  function pct(x){ return (x*100).toFixed(1)+"%"; }
  function pct0(x){ return Math.round(x*100)+"%"; }
  function usd(x){ return "$"+Math.round(x).toLocaleString("en-US"); }

  document.getElementById("subtitle").textContent = M.subtitle;
  document.getElementById("footerProv").innerHTML =
    "<b>Provenance.</b> Synthetic DTE-style fleet anchored to REAL NOAA "+
    "1991–2020 climate normals ("+M.noaa_stations.length+" stations) + "+
    "image-derived NDVI / structure features. Deterministic, offline, no "+
    "proprietary inputs. Generated "+esc(M.generated)+" · "+
    M.n_poles.toLocaleString()+" poles · tiers CRITICAL&ge;0.65 / "+
    "HIGH&ge;0.40 / ELEVATED&ge;0.22.<br>";

  // ---- KPI telemetry rail ----
  var inc=K.incumbent, abl=K.ablation, ec=K.economics;
  var kpiHtml = [
    ["Held-out ROC-AUC", K.roc_auc.toFixed(3), "", ""],
    ["Lift vs age-cycle", inc.lift_vs_age_cycle_x.toFixed(2)+"×", "", ""],
    ["Net benefit / cycle", usd(ec.net_benefit_fleet_usd), "", ""],
    ["Capture @20% budget", pct0(K.capture_at_20), "", ""],
    ["Imagery payoff",
      "Imagery adds +"+abl.auc_gain_from_imagery.toFixed(3)+" ROC-AUC",
      "headline", "sm"]
  ].map(function(k){
    return '<div class="kpi '+k[2]+'"><div class="label">'+esc(k[0])+
      '</div><div class="value '+k[3]+'">'+esc(k[1])+'</div></div>';
  }).join("");
  document.getElementById("kpis").innerHTML = kpiHtml;

  // ---- tier breakers (with live counts) ----
  var tc = document.getElementById("tierChips");
  function tierCount(t){ var n=0; for(var i=0;i<POLES.length;i++)
    if(POLES[i].tier===t) n++; return n; }
  tc.innerHTML = TIERS.map(function(t){
    return '<div class="chip" data-t="'+t+'"><span class="dot" style="'+
      'background:'+TIER_COLOR[t]+';color:'+TIER_COLOR[t]+'"></span>'+t+
      '<span class="ct">'+tierCount(t)+'</span></div>';
  }).join("");
  Array.prototype.forEach.call(tc.querySelectorAll(".chip"),function(el){
    el.addEventListener("click",function(){
      var t=el.getAttribute("data-t");
      state.tiers[t]=!state.tiers[t];
      el.classList.toggle("off",!state.tiers[t]); render();
    });
  });

  // ---- county ----
  var cs = document.getElementById("countySel");
  cs.innerHTML = '<option value="ALL">All counties</option>' +
    M.counties.slice().sort().map(function(c){
      return '<option value="'+esc(c)+'">'+esc(c)+'</option>'; }).join("");
  cs.addEventListener("change",function(){ state.county=cs.value; render(); });

  // ---- sliders ----
  var mr=document.getElementById("minRisk"),
      mv=document.getElementById("minRiskVal");
  mr.addEventListener("input",function(){
    state.minRisk=+mr.value; mv.innerHTML="≥ "+mr.value+"%"; render(); });
  var bg=document.getElementById("budget");
  bg.addEventListener("input",function(){ state.budget=+bg.value; render(); });

  // ---- projection (real WGS84 -> SVG) ----
  var BB=M.territory_bbox, VW=1000, VH=612,
      PADX=48, PADY=34, IW=VW-2*PADX, IH=VH-2*PADY;
  function projX(lon){
    return PADX + (lon-BB.lon_min)/(BB.lon_max-BB.lon_min)*IW; }
  function projY(lat){
    return PADY + (1-(lat-BB.lat_min)/(BB.lat_max-BB.lat_min))*IH; }

  var svg=document.getElementById("map");
  var SVGNS="http://www.w3.org/2000/svg";
  function el(tag,attrs){
    var e=document.createElementNS(SVGNS,tag);
    for(var k in attrs) e.setAttribute(k,attrs[k]); return e; }

  function filtered(){
    return POLES.filter(function(p){
      if(!state.tiers[p.tier]) return false;
      if(state.county!=="ALL" && p.county!==state.county) return false;
      if(p.risk*100 < state.minRisk) return false;
      return true;
    });
  }
  function budgetSet(){
    var n=Math.max(1,Math.round(POLES.length*state.budget/100));
    var s={}; for(var i=0;i<n && i<POLES.length;i++) s[POLES[i].id]=true;
    return {set:s,n:n};
  }

  var tip=document.getElementById("tooltip");
  function showTip(e,p){
    tip.style.display="block";
    tip.style.left=(e.clientX+15)+"px";
    tip.style.top=(e.clientY+15)+"px";
    tip.innerHTML="<b>"+esc(p.id)+"</b> &middot; "+esc(p.county)+
      "<br>RISK "+pct(p.risk)+" &middot; "+p.tier;
  }
  function hideTip(){ tip.style.display="none"; }

  function drawMap(){
    while(svg.firstChild) svg.removeChild(svg.firstChild);
    svg.appendChild(el("rect",{x:0,y:0,width:VW,height:VH,fill:"#05080c"}));
    // field grid
    var gx=10, gy=7;
    for(var i=0;i<=gx;i++){ var x=PADX+i*IW/gx;
      svg.appendChild(el("line",{x1:x,y1:PADY,x2:x,y2:VH-PADY,
        stroke:"#11203a","stroke-width":i%5===0?0.9:0.5})); }
    for(i=0;i<=gy;i++){ var y=PADY+i*IH/gy;
      svg.appendChild(el("line",{x1:PADX,y1:y,x2:VW-PADX,y2:y,
        stroke:"#11203a","stroke-width":i%5===0?0.9:0.5})); }
    // county dividers + labels
    var nC=M.counties.length;
    for(i=0;i<nC;i++){
      var cl=BB.lon_min+(i+0.5)*(BB.lon_max-BB.lon_min)/nC;
      var t=el("text",{x:projX(cl),y:PADY-11,fill:"#46587a",
        "font-size":10,"text-anchor":"middle",
        "font-family":"ui-monospace,Menlo,monospace",
        "letter-spacing":"1"});
      t.textContent=M.counties[i].toUpperCase(); svg.appendChild(t);
    }
    // bracket frame
    svg.appendChild(el("rect",{x:PADX,y:PADY,width:IW,height:IH,
      fill:"none",stroke:"#1d2c44","stroke-width":1}));
    [[PADX,PADY,1,1],[VW-PADX,PADY,-1,1],[PADX,VH-PADY,1,-1],
     [VW-PADX,VH-PADY,-1,-1]].forEach(function(c){
      svg.appendChild(el("path",{d:"M "+(c[0])+" "+(c[1]+c[3]*22)+" L "+
        c[0]+" "+c[1]+" L "+(c[0]+c[2]*22)+" "+c[1],
        stroke:"#ffb01f","stroke-width":2,fill:"none"}));
    });

    var fset = budgetSet().set, show = filtered(), showIds={};
    show.forEach(function(p){ showIds[p.id]=true; });

    // dimmed off-filter context first (keeps the territory readable)
    POLES.forEach(function(p){
      if(showIds[p.id]) return;
      svg.appendChild(el("circle",{cx:projX(p.lon).toFixed(1),
        cy:projY(p.lat).toFixed(1),r:1.7,fill:TIER_COLOR[p.tier],
        "fill-opacity":0.10}));
    });
    // live nodes — layered halo gives the high-voltage bloom on the
    // dangerous tiers without per-node SVG filters (smooth at fleet scale)
    show.forEach(function(p){
      var x=projX(p.lon).toFixed(1), y=projY(p.lat).toFixed(1);
      var r = p.tier==="CRITICAL"?4.6 : p.tier==="HIGH"?3.9 :
              p.tier==="ELEVATED"?3.1:2.6;
      if(p.tier==="CRITICAL"||p.tier==="HIGH"){
        svg.appendChild(el("circle",{cx:x,cy:y,r:r*2.8,
          fill:TIER_COLOR[p.tier],"fill-opacity":0.12}));
        svg.appendChild(el("circle",{cx:x,cy:y,r:r*1.7,
          fill:TIER_COLOR[p.tier],"fill-opacity":0.22}));
      }
      var c=el("circle",{cx:x,cy:y,r:r,fill:TIER_COLOR[p.tier],
        "fill-opacity":0.97,style:"cursor:pointer"});
      c.addEventListener("mousemove",function(e){ showTip(e,p); });
      c.addEventListener("mouseleave",hideTip);
      c.addEventListener("click",function(){ select(p.id); });
      svg.appendChild(c);
    });
    // priority reticles (within budget AND visible)
    show.forEach(function(p){
      if(!fset[p.id]) return;
      var x=+projX(p.lon).toFixed(1), y=+projY(p.lat).toFixed(1);
      svg.appendChild(el("rect",{x:x-7,y:y-7,width:14,height:14,fill:"none",
        stroke:"#ffffff","stroke-width":1,"stroke-opacity":0.5}));
    });
    // selected: locked-target reticle
    if(state.selected){
      var sp=null;
      for(var k=0;k<POLES.length;k++) if(POLES[k].id===state.selected){
        sp=POLES[k]; break; }
      if(sp && showIds[sp.id]){
        var sx=+projX(sp.lon).toFixed(1), sy=+projY(sp.lat).toFixed(1);
        var g=el("g",{});
        g.appendChild(el("circle",{cx:sx,cy:sy,r:13,fill:"none",
          stroke:"#ffb01f","stroke-width":1.4}));
        g.appendChild(el("line",{x1:sx-20,y1:sy,x2:sx-9,y2:sy,
          stroke:"#ffb01f","stroke-width":1.5}));
        g.appendChild(el("line",{x1:sx+9,y1:sy,x2:sx+20,y2:sy,
          stroke:"#ffb01f","stroke-width":1.5}));
        g.appendChild(el("line",{x1:sx,y1:sy-20,x2:sx,y2:sy-9,
          stroke:"#ffb01f","stroke-width":1.5}));
        g.appendChild(el("line",{x1:sx,y1:sy+9,x2:sx,y2:sy+20,
          stroke:"#ffb01f","stroke-width":1.5}));
        svg.appendChild(g);
      }
    }
    // NOAA stations
    M.noaa_stations.forEach(function(s){
      var x=projX(s.lon),y=projY(s.lat);
      svg.appendChild(el("polygon",{
        points:(x)+","+(y-6)+" "+(x+5.5)+","+(y+4.5)+" "+(x-5.5)+","+(y+4.5),
        fill:"none",stroke:"#36e0ff","stroke-width":1.2}));
    });
    // citizen-corroboration overlay markers (moderated ledger + my local)
    drawCommunity();
    // one-time power-on energize sweep
    var sweep=el("rect",{x:PADX,y:PADY,width:3,height:IH,
      fill:"#ffb01f","fill-opacity":0.55});
    var an=el("animate",{attributeName:"x",from:PADX,to:VW-PADX,
      dur:"1.1s",begin:"0.45s",fill:"freeze"});
    var an2=el("animate",{attributeName:"fill-opacity",from:"0.55",to:"0",
      dur:"1.1s",begin:"0.45s",fill:"freeze"});
    sweep.appendChild(an); sweep.appendChild(an2); svg.appendChild(sweep);
  }

  // ---- target dossier ----
  function select(id){
    state.selected=id;
    var p=null;
    for(var i=0;i<POLES.length;i++) if(POLES[i].id===id){ p=POLES[i]; break; }
    var emp=document.getElementById("detailEmpty"),
        bd=document.getElementById("detailBody");
    if(!p){ emp.style.display="block"; bd.style.display="none";
      drawMap(); drawTable(); return; }
    emp.style.display="none"; bd.style.display="block";
    var maxAbs=0;
    p.drivers.forEach(function(d){ maxAbs=Math.max(maxAbs,Math.abs(d[1])); });
    maxAbs=maxAbs||1;
    var drvHtml=p.drivers.map(function(d){
      var raises=d[1]>=0, w=Math.max(5,Math.abs(d[1])/maxAbs*100);
      return '<div class="drv"><div class="dl"><span class="nm">'+
        esc(d[0])+'</span><span class="cv">'+
        (raises?"+":"")+d[1].toFixed(4)+'</span></div>'+
        '<div class="bar"><i style="width:'+w.toFixed(1)+
        '%;background:'+(raises?"#ff394e":"#1fd98c")+
        ';box-shadow:0 0 8px '+(raises?"#ff394e":"#1fd98c")+
        '"></i></div></div>';
    }).join("");
    bd.innerHTML =
      '<div class="ttl">Target locked &mdash; Pole '+esc(p.id)+'</div>'+
      '<div class="big" style="color:'+TIER_COLOR[p.tier]+
      ';text-shadow:0 0 30px '+TIER_COLOR[p.tier]+'66">'+pct(p.risk)+'</div>'+
      '<div class="tierline"><span class="pill" style="background:'+
      TIER_COLOR[p.tier]+'">'+p.tier+'</span>'+
      '<span style="color:var(--mute);font-size:11px;letter-spacing:2px">'+
      'CALIBRATED FAILURE RISK</span></div>'+
      '<div class="meta">COUNTY <b>'+esc(p.county)+'</b> &nbsp;|&nbsp; '+
      'SEGMENT <b>'+esc(p.segment)+'</b><br>POSITION <b>'+p.lat.toFixed(4)+
      ', '+p.lon.toFixed(4)+'</b> (WGS84)</div>'+
      '<div class="ttl">Why this pole is flagged</div>'+drvHtml+
      fieldReportsBlock(p)+
      '<div class="callout"><div class="lab">Recommended directive</div>'+
      '<div class="act">&#9656; '+esc(p.action)+'</div></div>';
    drawMap(); drawTable();
  }

  // dossier FIELD REPORTS block — reconciles the model output against the
  // citizen-corroboration overlay (ingested moderated ledger + my local).
  // Pure presentation; does NOT alter risk/tier/drivers.
  function fieldReportsBlock(p){
    var ing = COMM_BY_POLE[p.id] || [];
    var mine = loadMine().filter(function(r){ return r.pole_id===p.id; });
    var all = ing.concat(mine);
    if(!all.length) return "";
    var nVer = ing.filter(function(r){ return r.status==="verified"; })
      .length;
    var mt = p.tier;  // model verdict label (unchanged by the overlay)
    var agree, un=false;
    if(nVer>0){
      agree = "Model: <b>"+mt+"</b> &mdash; corroborated by <b>"+
        nVer+"</b> verified field report"+(nVer===1?"":"s");
    } else if(ing.length>0){
      un=true;
      agree = "Model: <b>"+mt+"</b> &mdash; <b>"+ing.length+
        "</b> pending field report"+(ing.length===1?"":"s")+
        " (awaiting moderation, not yet corroboration)";
    } else {
      un=true;
      agree = "Model: <b>"+mt+"</b> &mdash; <b>uncorroborated</b> "+
        "(local report only, not submitted)";
    }
    var rows = all.map(function(r){
      var local = mine.indexOf(r)>=0;
      var sv = r.severity||"medium";
      var sc = sv==="urgent"?"#ff394e":sv==="medium"?"#ffd23f":"#1fd98c";
      var stCls = r.status==="verified"?"v":"p";
      var stTx = local? "LOCAL/PENDING" : r.status.toUpperCase();
      return '<div class="frow"><div class="ft">'+
        '<span class="fsev" style="background:'+sc+'">'+esc(sv)+
        '</span>'+
        '<span class="fst '+stCls+'">'+esc(stTx)+'</span>'+
        '<span class="fsrc">'+esc(r.source||"resident")+'</span></div>'+
        '<div class="fc">'+esc((r.conditions||[]).join(" · "))+'</div>'+
        (r.note? '<div class="fn">"'+esc(r.note)+'"</div>':"")+'</div>';
    }).join("");
    return '<div class="freports"><div class="lab">&#9826; FIELD REPORTS ('+
      all.length+')</div>'+
      '<div class="agree'+(un?" un":"")+'">'+agree+'</div>'+rows+'</div>';
  }

  // ---- worklist ----
  function sortRows(rows){
    var k=state.sortKey,d=state.sortDir;
    return rows.slice().sort(function(a,b){
      var av=a[k],bv=b[k];
      if(k==="county"){ return d*String(av).localeCompare(String(bv)); }
      return d*(av-bv);
    });
  }
  function drawTable(){
    var rows=sortRows(filtered());
    var tb=document.getElementById("wbody");
    document.getElementById("wcount").textContent=
      "· "+rows.length+" / "+POLES.length+" poles";
    document.getElementById("wcount").setAttribute("data-n",rows.length);
    if(!rows.length){
      tb.innerHTML='<tr><td colspan="8" class="empty">'+
        'No poles match the current filters.</td></tr>'; return; }
    var html="";
    for(var i=0;i<rows.length;i++){
      var p=rows[i], sel=(state.selected===p.id)?" sel":"";
      html+='<tr class="'+sel.trim()+'" data-id="'+esc(p.id)+'">'+
        '<td class="mono" style="color:#54637a">'+(i+1)+'</td>'+
        '<td class="mono">'+esc(p.id)+'</td>'+
        '<td>'+esc(p.county)+'</td>'+
        '<td class="mono">'+esc(p.segment)+'</td>'+
        '<td class="mono" style="color:'+TIER_COLOR[p.tier]+
        ';font-weight:700">'+pct(p.risk)+'</td>'+
        '<td><span class="pill" style="background:'+TIER_COLOR[p.tier]+
        '">'+p.tier+'</span></td>'+
        '<td style="color:#9fb0c6">'+
        esc(p.drivers.length?p.drivers[0][0]:"")+'</td>'+
        '<td><b style="color:var(--hv)">'+esc(p.action)+'</b></td></tr>';
    }
    tb.innerHTML=html;
    Array.prototype.forEach.call(tb.querySelectorAll("tr[data-id]"),
      function(tr){ tr.addEventListener("click",function(){
        select(tr.getAttribute("data-id")); }); });
  }
  Array.prototype.forEach.call(
    document.querySelectorAll("th.sortable"),function(th){
    th.addEventListener("click",function(){
      var k=th.getAttribute("data-k");
      if(state.sortKey===k){ state.sortDir*=-1; }
      else { state.sortKey=k; state.sortDir=(k==="risk")?-1:1; }
      drawTable();
    });
  });

  // ---- budget coverage gauge ----
  function drawCoverage(){
    var b=budgetSet(), n=b.n, m=POLES.length;
    var predFail=POLES.filter(function(p){ return p.risk>=0.40; }).length;
    var caught=0;
    for(var i=0;i<n && i<m;i++) if(POLES[i].risk>=0.40) caught++;
    var covPct = predFail? Math.round(caught/predFail*100):0;
    var reactivePct = Math.round(n/m*100);
    document.getElementById("budgetVal").innerHTML=
      "TOP <b style='color:var(--hv)'>"+n+"</b> / "+m+" poles &nbsp;("+
      reactivePct+"% of fleet)";
    document.getElementById("coverage").innerHTML=
      '<div class="big">Inspecting <b>'+n+'</b> poles, VISTA catches <b>'+
      covPct+'%</b> of the '+predFail+' predicted failures.</div>'+
      '<div class="react">Reactive / age-blind crew, same '+n+
      ' poles &asymp; '+reactivePct+'%.</div>';
  }

  // ---- CSV export (client-side Blob, offline) ----
  document.getElementById("exportBtn").addEventListener("click",function(){
    var rows=sortRows(filtered());
    var head=["rank","pole_id","county","segment","risk","tier",
              "top_driver","action"];
    var lines=[head.join(",")];
    rows.forEach(function(p,i){
      function q(v){ v=String(v); return /[",\n]/.test(v)?
        '"'+v.replace(/"/g,'""')+'"':v; }
      lines.push([i+1,p.id,q(p.county),p.segment,p.risk.toFixed(4),
        p.tier,q(p.drivers.length?p.drivers[0][0]:""),q(p.action)
      ].join(","));
    });
    var blob=new Blob([lines.join("\n")],{type:"text/csv"});
    var url=URL.createObjectURL(blob);
    var a=document.createElement("a");
    a.href=url; a.download="vista_inspection_plan.csv";
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
  });

  // ====================================================================
  // CITIZEN-CORROBORATION CHANNEL — in-app report mode (offline, vanilla).
  // Reports live in localStorage only; export emits the exact ledger
  // schema for the moderated GitHub PR flow. This is an overlay /
  // corroboration signal — it never touches the model or risk.
  // ====================================================================

  // inverse of projX/projY: SVG client point -> WGS84 lon/lat
  function invProj(px, py){
    var lon = BB.lon_min + (px-PADX)/IW*(BB.lon_max-BB.lon_min);
    var lat = BB.lat_min + (1-(py-PADY)/IH)*(BB.lat_max-BB.lat_min);
    return {lat:lat, lon:lon};
  }
  // nearest county by the SAME longitude banding the fleet uses
  function countyForLon(lon){
    var nC=M.counties.length;
    var f=(lon-BB.lon_min)/(BB.lon_max-BB.lon_min);
    var i=Math.floor(f*nC); if(i<0) i=0; if(i>nC-1) i=nC-1;
    return M.counties[i];
  }
  function svgPointFromEvent(e){
    var pt=svg.createSVGPoint();
    pt.x=e.clientX; pt.y=e.clientY;
    var m=svg.getScreenCTM();
    if(!m) return null;
    var p=pt.matrixTransform(m.inverse());
    return {x:p.x, y:p.y};
  }
  function nearestPole(lon, lat){
    var best=null, bd=1e9;
    for(var i=0;i<POLES.length;i++){
      var p=POLES[i];
      var dx=p.lon-lon, dy=p.lat-lat, d=dx*dx+dy*dy;
      if(d<bd){ bd=d; best=p; }
    }
    return {pole:best, d:Math.sqrt(bd)};
  }

  function loadMine(){
    try{ var v=JSON.parse(localStorage.getItem(LS_KEY)||"[]");
      return Array.isArray(v)?v:[]; }catch(e){ return []; }
  }
  function saveMine(a){
    try{ localStorage.setItem(LS_KEY,JSON.stringify(a)); }catch(e){}
  }
  function nextReportId(mine){
    // deterministic local id space, distinct from committed CR-* ledger ids
    var mx=0;
    mine.forEach(function(r){
      var m=/^LOCAL-(\d+)$/.exec(r.report_id||"");
      if(m){ var n=+m[1]; if(n>mx) mx=n; }
    });
    var s=String(mx+1); while(s.length<4) s="0"+s;
    return "LOCAL-"+s;
  }

  // condition checkboxes
  var rConds=document.getElementById("reportConds");
  rConds.innerHTML=CONDITIONS.map(function(c,i){
    return '<label class="rcond" data-c="'+esc(c)+'">'+
      '<input type="checkbox" data-ci="'+i+'">'+esc(c)+'</label>';
  }).join("");
  Array.prototype.forEach.call(rConds.querySelectorAll(".rcond"),
    function(lb){
      var cb=lb.querySelector("input");
      lb.addEventListener("click",function(e){
        if(e.target!==cb){ cb.checked=!cb.checked; }
        state.rConds[lb.getAttribute("data-c")]=cb.checked;
        lb.classList.toggle("on",cb.checked);
      });
    });

  // severity segmented control
  var rSev=document.getElementById("reportSev");
  Array.prototype.forEach.call(rSev.querySelectorAll("button"),
    function(b){
      b.addEventListener("click",function(){
        state.rSev=b.getAttribute("data-s");
        Array.prototype.forEach.call(rSev.querySelectorAll("button"),
          function(x){ x.classList.toggle("on",x===b); });
      });
    });

  function setReportLoc(loc, poleId, county){
    state.rLoc={lat:loc.lat, lon:loc.lon,
      county:county||countyForLon(loc.lon)};
    document.getElementById("reportLoc").innerHTML=
      loc.lat.toFixed(5)+", "+loc.lon.toFixed(5)+
      " &middot; <span style=\"color:var(--dim)\">"+
      esc(state.rLoc.county)+"</span>";
    document.getElementById("reportCap").textContent=
      poleId? ("Pole "+poleId+" prefilled") : "Location captured";
    if(poleId!=null){
      document.getElementById("reportPoleId").value=poleId;
    }
    drawMap();
  }

  function rMsg(t,bad){
    var m=document.getElementById("reportMsg");
    m.textContent=t||"";
    m.style.color=bad?"#ff8f1f":CY;
  }

  // toggle report mode
  var rToggle=document.getElementById("reportToggle");
  var rForm=document.getElementById("reportForm");
  var mapWrap=document.getElementById("mapwrap");
  rToggle.addEventListener("click",function(){
    state.reporting=!state.reporting;
    rToggle.classList.toggle("on",state.reporting);
    rForm.classList.toggle("show",state.reporting);
    mapWrap.classList.toggle("reporting",state.reporting);
    rMsg(state.reporting?
      "Report mode ON — click the field or a node to set a location":"");
  });
  // field-report marker visibility toggle
  var frToggle=document.getElementById("fieldReportsToggle");
  frToggle.addEventListener("click",function(){
    state.showFieldReports=!state.showFieldReports;
    frToggle.classList.toggle("on",state.showFieldReports);
    drawMap();
  });

  // map click while reporting -> capture location (svg-level handler so
  // empty field clicks also work; node clicks prefill via select()).
  svg.addEventListener("click",function(e){
    if(!state.reporting) return;
    var sp=svgPointFromEvent(e);
    if(!sp) return;
    var ll=invProj(sp.x, sp.y);
    if(ll.lon<BB.lon_min||ll.lon>BB.lon_max||
       ll.lat<BB.lat_min||ll.lat>BB.lat_max) return;
    var np=nearestPole(ll.lon, ll.lat);
    // snap to a node if the click landed essentially on it
    if(np.pole && np.d<0.012){
      setReportLoc({lat:np.pole.lat,lon:np.pole.lon}, np.pole.id,
        np.pole.county);
    } else {
      setReportLoc(ll, "", countyForLon(ll.lon));
    }
  },true);

  // submit -> append to localStorage, redraw
  document.getElementById("reportSubmit").addEventListener("click",
    function(){
      var pid=document.getElementById("reportPoleId").value.trim();
      var loc=state.rLoc;
      if(!loc && pid){
        // manual pole id with no map click: borrow that pole's location
        for(var i=0;i<POLES.length;i++) if(POLES[i].id===pid){
          loc={lat:POLES[i].lat,lon:POLES[i].lon,county:POLES[i].county};
          break; }
      }
      if(!loc){ rMsg("Set a location: click the map/a node, or enter a "+
        "known pole id",true); return; }
      var conds=CONDITIONS.filter(function(c){ return state.rConds[c]; });
      if(!conds.length){ rMsg("Select at least one condition",true);
        return; }
      var mine=loadMine();
      var rec={
        report_id: nextReportId(mine),
        pole_id: pid? pid : null,
        lat: Math.round(loc.lat*1e6)/1e6,
        lon: Math.round(loc.lon*1e6)/1e6,
        county: loc.county||countyForLon(loc.lon),
        conditions: conds,
        severity: state.rSev,
        note: document.getElementById("reportNote").value
          .trim().slice(0,280),
        reporter: document.getElementById("reportReporter").value
          .trim().slice(0,40),
        submitted: D.meta.generated,
        status: "pending",
        source: "resident"
      };
      mine.push(rec); saveMine(mine);
      // reset transient inputs
      document.getElementById("reportNote").value="";
      state.rConds={};
      Array.prototype.forEach.call(rConds.querySelectorAll(".rcond"),
        function(lb){ lb.classList.remove("on");
          lb.querySelector("input").checked=false; });
      rMsg("Saved locally as "+rec.report_id+
        " — download + open a PR to submit");
      updateCommStat(); drawMap();
      if(state.selected) select(state.selected);
    });

  // clear my reports
  document.getElementById("reportClear").addEventListener("click",
    function(){
      var mine=loadMine();
      if(!mine.length){ rMsg("No local reports to clear"); return; }
      saveMine([]);
      rMsg("Cleared "+mine.length+" local report"+
        (mine.length===1?"":"s"));
      updateCommStat(); drawMap();
      if(state.selected) select(state.selected);
    });

  // download my reports as ledger-schema JSON Lines (client-side Blob)
  document.getElementById("reportDownload").addEventListener("click",
    function(){
      var mine=loadMine();
      if(!mine.length){ rMsg("No local reports to download",true); return; }
      var lines=mine.map(function(r){
        // emit EXACTLY the ledger schema, status pending / source resident
        return JSON.stringify({
          report_id:r.report_id, pole_id:r.pole_id==null?null:r.pole_id,
          lat:r.lat, lon:r.lon, county:r.county,
          conditions:r.conditions, severity:r.severity, note:r.note,
          reporter:r.reporter, submitted:r.submitted,
          status:"pending", source:"resident"
        });
      });
      var blob=new Blob([lines.join("\n")+"\n"],
        {type:"application/x-ndjson"});
      var url=URL.createObjectURL(blob);
      var a=document.createElement("a");
      a.href=url; a.download="vista_community_reports.jsonl";
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
      rMsg("Downloaded "+mine.length+" report"+
        (mine.length===1?"":"s")+" — open a PR adding them to the ledger");
    });

  function updateCommStat(){
    var mine=loadMine();
    var base=COMM_SUM.n_reports||0;
    var bc=COMM_SUM.n_poles_corroborated||0;
    var elS=document.getElementById("commStat");
    if(!elS) return;
    var extra = mine.length? (" + <span style=\"color:"+CY+
      "\">"+mine.length+" local pending</span>") : "";
    elS.innerHTML="<b>Community:</b> "+base+" moderated field report"+
      (base===1?"":"s")+" · "+bc+" pole"+(bc===1?"":"s")+
      " corroborated"+extra+
      " &mdash; citizen-corroboration overlay, not a model input.";
  }

  // draw the community overlay markers (moderated ledger + my local).
  // Distinct from poles: hollow cyan diamonds; verified = filled core.
  function drawCommunity(){
    if(!state.showFieldReports) return;
    function diamond(lon,lat,filled,title){
      var x=+projX(lon).toFixed(1), y=+projY(lat).toFixed(1);
      var g=el("g",{style:"cursor:pointer"});
      g.appendChild(el("polygon",{
        points:x+","+(y-6.5)+" "+(x+6.5)+","+y+" "+x+","+(y+6.5)+" "+
          (x-6.5)+","+y,
        fill:filled?CY:"none","fill-opacity":filled?0.85:0,
        stroke:CY,"stroke-width":1.4}));
      var tEl=el("title",{}); tEl.textContent=title;
      g.appendChild(tEl);
      svg.appendChild(g);
    }
    COMM_REPORTS.forEach(function(r){
      diamond(r.lon, r.lat, r.status==="verified",
        "Field report "+r.report_id+" · "+r.status+" · "+
        (r.pole_id||"unmapped"));
    });
    loadMine().forEach(function(r){
      var x=+projX(r.lon).toFixed(1), y=+projY(r.lat).toFixed(1);
      svg.appendChild(el("polygon",{
        points:x+","+(y-7)+" "+(x+7)+","+y+" "+x+","+(y+7)+" "+
          (x-7)+","+y,
        fill:"none",stroke:CY,"stroke-width":1.3,
        "stroke-dasharray":"3 2"}));
    });
  }

  function render(){ drawMap(); drawTable(); drawCoverage();
    updateCommStat(); }
  document.getElementById("budgetVal").textContent="";
  render();
})();
</script>
</body>
</html>
"""


def build_app(fd: FleetData, fr: FitResult, vr: ValidationReport,
              econ: dict) -> str:
    """Write output/app_data.json + output/app.html and return the html path.

    The JSON is deterministic (json.dump with indent=2, sort_keys=True). The
    HTML embeds that exact JSON inline so the app is one self-contained file
    that works by double-click via file:// with zero network access.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = _build_payload(fd, fr, vr, econ)

    data_path = os.path.join(OUTPUT_DIR, "app_data.json")
    data_json = json.dumps(payload, indent=2, sort_keys=True)
    with open(data_path, "w", encoding="utf-8") as fh:
        fh.write(data_json)

    # Embed the SAME bytes inline. Escape the </script sequence so the JSON
    # cannot terminate the host <script> block (only realistic break vector
    # for inline JSON; no '<' is otherwise produced by json.dumps).
    inline = data_json.replace("</", "<\\/")
    html = _HTML_TEMPLATE.replace("__VISTA_DATA__", inline)

    html_path = os.path.join(OUTPUT_DIR, "app.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html_path
