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


def _build_payload(fd: FleetData, fr: FitResult, vr: ValidationReport,
                    econ: dict) -> dict:
    """Assemble the deterministic data dict embedded in the app."""
    proba = predict_proba(fr, fd.X)
    seg = segment_risk(np.asarray(fd.segment_id), proba)
    stations = load_noaa_normals()

    # poles, sorted by risk desc (stable tie-break by index → deterministic)
    order = np.argsort(-proba, kind="stable")
    poles: List[dict] = []
    for j in order:
        j = int(j)
        drv = explain_pole(fr, fd.X[j], top_k=5)
        drivers = [[str(name), round(float(c), 4)] for name, c in drv]
        top_driver = drivers[0][0] if drivers else ""
        poles.append({
            "id": str(fd.pole_id[j]),
            "lat": round(float(fd.lat[j]), 6),
            "lon": round(float(fd.lon[j]), 6),
            "county": str(fd.county[j]),
            "segment": str(fd.segment_id[j]),
            "risk": round(float(proba[j]), 4),
            "tier": _tier(float(proba[j])),
            "drivers": drivers,
            "action": _action_for_driver(top_driver),
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
  @media (max-width:1140px){.layout{grid-template-columns:1fr;}
    .panel,.panel.right{border:none;border-bottom:1px solid var(--line);
      max-height:none;}}
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
    <div class="mapwrap">
      <div class="fieldhdr">
        <span>FIG · DTE SE-MICHIGAN TERRITORY — POLE RISK FIELD</span>
        <span class="rt mono">WGS84 · LIVE</span>
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

<footer class="app" id="footer"></footer>
<div id="tooltip"></div>

<script id="vista-data" type="application/json">__VISTA_DATA__</script>
<script>
"use strict";
(function(){
  var D = JSON.parse(document.getElementById("vista-data").textContent);
  var M = D.meta, K = D.kpis, POLES = D.poles;
  var TIER_COLOR = {CRITICAL:"#ff394e",HIGH:"#ff8f1f",
                    ELEVATED:"#ffd23f",ROUTINE:"#1fd98c"};
  var TIERS = ["CRITICAL","HIGH","ELEVATED","ROUTINE"];

  var state = {
    tiers:{CRITICAL:true,HIGH:true,ELEVATED:true,ROUTINE:true},
    county:"ALL", minRisk:0, budget:20,
    sortKey:"risk", sortDir:-1, selected:null
  };

  function esc(s){ return String(s).replace(/[&<>"]/g,function(c){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]; }); }
  function pct(x){ return (x*100).toFixed(1)+"%"; }
  function pct0(x){ return Math.round(x*100)+"%"; }
  function usd(x){ return "$"+Math.round(x).toLocaleString("en-US"); }

  document.getElementById("subtitle").textContent = M.subtitle;
  document.getElementById("footer").innerHTML =
    "<b>Provenance.</b> Synthetic DTE-style fleet anchored to REAL NOAA "+
    "1991–2020 climate normals ("+M.noaa_stations.length+" stations) + "+
    "image-derived NDVI / structure features. Deterministic, offline, no "+
    "proprietary inputs. Generated "+esc(M.generated)+" · "+
    M.n_poles.toLocaleString()+" poles · tiers CRITICAL&ge;0.65 / "+
    "HIGH&ge;0.40 / ELEVATED&ge;0.22.";

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
      '<div class="callout"><div class="lab">Recommended directive</div>'+
      '<div class="act">&#9656; '+esc(p.action)+'</div></div>';
    drawMap(); drawTable();
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

  function render(){ drawMap(); drawTable(); drawCoverage(); }
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
