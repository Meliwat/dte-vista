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

The app is a dark, professional risk-profiling dashboard a DTE planner can
actually click: an interactive territory map (poles projected from real
WGS84 lon/lat via ``TERRITORY_BBOX``), click-to-drill-down per-pole "why"
panels, tier/county/min-risk filters, a budget slider that shows the
reactive→predictive coverage story live, a sortable prioritized worklist,
and a client-side CSV export of the current worklist.

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
# --------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VISTA — Utility Pole Risk Profiling · DTE · Hack Michigan 2026</title>
<style>
  :root {
    --bg:#0b0f17; --panel:#121826; --panel2:#0e1320; --line:#222c40;
    --ink:#e8eef8; --muted:#8a97ad; --accent:#4ea1ff; --accent2:#2bd6a6;
    --crit:#d4322c; --high:#f08a24; --elev:#f3c613; --rout:#3a9d4e;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; padding:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,
    Arial,sans-serif; -webkit-font-smoothing:antialiased; }
  body { min-height:100vh; }
  a { color:var(--accent); }
  header.app {
    display:flex; align-items:center; justify-content:space-between;
    padding:16px 24px; border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,#0f1626,#0b0f17);
  }
  .brand { display:flex; align-items:baseline; gap:14px; }
  .brand h1 { font-size:21px; margin:0; letter-spacing:.3px; font-weight:700; }
  .brand .sub { color:var(--muted); font-size:12.5px; }
  .badge-dte {
    font-size:11px; font-weight:700; letter-spacing:1.2px;
    padding:5px 10px; border-radius:5px; color:#08111f;
    background:linear-gradient(90deg,#4ea1ff,#2bd6a6);
  }
  .kpis { display:flex; gap:14px; padding:14px 24px; flex-wrap:wrap;
    border-bottom:1px solid var(--line); background:var(--panel2); }
  .kpi { background:var(--panel); border:1px solid var(--line);
    border-radius:9px; padding:11px 16px; min-width:165px; flex:1; }
  .kpi .label { color:var(--muted); font-size:11px; text-transform:uppercase;
    letter-spacing:.7px; }
  .kpi .value { font-size:23px; font-weight:700; margin-top:4px; }
  .kpi.headline { background:linear-gradient(135deg,#10233a,#0e2b24);
    border-color:#1e6f9c; }
  .kpi.headline .value { color:var(--accent2); font-size:20px; }
  .layout { display:grid; grid-template-columns:268px 1fr 392px; gap:0;
    min-height:calc(100vh - 168px); }
  .panel { padding:18px; border-right:1px solid var(--line); }
  .panel.right { border-right:none; border-left:1px solid var(--line);
    overflow-y:auto; max-height:calc(100vh - 168px); }
  .center { padding:14px 18px; display:flex; flex-direction:column; gap:12px;
    min-width:0; }
  h2.sec { font-size:12px; text-transform:uppercase; letter-spacing:1px;
    color:var(--muted); margin:0 0 10px; font-weight:700; }
  .ctl { margin-bottom:22px; }
  .chips { display:flex; flex-wrap:wrap; gap:7px; }
  .chip { font-size:11.5px; padding:6px 11px; border-radius:20px;
    border:1px solid var(--line); background:var(--panel); cursor:pointer;
    user-select:none; display:flex; align-items:center; gap:6px;
    transition:all .12s; }
  .chip .dot { width:9px; height:9px; border-radius:50%; }
  .chip.off { opacity:.32; }
  .chip:hover { border-color:var(--accent); }
  select, input[type=range] { width:100%; }
  select { background:var(--panel); color:var(--ink);
    border:1px solid var(--line); border-radius:7px; padding:8px;
    font-size:13px; }
  input[type=range] { accent-color:var(--accent); margin-top:8px; }
  .ctl .val { font-size:13px; color:var(--ink); margin-top:7px;
    font-variant-numeric:tabular-nums; }
  .ctl .hint { font-size:11.5px; color:var(--muted); margin-top:4px;
    line-height:1.5; }
  .coverage { background:var(--panel); border:1px solid #1e6f9c;
    border-radius:9px; padding:12px; margin-top:10px; font-size:12.5px;
    line-height:1.6; }
  .coverage b { color:var(--accent2); }
  .coverage .react { color:var(--muted); }
  .mapwrap { background:var(--panel2); border:1px solid var(--line);
    border-radius:11px; padding:6px; position:relative; }
  svg.map { width:100%; height:auto; display:block; border-radius:8px; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; padding:8px 6px 2px;
    font-size:11.5px; color:var(--muted); }
  .legend span { display:flex; align-items:center; gap:6px; }
  .legend i { width:11px; height:11px; border-radius:50%; display:inline-block; }
  #tooltip { position:fixed; pointer-events:none; background:#000d;
    border:1px solid var(--accent); color:var(--ink); font-size:12px;
    padding:7px 10px; border-radius:6px; display:none; z-index:50;
    white-space:nowrap; }
  .tablewrap { background:var(--panel2); border:1px solid var(--line);
    border-radius:11px; overflow:hidden; flex:1; display:flex;
    flex-direction:column; min-height:230px; }
  .tablebar { display:flex; align-items:center; justify-content:space-between;
    padding:10px 14px; border-bottom:1px solid var(--line); }
  .tablebar .count { color:var(--muted); font-size:12px; }
  button.exp { background:linear-gradient(90deg,#4ea1ff,#2bd6a6);
    color:#08111f; border:none; border-radius:7px; padding:8px 14px;
    font-size:12.5px; font-weight:700; cursor:pointer; }
  button.exp:hover { filter:brightness(1.08); }
  .tscroll { overflow:auto; flex:1; max-height:340px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th { position:sticky; top:0; background:#0c1220; color:var(--muted);
    text-align:left; padding:9px 12px; font-weight:600; font-size:11px;
    text-transform:uppercase; letter-spacing:.6px; border-bottom:1px solid
    var(--line); white-space:nowrap; }
  th.sortable { cursor:pointer; user-select:none; }
  th.sortable:hover { color:var(--ink); }
  td { padding:8px 12px; border-bottom:1px solid #182238;
    white-space:nowrap; }
  tbody tr { cursor:pointer; }
  tbody tr:hover { background:#16203400; background-color:#172136; }
  tbody tr.sel { background:#1d2c47; }
  .pill { font-size:10.5px; font-weight:700; padding:2px 8px;
    border-radius:11px; letter-spacing:.4px; }
  .mono { font-variant-numeric:tabular-nums;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .detail .big { font-size:42px; font-weight:800; line-height:1;
    margin:2px 0 8px; }
  .detail .meta { color:var(--muted); font-size:12.5px; margin-bottom:14px;
    line-height:1.7; }
  .drv { margin:9px 0; }
  .drv .dl { display:flex; justify-content:space-between; font-size:12px;
    margin-bottom:4px; }
  .drv .dl .nm { color:var(--ink); }
  .drv .dl .cv { color:var(--muted); font-variant-numeric:tabular-nums; }
  .bar { height:9px; border-radius:5px; background:#1a2540; overflow:hidden; }
  .bar > i { display:block; height:100%; border-radius:5px; }
  .callout { margin-top:18px; border:1px solid var(--accent2);
    background:linear-gradient(135deg,#0e2b24,#10233a); border-radius:10px;
    padding:14px; }
  .callout .lab { font-size:11px; text-transform:uppercase;
    letter-spacing:1px; color:var(--muted); }
  .callout .act { font-size:22px; font-weight:800; color:var(--accent2);
    margin-top:5px; }
  .empty { color:var(--muted); font-size:13px; padding:30px 4px;
    text-align:center; line-height:1.6; }
  footer.app { padding:14px 24px; border-top:1px solid var(--line);
    color:var(--muted); font-size:11.5px; line-height:1.6; }
  .ring { fill:none; stroke:#fff; stroke-width:1.6; }
  .gridln { stroke:#2a3450; }
  @media (max-width:1100px){ .layout{ grid-template-columns:1fr; }
    .panel,.panel.right{ border:none; border-bottom:1px solid var(--line);
    max-height:none; } }
</style>
</head>
<body>
<header class="app">
  <div class="brand">
    <h1>VISTA — Utility Pole Risk Profiling</h1>
    <span class="sub" id="subtitle"></span>
  </div>
  <span class="badge-dte">DTE ENERGY · HACK MICHIGAN 2026</span>
</header>

<div class="kpis" id="kpis"></div>

<div class="layout">
  <aside class="panel">
    <div class="ctl">
      <h2 class="sec">Tier filter</h2>
      <div class="chips" id="tierChips"></div>
    </div>
    <div class="ctl">
      <h2 class="sec">County</h2>
      <select id="countySel"></select>
    </div>
    <div class="ctl">
      <h2 class="sec">Minimum risk</h2>
      <input type="range" id="minRisk" min="0" max="100" value="0" step="1">
      <div class="val" id="minRiskVal">≥ 0%</div>
    </div>
    <div class="ctl">
      <h2 class="sec">Inspection budget</h2>
      <input type="range" id="budget" min="1" max="100" value="20" step="1">
      <div class="val" id="budgetVal"></div>
      <div class="hint">Drag to size the crew's truck-roll budget. The
        worklist + map "priority" ring follow the top-N by predicted risk.</div>
      <div class="coverage" id="coverage"></div>
    </div>
  </aside>

  <main class="center">
    <div class="mapwrap">
      <svg class="map" id="map" viewBox="0 0 1000 620"
           preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend">
        <span><i style="background:#d4322c"></i>CRITICAL ≥65%</span>
        <span><i style="background:#f08a24"></i>HIGH ≥40%</span>
        <span><i style="background:#f3c613"></i>ELEVATED ≥22%</span>
        <span><i style="background:#3a9d4e"></i>ROUTINE</span>
        <span><svg width="14" height="14"><polygon points="7,2 13,12 1,12"
          fill="#4ea1ff" stroke="#fff" stroke-width="0.7"/></svg>
          NOAA station</span>
        <span><svg width="16" height="16"><circle cx="8" cy="8" r="5"
          fill="none" stroke="#fff" stroke-width="1.6"/></svg>
          Within budget (priority)</span>
      </div>
    </div>

    <div class="tablewrap">
      <div class="tablebar">
        <div><strong>Prioritized inspection worklist</strong>
          <span class="count" id="wcount"></span></div>
        <button class="exp" id="exportBtn">⤓ Export inspection plan (CSV)</button>
      </div>
      <div class="tscroll">
        <table>
          <thead><tr>
            <th>#</th><th>Pole</th>
            <th class="sortable" data-k="county">County ▾</th>
            <th>Segment</th>
            <th class="sortable" data-k="risk">Risk ▾</th>
            <th>Tier</th><th>Top driver</th><th>Action</th>
          </tr></thead>
          <tbody id="wbody"></tbody>
        </table>
      </div>
    </div>
  </main>

  <aside class="panel right" id="detailPane">
    <div class="empty" id="detailEmpty">
      Click any pole on the map or a worklist row<br>
      to see <strong>why</strong> it's flagged and the recommended action.
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
  var TIER_COLOR = {CRITICAL:"#d4322c",HIGH:"#f08a24",
                    ELEVATED:"#f3c613",ROUTINE:"#3a9d4e"};
  var TIERS = ["CRITICAL","HIGH","ELEVATED","ROUTINE"];

  // ---- state ----
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
    "Data: synthetic DTE-style fleet anchored to REAL NOAA 1991–2020 climate "+
    "normals ("+M.noaa_stations.length+" stations) + image-derived NDVI / "+
    "structure features. Deterministic, offline, no proprietary inputs. "+
    "Generated "+esc(M.generated)+" · "+M.n_poles.toLocaleString()+" poles · "+
    "tier thresholds CRITICAL≥0.65 / HIGH≥0.40 / ELEVATED≥0.22.";

  // ---- KPI strip ----
  var inc=K.incumbent, abl=K.ablation, ec=K.economics;
  var kpiHtml = [
    ["Held-out ROC-AUC", K.roc_auc.toFixed(3), ""],
    ["Lift vs age-cycle", inc.lift_vs_age_cycle_x.toFixed(2)+"×", ""],
    ["Net benefit / cycle", usd(ec.net_benefit_fleet_usd), ""],
    ["Capture @20% budget", pct0(K.capture_at_20), ""],
    ["Imagery ablation",
      "Imagery adds +"+abl.auc_gain_from_imagery.toFixed(3)+" ROC-AUC",
      "headline"]
  ].map(function(k){
    return '<div class="kpi '+k[2]+'"><div class="label">'+esc(k[0])+
      '</div><div class="value">'+esc(k[1])+'</div></div>';
  }).join("");
  document.getElementById("kpis").innerHTML = kpiHtml;

  // ---- controls: tier chips ----
  var tc = document.getElementById("tierChips");
  tc.innerHTML = TIERS.map(function(t){
    return '<div class="chip" data-t="'+t+'"><span class="dot" style="'+
      'background:'+TIER_COLOR[t]+'"></span>'+t+'</div>';
  }).join("");
  Array.prototype.forEach.call(tc.querySelectorAll(".chip"),function(el){
    el.addEventListener("click",function(){
      var t=el.getAttribute("data-t");
      state.tiers[t]=!state.tiers[t]; render();
    });
  });

  // ---- controls: county ----
  var cs = document.getElementById("countySel");
  cs.innerHTML = '<option value="ALL">All counties</option>' +
    M.counties.slice().sort().map(function(c){
      return '<option value="'+esc(c)+'">'+esc(c)+'</option>'; }).join("");
  cs.addEventListener("change",function(){ state.county=cs.value; render(); });

  // ---- controls: sliders ----
  var mr=document.getElementById("minRisk"),
      mv=document.getElementById("minRiskVal");
  mr.addEventListener("input",function(){
    state.minRisk=+mr.value; mv.textContent="≥ "+mr.value+"%"; render(); });
  var bg=document.getElementById("budget");
  bg.addEventListener("input",function(){
    state.budget=+bg.value; render(); });

  // ---- map projection (real WGS84 -> SVG) ----
  var BB=M.territory_bbox, VW=1000, VH=620,
      PADX=46, PADY=30, IW=VW-2*PADX, IH=VH-2*PADY;
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

  // budget set = top-N of the WHOLE fleet by risk (POLES is risk-desc sorted)
  function budgetSet(){
    var n=Math.max(1,Math.round(POLES.length*state.budget/100));
    var s={}; for(var i=0;i<n && i<POLES.length;i++) s[POLES[i].id]=true;
    return {set:s,n:n};
  }

  var tip=document.getElementById("tooltip");
  function showTip(e,p){
    tip.style.display="block";
    tip.style.left=(e.clientX+14)+"px";
    tip.style.top=(e.clientY+14)+"px";
    tip.innerHTML="<b>"+esc(p.id)+"</b> · "+esc(p.county)+
      "<br>risk "+pct(p.risk)+" · "+p.tier;
  }
  function hideTip(){ tip.style.display="none"; }

  function drawMap(){
    while(svg.firstChild) svg.removeChild(svg.firstChild);
    // backdrop
    svg.appendChild(el("rect",{x:0,y:0,width:VW,height:VH,
      rx:8,fill:"#0c1220"}));
    // county vertical bands + labels
    var nC=M.counties.length;
    for(var i=0;i<=nC;i++){
      var lon=BB.lon_min+i*(BB.lon_max-BB.lon_min)/nC, x=projX(lon);
      svg.appendChild(el("line",{x1:x,y1:PADY,x2:x,y2:VH-PADY,
        stroke:"#1f2a40","stroke-width":1,"stroke-dasharray":"2 4"}));
    }
    for(i=0;i<nC;i++){
      var cl=BB.lon_min+(i+0.5)*(BB.lon_max-BB.lon_min)/nC;
      var t=el("text",{x:projX(cl),y:PADY+13,fill:"#5c6b86",
        "font-size":11,"text-anchor":"middle"});
      t.textContent=M.counties[i]; svg.appendChild(t);
    }
    // frame
    svg.appendChild(el("rect",{x:PADX,y:PADY,width:IW,height:IH,
      fill:"none",stroke:"#26334d","stroke-width":1}));

    var fset = budgetSet().set;
    var show = filtered();
    var showIds={}; show.forEach(function(p){ showIds[p.id]=true; });

    // poles (dim the ones filtered out, keep geography readable)
    POLES.forEach(function(p){
      var vis = showIds[p.id];
      var c=el("circle",{cx:projX(p.lon).toFixed(1),
        cy:projY(p.lat).toFixed(1), r:vis?4.2:2.2,
        fill:TIER_COLOR[p.tier], "fill-opacity":vis?0.95:0.12,
        stroke:(state.selected===p.id)?"#fff":"none",
        "stroke-width":(state.selected===p.id)?2:0});
      if(vis){
        c.style.cursor="pointer";
        c.addEventListener("mousemove",function(e){ showTip(e,p); });
        c.addEventListener("mouseleave",hideTip);
        c.addEventListener("click",function(){ select(p.id); });
      }
      svg.appendChild(c);
    });
    // priority rings = within-budget AND currently visible
    POLES.forEach(function(p){
      if(fset[p.id] && showIds[p.id]){
        svg.appendChild(el("circle",{cx:projX(p.lon).toFixed(1),
          cy:projY(p.lat).toFixed(1), r:7.5, fill:"none",
          stroke:"#ffffff","stroke-width":1.4,"stroke-opacity":0.85}));
      }
    });
    // NOAA stations
    M.noaa_stations.forEach(function(s){
      var x=projX(s.lon),y=projY(s.lat);
      svg.appendChild(el("polygon",{
        points:(x)+","+(y-6)+" "+(x+6)+","+(y+5)+" "+(x-6)+","+(y+5),
        fill:"#4ea1ff",stroke:"#fff","stroke-width":0.8}));
    });
  }

  // ---- detail pane ----
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
      var raises=d[1]>=0, w=Math.max(4,Math.abs(d[1])/maxAbs*100);
      return '<div class="drv"><div class="dl"><span class="nm">'+
        esc(d[0])+'</span><span class="cv">'+
        (raises?"+":"")+d[1].toFixed(4)+'</span></div>'+
        '<div class="bar"><i style="width:'+w.toFixed(1)+
        '%;background:'+(raises?"#d4322c":"#3a9d4e")+'"></i></div></div>';
    }).join("");
    bd.innerHTML =
      '<h2 class="sec">Pole '+esc(p.id)+' — risk profile</h2>'+
      '<div class="big" style="color:'+TIER_COLOR[p.tier]+'">'+pct(p.risk)+
      ' <span class="pill" style="background:'+TIER_COLOR[p.tier]+
      ';color:#0b0f17;font-size:12px;vertical-align:middle">'+p.tier+
      '</span></div>'+
      '<div class="meta">County <b>'+esc(p.county)+'</b> · Segment <b>'+
      esc(p.segment)+'</b><br>Location '+p.lat.toFixed(4)+', '+
      p.lon.toFixed(4)+' (WGS84)</div>'+
      '<h2 class="sec">Why this pole is flagged</h2>'+drvHtml+
      '<div class="callout"><div class="lab">Recommended action</div>'+
      '<div class="act">'+esc(p.action)+'</div></div>';
    drawMap(); drawTable();
  }

  // ---- worklist table ----
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
      "· "+rows.length+" of "+POLES.length+" poles";
    document.getElementById("wcount").setAttribute("data-n",rows.length);
    if(!rows.length){
      tb.innerHTML='<tr><td colspan="8" class="empty">'+
        'No poles match the current filters.</td></tr>'; return; }
    var html="";
    for(var i=0;i<rows.length;i++){
      var p=rows[i], sel=(state.selected===p.id)?" sel":"";
      html+='<tr class="'+sel.trim()+'" data-id="'+esc(p.id)+'">'+
        '<td class="mono">'+(i+1)+'</td>'+
        '<td class="mono">'+esc(p.id)+'</td>'+
        '<td>'+esc(p.county)+'</td>'+
        '<td class="mono">'+esc(p.segment)+'</td>'+
        '<td class="mono">'+pct(p.risk)+'</td>'+
        '<td><span class="pill" style="background:'+TIER_COLOR[p.tier]+
        ';color:#0b0f17">'+p.tier+'</span></td>'+
        '<td>'+esc(p.drivers.length?p.drivers[0][0]:"")+'</td>'+
        '<td><b>'+esc(p.action)+'</b></td></tr>';
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

  // ---- budget coverage story ----
  function drawCoverage(){
    var b=budgetSet(), n=b.n, m=POLES.length;
    // predicted failures = poles at HIGH+ (the model's own failure call)
    var predFail=POLES.filter(function(p){ return p.risk>=0.40; }).length;
    var caught=0;
    for(var i=0;i<n && i<m;i++) if(POLES[i].risk>=0.40) caught++;
    var covPct = predFail? Math.round(caught/predFail*100):0;
    var reactivePct = Math.round(n/m*100);
    document.getElementById("budgetVal").textContent=
      "Inspect top "+n+" of "+m+" poles ("+reactivePct+"%)";
    document.getElementById("coverage").innerHTML=
      "Inspecting <b>"+n+"</b> of "+m+" poles, VISTA catches <b>"+covPct+
      "%</b> of the "+predFail+" predicted failures.<br>"+
      '<span class="react">A reactive / age-blind crew inspecting the same '+
      n+" poles would expect ≈"+reactivePct+"%.</span>";
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

  // init
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
