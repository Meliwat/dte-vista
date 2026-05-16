"""The interactive offline app — the demoable DTE deliverable.

Asserts:
  * a full pipeline run produces output/app.html + output/app_data.json,
  * app_data.json has one record per pole and every required key,
  * app.html is ONE self-contained file: the data is embedded inline (not
    fetched) and there are NO http(s):// external resource references, so it
    opens by double-click via file:// and works fully offline,
  * the embedded JSON is byte-identical to app_data.json,
  * two pipeline runs produce a byte-identical app_data.json (determinism),
  * the action-derivation rule maps drivers correctly,
  * key interactive DOM ids are present in the shell.
"""

import json
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "output", "app.html")
DATA = os.path.join(REPO, "output", "app_data.json")

REQUIRED_TOP = ("meta", "kpis", "poles", "segments")
REQUIRED_META = ("territory_bbox", "counties", "noaa_stations", "generated",
                 "tier_thresholds", "n_poles")
REQUIRED_KPI = ("roc_auc", "pr_auc", "brier", "capture_at_20",
                "calibration_gap", "spatial_summary", "temporal_summary",
                "incumbent", "ablation", "economics")
REQUIRED_POLE = ("id", "lat", "lon", "county", "segment", "risk", "tier",
                 "drivers", "action")


def _run():
    return subprocess.run(
        [sys.executable, "-m", "vista"], cwd=REPO,
        capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def two_runs():
    r1 = _run()
    assert r1.returncode == 0, f"pipeline failed:\n{r1.stderr}"
    d1 = open(DATA, encoding="utf-8").read()
    html = open(APP, encoding="utf-8").read()
    r2 = _run()
    assert r2.returncode == 0, f"second pipeline run failed:\n{r2.stderr}"
    d2 = open(DATA, encoding="utf-8").read()
    return d1, d2, html, r1.stdout


def test_app_artifacts_emitted_by_pipeline(two_runs):
    d1, _, html, stdout = two_runs
    assert os.path.exists(APP) and os.path.getsize(APP) > 10_000
    assert os.path.exists(DATA) and os.path.getsize(DATA) > 10_000
    assert "[9] Building interactive offline app" in stdout
    assert "app.html" in stdout
    json.loads(d1)
    assert len(html) > 10_000


def test_app_data_shape_and_keys(two_runs):
    d1 = two_runs[0]
    obj = json.loads(d1)
    for k in REQUIRED_TOP:
        assert k in obj, f"missing top-level key {k}"
    for k in REQUIRED_META:
        assert k in obj["meta"], f"missing meta.{k}"
    for k in REQUIRED_KPI:
        assert k in obj["kpis"], f"missing kpis.{k}"

    # one pole record per fleet pole, all required fields, sorted by risk desc
    from vista.data_gen import generate_fleet
    fd = generate_fleet()
    poles = obj["poles"]
    assert len(poles) == len(fd.y), "poles must cover the whole fleet"
    assert obj["meta"]["n_poles"] == len(fd.y)
    risks = [p["risk"] for p in poles]
    assert risks == sorted(risks, reverse=True), "poles must be risk-desc"
    for p in poles:
        for k in REQUIRED_POLE:
            assert k in p, f"pole missing {k}"
        assert p["tier"] in ("CRITICAL", "HIGH", "ELEVATED", "ROUTINE")
        assert 0.0 <= p["risk"] <= 1.0
        assert isinstance(p["drivers"], list) and len(p["drivers"]) == 5
        for nm, c in p["drivers"]:
            assert isinstance(nm, str) and isinstance(c, (int, float))
    # segments sorted by risk desc and cover the fleet's segments
    segs = obj["segments"]
    assert {s["id"] for s in segs} == set(map(str, fd.segment_id))
    srisk = [s["risk"] for s in segs]
    assert srisk == sorted(srisk, reverse=True)
    assert sum(s["n_poles"] for s in segs) == len(fd.y)


def test_app_html_is_self_contained_and_offline(two_runs):
    html = two_runs[2]
    # the data is embedded inline in the marked script block (NOT fetched)
    assert '<script id="vista-data" type="application/json">' in html
    assert "__VISTA_DATA__" not in html, "data placeholder not substituted"
    assert "fetch(" not in html, "must not fetch — data is inline"
    assert "XMLHttpRequest" not in html

    # ZERO external resource references: no http(s) URLs pulling resources.
    # The only allowed http occurrence is the SVG XML namespace URI, which is
    # an identifier (createElementNS), not a network fetch.
    bad = re.findall(r'(?:src|href)\s*=\s*["\']https?://', html, re.I)
    assert not bad, f"external resource refs found: {bad[:3]}"
    bad2 = re.findall(r'@import|url\(\s*["\']?https?://', html, re.I)
    assert not bad2, f"external CSS refs found: {bad2[:3]}"
    for tok in ("http://", "https://"):
        for m in re.finditer(re.escape(tok), html):
            ctx = html[max(0, m.start() - 60):m.start() + 40]
            assert "www.w3.org/2000/svg" in ctx or "w3.org" in ctx, (
                f"unexpected external URL near: ...{ctx}...")
    # no CDN / web-font / map-tile giveaways
    for term in ("googleapis", "cdn.", "unpkg", "jsdelivr", "cloudflare",
                 "tile.openstreetmap", "mapbox", ".woff", "fonts.g"):
        assert term not in html, f"external dependency token: {term}"


def test_embedded_json_matches_app_data(two_runs):
    d1, _, html, _ = two_runs
    start = html.index('type="application/json">') + len(
        'type="application/json">')
    end = html.index("</script>", start)
    embedded = html[start:end].replace("<\\/", "</")
    assert json.loads(embedded) == json.loads(d1), (
        "embedded JSON must equal app_data.json")


def test_app_data_is_byte_identical_across_runs(two_runs):
    d1, d2 = two_runs[0], two_runs[1]
    assert d1 == d2, "app_data.json is not byte-deterministic across runs"


def test_key_interactive_dom_ids_present(two_runs):
    html = two_runs[2]
    for el_id in ("map", "kpis", "tierChips", "countySel", "minRisk",
                  "budget", "coverage", "wbody", "exportBtn", "detailPane"):
        assert ('id="%s"' % el_id) in html, f"missing #{el_id} in app shell"
    # the headline ablation stat + CSV export are wired
    assert "Imagery adds +" in html
    assert "vista_inspection_plan.csv" in html


def test_action_rule_maps_drivers_correctly():
    from vista.app_export import _action_for_driver
    assert _action_for_driver(
        "canopy encroaching the right-of-way (imagery)") == "TREE TRIM"
    assert _action_for_driver(
        "vegetation vigor / NDVI (imagery)") == "TREE TRIM"
    assert _action_for_driver(
        "pole lean detected from imagery") == "INSPECT/REPLACE"
    assert _action_for_driver("pole age") == "INSPECT/REPLACE"
    assert _action_for_driver(
        "flood-zone exposure (FEMA-style)") == "INSPECT (ground)"
    assert _action_for_driver("corrosive soil") == "INSPECT (ground)"
    assert _action_for_driver("electrical loading") == "INSPECT"
