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


# --------------------------------------------------------------------------
# Citizen-corroboration overlay (community field reports). The moderated,
# repo-committed ledger is folded into the payload as an OVERLAY ONLY: it is
# a deterministic priority/corroboration signal reconciled against the model
# and is NEVER model input. These tests do not weaken any test above.
# --------------------------------------------------------------------------

LEDGER = os.path.join(REPO, "community_reports", "ledger.jsonl")


def test_community_ledger_file_parses_and_excludes_rejected():
    """The committed ledger is valid JSON Lines with the documented fields,
    and at least one 'rejected' row exists (to prove it is dropped on
    ingest, not absent from the file)."""
    assert os.path.isfile(LEDGER), "community_reports/ledger.jsonl missing"
    rows = []
    with open(LEDGER, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))  # must be valid JSON
    assert len(rows) >= 5, "expected 5-6 seed rows"
    req = ("report_id", "pole_id", "lat", "lon", "county", "conditions",
           "severity", "note", "reporter", "submitted", "status", "source")
    for r in rows:
        for k in req:
            assert k in r, f"ledger row missing {k}"
        assert r["severity"] in ("low", "medium", "urgent")
        assert r["status"] in ("verified", "pending", "rejected")
        assert r["source"] in ("resident", "lineman", "sample")
        assert isinstance(r["conditions"], list) and r["conditions"]
    assert any(r["status"] == "rejected" for r in rows), (
        "ledger should contain a rejected row to exercise the filter")
    assert any(r["pole_id"] is None for r in rows), (
        "ledger should contain one unmapped (pole_id=null) row")


def test_payload_has_community_overlay_shape(two_runs):
    obj = json.loads(two_runs[0])
    assert "community" in obj, "missing top-level community overlay"
    c = obj["community"]
    for k in ("reports", "summary", "ledger_path", "note"):
        assert k in c, f"community missing {k}"
    s = c["summary"]
    for k in ("n_reports", "n_verified", "n_pending", "n_unmapped",
              "n_poles_corroborated", "by_severity"):
        assert k in s, f"community.summary missing {k}"

    reports = c["reports"]
    # only verified/pending survive ingest; rejected dropped entirely
    assert reports, "expected ingested community reports"
    for r in reports:
        assert r["status"] in ("verified", "pending"), (
            "rejected row leaked into the overlay")
        for k in ("report_id", "pole_id", "lat", "lon", "county",
                  "conditions", "severity", "note", "reporter",
                  "submitted", "status", "source"):
            assert k in r, f"normalized report missing {k}"
    # deterministic ordering: sorted by report_id
    ids = [r["report_id"] for r in reports]
    assert ids == sorted(ids), "community reports must be report_id-sorted"
    assert s["n_reports"] == len(reports)
    assert s["n_verified"] == sum(
        1 for r in reports if r["status"] == "verified")

    # cross-check the file: every rejected row in the ledger is absent
    rej = set()
    with open(LEDGER, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if row["status"] == "rejected":
                rej.add(row["report_id"])
    assert rej, "test fixture expects a rejected row in the ledger"
    assert not (rej & set(ids)), "a rejected report_id leaked into payload"


def test_corroborated_pole_exposes_community_fields(two_runs):
    obj = json.loads(two_runs[0])
    poles = obj["poles"]
    # every pole carries the overlay fields (added keys; required keys above
    # are still all asserted by test_app_data_shape_and_keys)
    for p in poles:
        assert "community_n" in p and "community_status" in p
        assert isinstance(p["community_n"], int) and p["community_n"] >= 0
        assert p["community_status"] in ("corroborated", "reported", "none")
    corr = [p for p in poles if p["community_n"] > 0]
    assert corr, "expected at least one pole with a community report"
    # a verified report => 'corroborated'; pole_id must match a real pole
    ver_ids = {r["pole_id"] for r in obj["community"]["reports"]
               if r["status"] == "verified" and r["pole_id"] is not None}
    assert ver_ids, "fixture expects >=1 verified mapped report"
    by_id = {p["id"]: p for p in poles}
    for pid in ver_ids:
        assert pid in by_id, f"verified report references unknown pole {pid}"
        assert by_id[pid]["community_n"] >= 1
        assert by_id[pid]["community_status"] == "corroborated"


def test_overlay_does_not_change_model_output():
    """Hard invariant: the overlay is reconciliation only. risk / tier /
    drivers / action / ordering / kpis / segments must be byte-identical
    whether or not the ledger is present."""
    import vista.app_export as ae
    from vista.data_gen import generate_fleet
    from vista.impact import economic_impact
    from vista.model import fit, split_indices
    from vista.validation import run_validation
    from vista.config import TEST_FRACTION, VAL_FRACTION

    fd = generate_fleet()
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    fr = fit(fd.X, fd.y, fd.feature_names, tr, va, te)
    vr = run_validation(fd, fr)
    econ = economic_impact(fr, fd)

    p_with = ae._build_payload(fd, fr, vr, econ)
    orig = ae._load_community_reports
    try:
        ae._load_community_reports = lambda path=ae.COMMUNITY_LEDGER_PATH: []
        p_without = ae._build_payload(fd, fr, vr, econ)
    finally:
        ae._load_community_reports = orig

    def core(pl):
        return [(x["id"], x["risk"], x["tier"], x["action"], x["segment"],
                 tuple(tuple(d) for d in x["drivers"])) for x in pl["poles"]]

    assert core(p_with) == core(p_without), (
        "community overlay altered model risk/tier/drivers/order")
    assert p_with["kpis"] == p_without["kpis"]
    assert p_with["segments"] == p_without["segments"]
    # empty/missing ledger degrades gracefully
    assert p_without["community"]["summary"]["n_reports"] == 0
    assert all(x["community_n"] == 0 for x in p_without["poles"])


def test_report_mode_dom_ids_present(two_runs):
    html = two_runs[2]
    for el_id in ("reportToggle", "reportForm", "reportPoleId",
                  "reportConds", "reportSev", "reportNote",
                  "reportReporter", "reportSubmit", "reportDownload",
                  "reportClear", "reportMsg", "fieldReportsToggle",
                  "mapwrap", "commStat", "reportLoc", "reportCap",
                  "reportFlow"):
        assert ('id="%s"' % el_id) in html, (
            f"missing report-mode #{el_id} in app shell")
    # the client-side ledger-schema export is wired
    assert "vista_community_reports.jsonl" in html
    # the moderated-flow plain-text panel + ledger path are present, with
    # NO clickable URL (the offline/self-contained test still enforces this)
    assert "community_reports/ledger.jsonl" in html
    assert "pull request" in html.lower()


def test_offline_self_contained_still_holds_with_overlay(two_runs):
    """Re-assert the offline guarantee now that the overlay UI/data exist:
    no new external refs or banned tokens were introduced."""
    html = two_runs[2]
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    for tok in ("http://", "https://"):
        for m in re.finditer(re.escape(tok), html):
            ctx = html[max(0, m.start() - 60):m.start() + 40]
            assert "w3.org" in ctx, f"unexpected external URL: ...{ctx}..."
    for term in ("googleapis", "cdn.", "unpkg", "jsdelivr", "cloudflare",
                 "tile.openstreetmap", "mapbox", ".woff", "fonts.g"):
        assert term not in html, f"external dependency token: {term}"
    # the exact embed contract is intact
    assert '<script id="vista-data" type="application/json">' in html
    assert "__VISTA_DATA__" not in html
