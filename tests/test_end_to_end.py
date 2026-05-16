"""End-to-end: the one-command pipeline runs and produces every artifact,
byte-identically across two invocations (defends the Completeness axis).

The pipeline is invoked exactly twice (module-scoped) and all assertions
reuse those two runs, so the suite stays fast.
"""

import hashlib
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run():
    return subprocess.run(
        [sys.executable, "-m", "vista"], cwd=REPO,
        capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def two_runs():
    r1 = _run()
    png = os.path.join(REPO, "output", "dashboard.png")
    js = os.path.join(REPO, "output", "summary.json")
    h1 = hashlib.md5(open(png, "rb").read()).hexdigest()
    j1 = open(js).read()
    r2 = _run()
    h2 = hashlib.md5(open(png, "rb").read()).hexdigest()
    j2 = open(js).read()
    return r1, r2, h1, h2, j1, j2


def test_pipeline_runs_clean_and_emits_artifacts(two_runs):
    r1, _, _, _, j1, _ = two_runs
    assert r1.returncode == 0, f"pipeline failed:\n{r1.stderr}"
    assert "DONE" in r1.stdout
    png = os.path.join(REPO, "output", "dashboard.png")
    assert os.path.exists(png) and os.path.getsize(png) > 50_000
    s = json.loads(j1)
    for k in ("noaa", "heldout", "calibration", "spatial_summary",
              "temporal_summary", "incumbent", "ablation", "economics"):
        assert k in s


def test_pipeline_is_byte_identical_across_runs(two_runs):
    r1, r2, h1, h2, j1, j2 = two_runs
    assert h1 == h2, "dashboard PNG is not byte-deterministic"
    assert j1 == j2, "summary.json is not byte-deterministic"
    assert r1.stdout == r2.stdout, "stdout is not deterministic"


def test_headline_numbers_match_brief_question(two_runs):
    """The pipeline must answer THE QUESTION with a number: lift over the
    incumbent + a dollar figure must be present and favourable."""
    _, _, _, _, j1, _ = two_runs
    s = json.loads(j1)
    assert s["incumbent"]["capture_vista"] > s["incumbent"]["capture_age_cycle"]
    assert s["economics"]["net_benefit_fleet_usd"] > 0
    assert s["ablation"]["auc_gain_from_imagery"] > 0.03
