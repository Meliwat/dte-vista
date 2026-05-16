"""Economic impact arithmetic + the provider-agnostic LLM seam."""

import pytest

from vista.config import TEST_FRACTION, VAL_FRACTION
from vista.data_gen import generate_fleet
from vista.impact import economic_impact
from vista.llm import generate_brief
from vista.model import fit, split_indices


@pytest.fixture(scope="module")
def fitted():
    fd = generate_fleet()
    tr, va, te = split_indices(len(fd.y), fd.y, TEST_FRACTION, VAL_FRACTION)
    return fd, fit(fd.X, fd.y, fd.feature_names, tr, va, te)


def test_economic_impact_positive_and_consistent(fitted):
    fd, fr = fitted
    e = economic_impact(fr, fd)
    assert e["extra_failures_caught_fleet"] > 0
    assert e["avoided_outage_cost_fleet_usd"] > 0
    assert e["net_benefit_fleet_usd"] > 0
    assert e["benefit_cost_ratio"] > 1.0
    assert e["budget_inspections_per_cycle_fleet"] == int(round(1400 * 0.20))


def test_economic_impact_deterministic(fitted):
    fd, fr = fitted
    assert economic_impact(fr, fd) == economic_impact(fr, fd)


def test_llm_stub_is_deterministic_and_offline():
    drivers = [("canopy encroaching the right-of-way (imagery)", 0.21),
               ("pole lean detected from imagery", 0.14)]
    a = generate_brief("P00001", 0.82, drivers, provider="stub")
    b = generate_brief("P00001", 0.82, drivers, provider="stub")
    assert a == b
    assert "P00001" in a and "CRITICAL" in a
    assert "right-of-way" in a


def test_llm_tiers_change_with_risk():
    d = [("pole age", 0.1)]
    assert "CRITICAL" in generate_brief("P", 0.90, d)
    assert "ROUTINE" in generate_brief("P", 0.05, d)


def test_real_llm_providers_fail_loudly_without_keys(monkeypatch):
    """The documented real seams must NOT silently degrade determinism -
    they raise a clear error when selected without credentials."""
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        generate_brief("P", 0.5, [("x", 0.1)], provider="watsonx")
    with pytest.raises(RuntimeError):
        generate_brief("P", 0.5, [("x", 0.1)], provider="google_adk")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        generate_brief("P", 0.5, [("x", 0.1)], provider="bogus")
