"""Provider-agnostic LLM seam.

VISTA needs zero network and zero keys to run, score, validate and render.
The optional LLM layer only turns the model's *already-computed* numeric
driver attributions into a one-paragraph field briefing for a planner. The
default provider is a fully deterministic local stub (templated from the
structured drivers) so output is byte-identical offline.

Real providers are documented seams, NOT runtime dependencies:

  * IBM watsonx.ai (Granite)  - set provider="watsonx"; reads
    WATSONX_API_KEY / WATSONX_PROJECT_ID / WATSONX_URL and calls the
    granite-3 chat endpoint. Mapped in README "Technology Utilized".
  * Google ADK / Gemini       - set provider="google_adk"; wraps a
    google-adk Agent (model "gemini-2.0-flash"); reads GOOGLE_API_KEY.

Both real branches raise a clear error if selected without credentials -
they never degrade determinism of the default path.
"""

from __future__ import annotations

import os
from typing import List, Tuple


def _stub_brief(pole_id: str, risk: float, drivers: List[Tuple[str, float]]) -> str:
    tier = ("CRITICAL" if risk >= 0.65 else
            "HIGH" if risk >= 0.40 else
            "ELEVATED" if risk >= 0.22 else "ROUTINE")
    up = [d for d, c in drivers if c > 0][:3]
    drv = "; ".join(up) if up else "no dominant adverse driver"
    rec = {
        "CRITICAL": "dispatch a priority field inspection and stage replacement materials",
        "HIGH": "schedule inspection this cycle; pre-position a tree-trim crew if vegetation-driven",
        "ELEVATED": "add to the routed inspection batch this season",
        "ROUTINE": "retain on the standard cycle; monitor imagery next pass",
    }[tier]
    return (f"Pole {pole_id} - {tier} risk ({risk:.0%}). "
            f"Primary drivers: {drv}. Recommended action: {rec}.")


def generate_brief(pole_id: str, risk: float,
                   drivers: List[Tuple[str, float]],
                   provider: str = "stub") -> str:
    """Return a one-paragraph field briefing for a pole.

    provider: "stub" (default, deterministic, offline) | "watsonx" | "google_adk".
    """
    if provider == "stub":
        return _stub_brief(pole_id, risk, drivers)

    if provider == "watsonx":  # pragma: no cover - documented seam
        key = os.environ.get("WATSONX_API_KEY")
        proj = os.environ.get("WATSONX_PROJECT_ID")
        if not key or not proj:
            raise RuntimeError(
                "watsonx selected but WATSONX_API_KEY/WATSONX_PROJECT_ID unset. "
                "Use provider='stub' for the offline deterministic path.")
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials={"apikey": key,
                         "url": os.environ.get("WATSONX_URL",
                                                "https://us-south.ml.cloud.ibm.com")},
            project_id=proj)
        prompt = (f"Write one concise field briefing for utility pole {pole_id}, "
                  f"failure risk {risk:.0%}, drivers {drivers}.")
        return model.generate_text(prompt=prompt)

    if provider == "google_adk":  # pragma: no cover - documented seam
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "google_adk selected but GOOGLE_API_KEY unset. "
                "Use provider='stub' for the offline deterministic path.")
        from google.adk.agents import Agent  # type: ignore
        agent = Agent(name="vista_briefer", model="gemini-2.0-flash",
                      instruction="Write one concise utility-pole field briefing.")
        return agent.run(f"Pole {pole_id} risk {risk:.0%} drivers {drivers}")

    raise ValueError(f"Unknown LLM provider: {provider!r}")
