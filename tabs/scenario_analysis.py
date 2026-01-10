# tabs/scenario_analysis.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ollama_client import ollama_generate_json, ollama_is_running, ollama_list_models

Direction = Literal["Up", "Down", "Mixed/Unclear"]
Horizon = Literal["1W", "1M", "3M", "12M"]

GOLD = "#DEB64B"
BG = "#0E1117"


# ---------------------------
# Scenario Library (your mentor’s examples)
# ---------------------------
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "A peaceful world": {
        "headline": "Global trade risks fade",
        "assumptions": [
            "Russia–Ukraine war eases OR Middle East conflicts recede",
            "Global supply chains normalize",
            "Risk sentiment improves; volatility declines",
        ],
        "notes": "Lower geopolitical risk typically reduces safe-haven demand for gold.",
    },
    "America is great again": {
        "headline": "US economy thrives; Fed keeps rates high while inflation stabilizes",
        "assumptions": [
            "AI productivity breakthrough supports growth",
            "Inflation stabilizes; Fed holds policy restrictive (higher-for-longer)",
            "USD remains strong on rate differentials / growth outperformance",
        ],
        "notes": "Higher real rates + stronger USD increase opportunity cost of holding gold.",
    },
    "Running out of steam": {
        "headline": "After a strong run-up, speculative positioning turns bearish",
        "assumptions": [
            "Gold up sharply over the past 12 months; momentum fades",
            "Growth weakens; consumers become price sensitive",
            "Profit-taking reduces investment demand / ETF flows",
        ],
        "notes": "Late-cycle positioning can pressure flows even if macro is mixed.",
    },
}


# ---------------------------
# Context from your dashboard data (so AI stays grounded)
# ---------------------------
@dataclass
class DashboardContext:
    window_start: str
    window_end: str
    gold_return_pct: float
    gold_vol_daily: float
    gold_trend: str


def _compute_context(data: pd.DataFrame, currency: str) -> DashboardContext:
    df = data[["date", currency]].dropna().sort_values("date").reset_index(drop=True)
    df = df.rename(columns={currency: "value"})
    df["ret"] = df["value"].pct_change()

    start = df["date"].iloc[0].date().isoformat()
    end = df["date"].iloc[-1].date().isoformat()

    start_val = float(df["value"].iloc[0])
    end_val = float(df["value"].iloc[-1])
    gold_ret_pct = (end_val / start_val - 1.0) * 100.0 if start_val else float("nan")
    vol_daily = float(df["ret"].std()) if df["ret"].dropna().shape[0] else float("nan")

    n = min(90, len(df))
    if n >= 5:
        y = df["value"].tail(n).to_numpy()
        x = np.arange(n)
        slope = np.polyfit(x, y, 1)[0]
        trend = "up" if slope > 0 else "down" if slope < 0 else "flat"
    else:
        trend = "flat"

    return DashboardContext(
        window_start=start,
        window_end=end,
        gold_return_pct=float(gold_ret_pct),
        gold_vol_daily=float(vol_daily),
        gold_trend=trend,
    )


# ---------------------------
# Prompt builder: forces mentor-style driver decomposition
# ---------------------------
def _build_prompt(
    scenario_title: str,
    scenario_headline: str,
    assumptions: List[str],
    custom_text: str,
    horizon: Horizon,
    ctx: DashboardContext,
) -> str:
    schema = """
Return ONLY valid JSON with this exact schema:

{
  "scenario": {
    "title": string,
    "headline": string,
    "assumptions": [string, ...],
    "user_input": string
  },
  "drivers": [
    {
      "name": "Risk & uncertainty (safe-haven demand)",
      "direction": "Up" | "Down" | "Mixed/Unclear",
      "strength": integer 0..5,
      "reasoning": string
    },
    {
      "name": "Opportunity cost (USD/DXY + real rates)",
      "direction": "Up" | "Down" | "Mixed/Unclear",
      "strength": integer 0..5,
      "reasoning": string
    },
    {
      "name": "ETF flows / investment demand",
      "direction": "Up" | "Down" | "Mixed/Unclear",
      "strength": integer 0..5,
      "reasoning": string
    }
  ],
  "net_view": {
    "bias": string,
    "score": integer -10..10,
    "time_horizon": "1W" | "1M" | "3M" | "12M",
    "key_risks": [string, ...],
    "what_would_change_my_mind": [string, ...]
  },
  "actions": [string, ...]
}

Rules:
- Use the assumptions as the primary facts.
- Do NOT invent precise statistics or claim specific ETF data.
- Keep reasoning crisp (1–3 sentences each).
- If the user_input contradicts assumptions, prefer user_input and mention conflict in risks.
"""

    prompt = f"""
You are a macro/commodities analyst building a driver-based scenario analysis for gold.

Scenario selected:
Title: {scenario_title}
Headline: {scenario_headline}
Assumptions:
- """ + "\n- ".join(assumptions) + f"""

User additional input (may be empty):
{custom_text}

Time horizon: {horizon}

Dashboard context (computed from app; do not contradict):
{asdict(ctx)}

{schema}
"""
    return prompt.strip()


def _driver_score(drivers: List[Dict[str, Any]]) -> int:
    """
    Deterministic score from drivers (resume-grade, makes output consistent).
    Up = +strength, Down = -strength, Mixed = 0. Clamp to [-10, 10].
    """
    s = 0
    for d in drivers:
        direction = d.get("direction", "Mixed/Unclear")
        strength = int(d.get("strength", 0) or 0)
        if direction == "Up":
            s += strength
        elif direction == "Down":
            s -= strength
    return max(-10, min(10, s))


def _validate(obj: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["scenario", "drivers", "net_view", "actions"]:
        if key not in obj:
            raise ValueError(f"Missing key: {key}")

    if not isinstance(obj["drivers"], list) or len(obj["drivers"]) != 3:
        raise ValueError("drivers must be a list of exactly 3 items")

    # Ensure driver names exist; don’t hard fail if slightly off, but normalize order if needed
    expected = [
        "Risk & uncertainty (safe-haven demand)",
        "Opportunity cost (USD/DXY + real rates)",
        "ETF flows / investment demand",
    ]

    # Light normalization: if names mismatch, keep whatever we got but warn
    names = [d.get("name", "") for d in obj["drivers"]]
    if names != expected:
        obj["_warning"] = "Driver names/order did not exactly match expected. Output may be less consistent."

    return obj


@st.cache_data(show_spinner=False)
def analyze_scenario_cached(
    scenario_title: str,
    scenario_headline: str,
    assumptions: List[str],
    custom_text: str,
    horizon: Horizon,
    ctx: DashboardContext,
    model: str,
) -> Dict[str, Any]:
    prompt = _build_prompt(
        scenario_title=scenario_title,
        scenario_headline=scenario_headline,
        assumptions=assumptions,
        custom_text=custom_text,
        horizon=horizon,
        ctx=ctx,
    )
    raw = ollama_generate_json(prompt=prompt, model=model, max_retries=1)
    out = _validate(raw)

    # Deterministic score override (optional but recommended)
    out["net_view"]["score"] = _driver_score(out["drivers"])
    out["net_view"]["time_horizon"] = horizon

    return out


def render_scenario_tab(data: pd.DataFrame):
    st.subheader("Scenario Analysis")

    # Ollama health check + helpful UX
    if not ollama_is_running():
        st.error(
            "Ollama is not reachable. Start it with `ollama serve` and make sure it listens on 127.0.0.1:11434."
        )
        st.caption("If Streamlit runs in Docker, set OLLAMA_HOST=http://host.docker.internal:11434")
        return

    # Models dropdown
    try:
        tags = ollama_list_models()
        models = [m["name"] for m in tags.get("models", [])] or ["llama3.1"]
    except Exception:
        models = ["llama3.1"]

    gold_currencies = [c for c in data.columns if c != "date"]
    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        currency = st.selectbox("Gold currency", gold_currencies, index=0, key="scen_currency")
    with c2:
        horizon: Horizon = st.selectbox("Horizon", ["1W", "1M", "3M", "12M"], index=1)  # type: ignore
    with c3:
        model = st.selectbox("Ollama model", models, index=0)

    st.markdown("### 1) Select an existing scenario")
    scenario_options = ["(Custom scenario)"] + list(SCENARIOS.keys())
    scenario_choice = st.selectbox("Scenario", scenario_options, index=1)


    if scenario_choice == "(Custom scenario)":
        st.markdown("### Define your custom scenario")

        scenario_title = st.text_input(
            "Custom title",
            value="My custom scenario",
            key="custom_scen_title",
        )

        scenario_headline = st.text_input(
            "Custom headline",
            value="Short description of the macro environment",
            key="custom_scen_headline",
        )

        assumptions_text = st.text_area(
            "Custom assumptions (one per line)",
            value=(
                "Example: DXY strengthens due to higher-for-longer Fed policy\n"
                "Example: Real yields rise 50 bps\n"
                "Example: Risk sentiment improves"
            ),
            height=140,
            key="custom_scen_assumptions",
        )

        # Parse assumptions: one per line, strip bullets
        assumptions = [
            line.strip().lstrip("-•").strip()
            for line in assumptions_text.splitlines()
            if line.strip()
        ]

        with st.expander("Assumptions preview", expanded=True):
            if assumptions:
                for a in assumptions:
                    st.write(f"- {a}")
            else:
                st.warning("Add at least 1 assumption line.")
    else:
        scenario_title = scenario_choice
        scen = SCENARIOS[scenario_title]
        scenario_headline = scen["headline"]
        assumptions = scen["assumptions"]

        st.write(f"**{scenario_title}** — {scenario_headline}")

        with st.expander("Assumptions", expanded=True):
            for a in assumptions:
                st.write(f"- {a}")
            if scen.get("notes"):
                st.caption(scen["notes"])


    st.markdown("### 2) (Optional) Add custom details")
    custom_text = st.text_area(
        "Add any extra scenario details (optional)",
        placeholder="Example: 'DXY up 3% and real yields +50bps', or 'equity selloff triggers risk-off demand'.",
        height=110,
        key="scen_custom",
    )

    ctx = _compute_context(data, currency)
    with st.expander("Computed context used (from your dashboard data)", expanded=False):
        st.json(asdict(ctx))

    if st.button("Analyze drivers", type="primary"):
        if scenario_choice == "(Custom scenario)" and len(assumptions) == 0:
            st.error("Custom scenario requires at least 1 assumption.")
            return

        try:
            with st.spinner("Running local AI analysis..."):
                result = analyze_scenario_cached(
                    scenario_title=scenario_title,
                    scenario_headline=scenario_headline,
                    assumptions=assumptions,
                    custom_text=custom_text.strip(),
                    horizon=horizon,
                    ctx=ctx,
                    model=model,
                )

        except Exception as e:
            st.error(f"Scenario analysis failed: {e}")
            st.info("Tip: confirm the model exists via `ollama list` and try again.")
            return

        if result.get("_warning"):
            st.warning(result["_warning"])

        st.markdown("## Output")

        st.markdown("### Scenario summary")
        s = result["scenario"]
        st.write(f"**{s.get('title','')}** — {s.get('headline','')}")
        st.write("**Assumptions:**")
        st.write("- " + "\n- ".join(s.get("assumptions", [])))
        if s.get("user_input"):
            st.write("**User add-on:**")
            st.write(s["user_input"])

        st.markdown("### Driver impacts (Gold)")
        drivers = result["drivers"]
        for d in drivers:
            with st.container(border=True):
                st.write(f"**{d.get('name','Driver')}** — **{d.get('direction','Mixed/Unclear')}** "
                         f"(strength: {d.get('strength',0)}/5)")
                st.write(d.get("reasoning", ""))

        st.markdown("### Net view")
        nv = result["net_view"]
        st.write(f"**Bias:** {nv.get('bias','')}")
        st.write(f"**Score:** {nv.get('score','')}  (−10 bearish → +10 bullish)")
        st.write(f"**Horizon:** {nv.get('time_horizon','')}")

        risks = nv.get("key_risks", [])
        if risks:
            st.write("**Key risks:**")
            st.write("- " + "\n- ".join(risks))

        wcm = nv.get("what_would_change_my_mind", [])
        if wcm:
            st.write("**What would change the view:**")
            st.write("- " + "\n- ".join(wcm))

        st.markdown("### Suggested dashboard checks")
        actions = result.get("actions", [])
        if actions:
            st.write("- " + "\n- ".join(actions))
        else:
            st.write("- Check Macro Comparisons: Gold vs DXY and Real Rate in the same window.")
            st.write("- Check Gold Prices: regime/volatility changes around the scenario period.")
