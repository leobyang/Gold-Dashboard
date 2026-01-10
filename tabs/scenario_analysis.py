from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Any

import numpy as np
import pandas as pd
import streamlit as st

from ollama_client import ollama_generate_json

Direction = Literal["Up", "Down", "Mixed/Unclear"]
Horizon = Literal["1W", "1M", "3M", "12M"]

GOLD = "#DEB64B"
BG = "#0E1117"


@dataclass
class DashboardContext:
    window_start: str
    window_end: str
    gold_return_pct: float
    gold_vol_daily: float
    gold_trend: str  # "up" / "down" / "flat"


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

    # simple trend: slope over last 90 points (or fewer)
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


def _build_prompt(scenario_text: str, horizon: Horizon, ctx: DashboardContext) -> str:
    """
    Strong prompt to force JSON-only output and driver framework.
    """
    schema = """
Return ONLY valid JSON with exactly these keys:
{
  "scenario_summary": string,
  "drivers": [
    {
      "name": string,
      "direction": "Up" | "Down" | "Mixed/Unclear",
      "strength": integer 0..5,
      "reasoning": string,
      "key_variables": [string, ...]
    }
  ],
  "net_view": {
    "bias": string,
    "score": integer -10..10,
    "time_horizon": "1W" | "1M" | "3M" | "12M",
    "what_would_change_my_mind": [string, ...]
  },
  "dashboard_actions": [string, ...]
}
Rules:
- Use these drivers (exactly these 3, in this order):
  1) Risk & uncertainty (safe-haven demand)
  2) Opportunity cost (real rates + USD/DXY)
  3) ETF flows / investment demand
- Do NOT invent exact numerical facts not provided.
- Base reasoning on standard macro relationships + provided dashboard context.
- Keep reasoning crisp (1-3 sentences per driver).
"""

    prompt = f"""
You are a macro/commodities analyst for a gold dashboard.

Scenario (user input):
{scenario_text}

Time horizon: {horizon}

Dashboard context (computed from the app, not to be contradicted):
{asdict(ctx)}

{schema}
"""
    return prompt.strip()


def _validate_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal validation + normalization to avoid UI crashes.
    """
    required_top = ["scenario_summary", "drivers", "net_view", "dashboard_actions"]
    for k in required_top:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["drivers"], list) or len(obj["drivers"]) != 3:
        raise ValueError("drivers must be a list of exactly 3 items.")

    # Ensure the three expected drivers exist; if not, we still render but warn.
    expected = [
        "Risk & uncertainty (safe-haven demand)",
        "Opportunity cost (real rates + USD/DXY)",
        "ETF flows / investment demand",
    ]
    # Soft check only:
    names = [d.get("name", "") for d in obj["drivers"]]
    if names != expected:
        obj["_driver_warning"] = (
            "Driver names/order differ from expected. Consider re-running."
        )

    return obj


@st.cache_data(show_spinner=False)
def analyze_scenario_cached(
    scenario_text: str,
    horizon: Horizon,
    ctx: DashboardContext,
    model: str,
) -> Dict[str, Any]:
    prompt = _build_prompt(scenario_text, horizon, ctx)
    raw = ollama_generate_json(prompt=prompt, model=model)
    return _validate_result(raw)


def render_scenario_tab(data: pd.DataFrame):
    st.subheader("Scenario Analysis (Local AI via Ollama)")

    # Quick health check: only show if Ollama is reachable
    with st.expander("Ollama status / setup", expanded=False):
        st.write("Ollama must be running locally at http://localhost:11434")
        st.code("ollama serve\nollama pull llama3.1\nollama run llama3.1 \"Return only JSON: {\\\"ok\\\": true}\"")

    gold_currencies = [c for c in data.columns if c != "date"]
    currency = st.selectbox("Gold currency", gold_currencies, index=0, key="scenario_currency")

    presets = {
        "Risk-off shock (flight to safety)":
            "Equity selloff + geopolitical risk rises sharply + recession fears increase.",
        "Fed cuts + weak USD":
            "Fed signals cuts; real yields fall; USD weakens; risk sentiment stabilizes.",
        "Sticky inflation + higher real rates":
            "Inflation remains sticky; real yields rise; USD strengthens.",
        "ETF inflow surge":
            "Large ETF inflows into gold; retail/institutional demand increases; macro steady.",
    }

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        preset = st.selectbox("Preset", ["(Custom)"] + list(presets.keys()), index=0)
    with col2:
        horizon: Horizon = st.selectbox("Horizon", ["1W", "1M", "3M", "12M"], index=1)  # type: ignore
    with col3:
        model = st.selectbox("Ollama model", ["llama3.1", "mistral", "phi3"], index=0)

    scenario_text = st.text_area(
        "Scenario input",
        value=presets.get(preset, ""),
        placeholder="Type a macro scenario: e.g., 'DXY up, real rates up 75bps, risk-off event, ETF outflows'.",
        height=120,
        key="scenario_text",
    )

    # Optional: show the computed context (good for trust)
    ctx = _compute_context(data, currency)
    with st.expander("Computed context used for analysis", expanded=False):
        st.json(asdict(ctx))

    if st.button("Generate analysis", type="primary", disabled=not scenario_text.strip()):
        try:
            with st.spinner("Running local model..."):
                result = analyze_scenario_cached(scenario_text.strip(), horizon, ctx, model=model)
        except Exception as e:
            st.error(f"Scenario analysis failed: {e}")
            st.info("If you see connection errors, make sure Ollama is running: `ollama serve`")
            return

        if "_driver_warning" in result:
            st.warning(result["_driver_warning"])

        st.markdown("### Summary")
        st.write(result["scenario_summary"])

        st.markdown("### Drivers")
        for d in result["drivers"]:
            name = d.get("name", "Driver")
            direction = d.get("direction", "Mixed/Unclear")
            strength = d.get("strength", 0)
            reasoning = d.get("reasoning", "")
            key_vars = d.get("key_variables", [])

            with st.container(border=True):
                st.write(f"**{name}** — **{direction}** (strength: {strength}/5)")
                st.write(reasoning)
                if key_vars:
                    st.caption("Key variables: " + ", ".join(key_vars))

        st.markdown("### Net view")
        nv = result["net_view"]
        st.write(f"**Bias:** {nv.get('bias','')}")
        st.write(f"**Score:** {nv.get('score','')}  (−10 bearish → +10 bullish)")
        st.write(f"**Horizon:** {nv.get('time_horizon','')}")
        wcm = nv.get("what_would_change_my_mind", [])
        if wcm:
            st.write("**What would change the view:**")
            st.write("- " + "\n- ".join(wcm))

        st.markdown("### Suggested dashboard checks")
        actions = result.get("dashboard_actions", [])
        if actions:
            st.write("- " + "\n- ".join(actions))
