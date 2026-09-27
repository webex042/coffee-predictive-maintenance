"""
dashboard/app.py  --  Phase 7: the Streamlit dashboard
======================================================

This is the face of the whole project: a live-looking control room for the
(simulated) coffee line. It does NO machine learning itself -- it just loads
the saved model and the shared scoring code (src/scoring.py) and draws the
results. That separation keeps the dashboard simple and guarantees it shows
exactly what the model was trained and evaluated to do.

Run it with:
    streamlit run dashboard/app.py

Everything shown here is built from SIMULATED data. See the disclaimer footer.
"""

import os
import sys
import io

# When Streamlit runs this file directly, the project root isn't on the import
# path yet, so `import config` / `from src import ...` would fail. Add it.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config
from src import scoring
from dashboard import theme

# The order machines appear in along the real production process.
PROCESS_ORDER = ["Roaster", "Grinder", "Extraction",
                 "SprayDryer", "Conveyor", "Packaging"]

st.set_page_config(page_title="Coffee Line — Predictive Maintenance",
                   page_icon="☕", layout="wide")

# Paint the retro-arcade skin (CSS only -- nothing below computes differently).
theme.inject_theme()


# ---------------------------------------------------------------------------
# Data loading (cached so we don't reload the model / re-score on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle():
    """Load the saved model bundle once and keep it in memory."""
    return scoring.load_bundle()


@st.cache_data(show_spinner="Scoring machines...")
def get_scored(csv_bytes):
    """
    Score a dataset and cache the result.

    `csv_bytes` is None for the built-in simulated data, or the raw bytes of an
    uploaded CSV. Because the cache key is those bytes, we only re-score when
    the data actually changes -- moving the time-slider does NOT re-score.
    """
    bundle = get_bundle()
    if csv_bytes is None:
        raw = scoring.load_raw()
    else:
        raw = pd.read_csv(io.BytesIO(csv_bytes))
    scored = scoring.score_history(raw, bundle)
    return raw, scored


bundle = get_bundle()

# ---------------------------------------------------------------------------
# Header + disclaimer
# ---------------------------------------------------------------------------
theme.marquee(
    "☕ PREDICTIVE MAINTENANCE",
    "Coffee production line — predicting the chance each machine fails within "
    "the next 24 operating hours, so the crew can fix the right machine before "
    "it breaks.",
)
theme.ticker(config.DISCLAIMER)

# ---------------------------------------------------------------------------
# Sidebar: choose the data source (built-in demo data, or upload your own)
# ---------------------------------------------------------------------------
st.sidebar.header("📤 Data source")
source = st.sidebar.radio(
    "Which data should the dashboard use?",
    ["Built-in simulated data", "Upload my own CSV"],
)

csv_bytes = None
if source == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("Upload a sensor CSV", type=["csv"])
    if uploaded is not None:
        csv_bytes = uploaded.getvalue()

    with st.sidebar.expander("What format does my CSV need?"):
        st.write("One row per machine per hour, with these columns:")
        st.code(", ".join(scoring.REQUIRED_RAW_COLUMNS))
        st.caption(
            "machine_type must be one of: "
            + ", ".join(sorted(config.MACHINE_BASELINES.keys())) + ". "
            "The model was trained on the simulated factory, so use the same "
            "machine types, sensor units and ~hourly spacing or the scores "
            "won't be meaningful. Label columns are not needed."
        )
        st.download_button("⬇️ Download a template CSV",
                           data=scoring.make_template_csv(),
                           file_name="sensor_template.csv", mime="text/csv")

# Validate an uploaded file BEFORE trying to score it, and show friendly errors.
if source == "Upload my own CSV":
    if csv_bytes is None:
        st.warning("⬅️ Upload a CSV in the sidebar, or switch back to "
                   "built-in data to explore the demo.")
        st.stop()
    problems = scoring.validate_raw(pd.read_csv(io.BytesIO(csv_bytes)))
    if problems:
        st.error("This CSV can't be scored yet — please fix:")
        for problem in problems:
            st.write("• " + problem)
        st.stop()

# Score the chosen data (cached: only re-runs when the data changes).
raw, scored = get_scored(csv_bytes)
if scored.empty:
    st.error("No rows could be scored — after dropping warm-up rows there was "
             "nothing left. Please give each machine more hourly history.")
    st.stop()

if source == "Upload my own CSV":
    st.success(f"✅ Scored your data: {scored['machine_id'].nunique()} machines, "
               f"{len(scored):,} usable rows.")

# ---------------------------------------------------------------------------
# Sidebar: a time machine + model facts
# ---------------------------------------------------------------------------
min_hour = int(scored["operating_hours"].min())
max_hour = int(scored["operating_hours"].max())

st.sidebar.header("🕒 Control room clock")
st.sidebar.caption(
    "Slide to pretend 'now' is any past hour of the 6-month simulation. "
    "Everything below updates to that moment."
)
as_of_hour = st.sidebar.slider(
    "Current operating hour", min_hour, max_hour, max_hour, step=1
)
day = as_of_hour // 24
hour_of_day = as_of_hour % 24
st.sidebar.write(f"**Day {day}, {hour_of_day:02d}:00** (hour {as_of_hour})")

st.sidebar.divider()
st.sidebar.header("🤖 Model")
metrics = bundle.get("metrics", {})
st.sidebar.write(f"**{bundle['model_name']}** (calibrated)")
st.sidebar.write(f"Alert threshold: `{bundle['threshold']:.3f}` "
                 "(tuned to favour recall — better to over-warn than miss a failure)")
if metrics:
    st.sidebar.write(
        f"Test recall **{metrics.get('Recall', 0):.2f}**, "
        f"precision **{metrics.get('Precision', 0):.2f}**, "
        f"PR-AUC **{metrics.get('PR_AUC', 0):.2f}**"
    )

# ---------------------------------------------------------------------------
# Build the "right now" snapshot for the chosen hour
# ---------------------------------------------------------------------------
snap = scoring.latest_snapshot(scored, bundle, explain=True, as_of_hour=as_of_hour)
pq = scoring.priority_table(snap)

# ---------------------------------------------------------------------------
# KPI tiles across the top
# ---------------------------------------------------------------------------
n_machines = len(snap)
n_at_risk = int((snap["failure_probability"] >= bundle["threshold"]).sum())
avg_health = float(snap["health_score"].mean())
total_expected_loss = float(pq["expected_units_lost"].sum())

theme.scoreboard([
    {"label": "MACHINES", "value": n_machines},
    {"label": "AT RISK 24H", "value": n_at_risk,
     "kind": "alert" if n_at_risk else None},
    {"label": "AVG HEALTH", "value": f"{avg_health:.0f}", "unit": "/100",
     "kind": "hero"},
    {"label": "UNITS AT RISK", "value": f"{total_expected_loss:,.0f}"},
])
st.caption("‘Units at risk’ = sum over machines of "
           "P(failure) × (units/hour × downtime hours). A risk-weighted, "
           "placeholder estimate of production on the line.")

st.divider()

# ---------------------------------------------------------------------------
# A small helper to draw one coloured machine "card"
# ---------------------------------------------------------------------------
def machine_card_html(row):
    """Return HTML for one machine's arcade 'unit module' card.

    Dark panel with a neon left edge in the machine's health colour; a
    predicted failure adds a pulsing magenta DANGER led.

    The health colour is applied through a per-band CSS class
    (``pm-unit--Healthy`` etc., defined in theme.py) rather than an inline
    style, because Streamlit's HTML sanitizer strips inline CSS custom
    properties — a plain ``class`` attribute always survives.
    """
    band = row["health_band"]
    prob_pct = row["failure_probability"] * 100
    is_alert = row["failure_probability"] >= bundle["threshold"]
    led = '<span class="pm-unit__led"></span>' if is_alert else ""
    return (
        f'<div class="pm-unit pm-unit--{band}">'
        f'<div class="pm-unit__top">'
        f'<span class="pm-unit__id">{row["machine_id"]}</span>{led}</div>'
        f'<div class="pm-unit__type">{row["machine_type"]}</div>'
        f'<div class="pm-unit__score">{row["health_score"]:.0f}'
        f'<small>/100</small></div>'
        f'<div class="pm-unit__band">{band} · '
        f'P(fail) {prob_pct:.0f}%</div>'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Three tabs: line overview, one-machine detail, and the priority queue
# ---------------------------------------------------------------------------
tab_overview, tab_detail, tab_priority = st.tabs(
    ["🏭 Line overview", "🔎 Machine detail", "🚨 Priority queue"]
)

# ===== TAB 1: the production line, drawn stage by stage ====================
with tab_overview:
    st.subheader("Production line status")
    st.caption("Each machine is coloured by its health band. A pulsing magenta "
               "light = the model predicts a failure within 24h. Coffee flows "
               "left → right.")

    # One column per process stage, machines stacked inside.
    stage_columns = st.columns(len(PROCESS_ORDER))
    for col, stage in zip(stage_columns, PROCESS_ORDER):
        with col:
            st.markdown(f'<div class="pm-stage">{stage}</div>',
                        unsafe_allow_html=True)
            stage_rows = snap[snap["machine_type"] == stage]
            for _, row in stage_rows.iterrows():
                st.markdown(machine_card_html(row), unsafe_allow_html=True)

    # A small colour legend so the bands are self-explanatory.
    st.markdown("---")
    legend_bits = []
    for band in config.HEALTH_BANDS:
        legend_bits.append(
            f"<span style='background:{theme.neon_colour(band['name'])};"
            f"color:#0a0e27;padding:2px 10px;border-radius:6px;margin-right:6px;"
            f"font-weight:700;font-size:.82rem;'>"
            f"{band['name']} ≥ {band['min_score']}</span>"
        )
    st.markdown("**Health bands:** " + " ".join(legend_bits),
                unsafe_allow_html=True)

# ===== TAB 2: everything about one machine =================================
with tab_detail:
    machine_ids = sorted(snap["machine_id"].tolist())
    selected = st.selectbox("Choose a machine", machine_ids)
    row = snap[snap["machine_id"] == selected].iloc[0]

    # --- current status line ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Health score", f"{row['health_score']:.0f}/100", row["health_band"])
    c2.metric("P(failure next 24h)", f"{row['failure_probability'] * 100:.1f}%")
    predicted = row["failure_probability"] >= bundle["threshold"]
    c3.metric("Model verdict", "AT RISK" if predicted else "OK")

    # --- current raw sensor readings ---
    st.markdown("**Current sensor readings** (this hour)")
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "motor_current"]
    sensor_view = pd.DataFrame({
        "sensor": sensor_cols,
        "reading": [round(float(row[s]), 2) for s in sensor_cols],
    })
    st.dataframe(sensor_view, hide_index=True, use_container_width=True)

    # --- plain-language recommendation ---
    if row["rec_urgent"]:
        st.warning(f"**Recommendation:** {row['rec_headline']}\n\n{row['rec_action']}")
    else:
        st.success(f"**Recommendation:** {row['rec_headline']}\n\n{row['rec_action']}")

    # --- SHAP explanation: WHY did the model give this probability? ---
    st.markdown("#### Why the model said this")
    st.caption("SHAP shows how each feature pushed *this machine's* prediction. "
               "Magenta bars pushed **toward** failure; cyan bars pushed **away**.")

    # Pull this machine's latest full feature row (at or before the chosen hour).
    hist = scored[(scored["machine_id"] == selected) &
                  (scored["operating_hours"] <= as_of_hour)]
    latest_full = hist.iloc[-1]
    X_row = latest_full[bundle["feature_columns"]].to_frame().T

    try:
        shap_values, _ = scoring.compute_shap(
            bundle["base_model"], X_row, bundle["feature_columns"]
        )
        contributions = pd.DataFrame({
            "feature": bundle["feature_columns"],
            "shap": shap_values[0],
        })
        # Show the 8 features with the biggest push (either direction).
        contributions["abs"] = contributions["shap"].abs()
        top = contributions.sort_values("abs", ascending=False).head(8)
        top = top.sort_values("shap")   # for a tidy horizontal bar order

        bar_colours = ["#ff2e97" if v > 0 else "#33e1ff" for v in top["shap"]]
        fig = go.Figure(go.Bar(
            x=top["shap"], y=top["feature"], orientation="h",
            marker_color=bar_colours,
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="push toward failure  →")
        theme.style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as err:
        st.caption(f"(SHAP explanation unavailable: {err})")

    # --- trend charts: the last 72 hours leading up to 'now' ---
    st.markdown("#### Recent history (last 72 operating hours)")
    recent = hist.tail(72)

    # Chart 1: how far each key sensor has drifted from normal (z-scores).
    fig_z = go.Figure()
    for zcol, label in [("vibration_zscore", "vibration"),
                        ("temperature_zscore", "temperature"),
                        ("pressure_zscore", "pressure")]:
        fig_z.add_trace(go.Scatter(
            x=recent["operating_hours"], y=recent[zcol],
            mode="lines", name=label,
        ))
    fig_z.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                        title="Sensor deviation from normal (sigma)",
                        xaxis_title="operating hour", yaxis_title="z-score")
    theme.style_fig(fig_z)
    st.plotly_chart(fig_z, use_container_width=True)

    # Chart 2: the model's failure probability, with the alert threshold line.
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=recent["operating_hours"], y=recent["failure_probability"],
        mode="lines", name="P(failure)", line=dict(color="#ff2e97"),
    ))
    fig_p.add_hline(y=bundle["threshold"], line_dash="dash",
                    line_color="#ffd21e", annotation_text="alert threshold")
    fig_p.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                        title="Predicted failure probability",
                        xaxis_title="operating hour", yaxis_title="P(failure)")
    theme.style_fig(fig_p)
    st.plotly_chart(fig_p, use_container_width=True)

# __APPEND_HERE__

# ===== TAB 3: the priority queue ==========================================
with tab_priority:
    st.subheader("🏆 Fix these first")
    st.caption("Ranked by **expected units lost** = P(failure) × "
               "(units/hour × downtime hours). This chases the biggest "
               "risk-weighted production loss, not just the highest probability.")

    view = pq.copy()
    view["failure_probability"] = (view["failure_probability"] * 100).round(1)
    view["expected_units_lost"] = view["expected_units_lost"].round(0)
    view = view.rename(columns={
        "priority_rank": "rank",
        "machine_id": "machine",
        "machine_type": "type",
        "failure_probability": "P(fail) %",
        "health_score": "health",
        "health_band": "band",
        "expected_units_lost": "exp. units lost",
    })
    show_cols = ["rank", "machine", "type", "P(fail) %", "health", "band",
                 "exp. units lost"]
    st.dataframe(view[show_cols], hide_index=True, use_container_width=True)

    st.caption("Production-impact numbers are placeholders defined in config.py, "
               "not real plant figures.")

# ---------------------------------------------------------------------------
# Footer disclaimer (always visible)
# ---------------------------------------------------------------------------
st.divider()
st.caption("☕ " + config.DISCLAIMER)







