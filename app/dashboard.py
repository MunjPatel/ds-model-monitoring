from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # local shared/ for standalone + Streamlit Cloud

from shared.plot_theme import COLORS, apply_plotly_theme  # noqa: E402

st.set_page_config(page_title="Model Health Monitor", layout="wide")
st.markdown(
    f"<style>.stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}</style>",
    unsafe_allow_html=True,
)
st.title("Model Health Monitor")
st.caption("Michelangelo-style prediction logs · outcome joins · PSI drift alerts")

out = ROOT / "outputs"
if not (out / "meta.json").exists():
    st.warning("Outputs missing. Run `python 06_ds_model_monitoring/scripts/run_mvp.py` first.")
    st.stop()

meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
perf = pd.read_csv(out / "monthly_performance.csv")
psi = pd.read_csv(out / "psi_by_month.csv")
alerts = pd.read_csv(out / "alerts.csv") if (out / "alerts.csv").exists() else pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train rows", f"{meta['n_train']:,}")
c2.metric("Serve rows", f"{meta['n_serve']:,}")
c3.metric("Alerts (injected drift)", meta["n_alerts_on_injected_drift"])
c4.metric("Injected score PSI", f"{meta.get('injected_score_psi', float('nan')):.3f}")

tab1, tab2, tab3 = st.tabs(["Performance over time", "Feature PSI", "Alerts / incident"])

with tab1:
    fig = px.line(perf, x="serve_month", y="auc", markers=True, title="AUC by serve month")
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.line(perf, x="serve_month", y="brier", markers=True, title="Brier score by serve month")
    apply_plotly_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(perf, use_container_width=True)

with tab2:
    heat = psi.pivot_table(index="feature", columns="serve_month", values="psi")
    fig = px.imshow(heat, aspect="auto", title="Feature PSI heatmap vs reference month", color_continuous_scale="Teal")
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PSI ≥ 0.10 warn, ≥ 0.25 fail (see monitoring/rules.yaml).")

with tab3:
    st.subheader("Alerts from injected drift test")
    st.dataframe(alerts, use_container_width=True)
    if (out / "INCIDENT_POSTMORTEM.md").exists():
        st.markdown((out / "INCIDENT_POSTMORTEM.md").read_text(encoding="utf-8"))
    st.info(
        "Uber Michelangelo monitors production by logging predictions, joining outcomes, "
        "and tracking feature/prediction distributions when labels lag."
    )
