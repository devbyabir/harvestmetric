"""HarvestMetric — Crop Yield Analytics (Streamlit entry point).

Run locally:  streamlit run streamlit_app.py

Part A: synthetic-data leakage case study (aggregate reports + a live simulator).
Part B: Iowa county corn yield, forecast issued 1 August (public-domain data).
Everything shown is committed to the repository; no local-only files are needed.
"""

import streamlit as st

from harvestmetric.ui import forecasts_enabled

st.set_page_config(page_title="HarvestMetric — Crop Yield Analytics", page_icon="🌾", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1200px;}
      .hm-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
                 font-weight:600; margin: 0 6px 0.5rem 0;}
      .hm-a {background:#fdecc8; color:#7a4b00;}
      .hm-b {background:#dcefe3; color:#1d5534;}
      [data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

part_b = [
    st.Page("app_pages/explore.py", title="Explore data", icon="🗺️", url_path="explore"),
    st.Page("app_pages/evaluation.py", title="Forecast evaluation", icon="📈", url_path="evaluation"),
    st.Page("app_pages/errors.py", title="Error analysis", icon="🔍", url_path="errors"),
    st.Page("app_pages/scenarios.py", title="Scenarios", icon="⚖️", url_path="scenarios"),
]
if forecasts_enabled():  # only when the real-data model passed its pre-registered gate
    part_b.append(st.Page("app_pages/forecasts.py", title="2026 forecasts", icon="🔮", url_path="forecasts"))

st.navigation({
    "": [st.Page("app_pages/overview.py", title="Overview", icon="🏠", default=True)],
    "Part A · Synthetic case study": [
        st.Page("app_pages/case_study.py", title="Leakage case study", icon="🧪", url_path="case-study"),
        st.Page("app_pages/simulator.py", title="Leakage simulator", icon="🎛️", url_path="simulator"),
    ],
    "Part B · Iowa corn (real data)": part_b,
    "About": [st.Page("app_pages/about.py", title="Data & limitations", icon="📄", url_path="about")],
}).run()
