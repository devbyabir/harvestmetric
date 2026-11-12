"""Page implementations for the Streamlit app (one function per page).

Each file in app_pages/ calls one of these. Data is reloaded on every page run
so the app always reflects the committed reports (and tests can redirect the
data location with environment variables).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import app_data
from .real import config as rcfg

GREEN, GREY, RED, GOLD = "#2f7d4f", "#9aa5a0", "#c0504d", "#c9a227"

# ------------------------------------------------------------------ data (all committed)
syn_metrics: dict | None = None
syn_audit: dict | None = None
real_metrics: dict | None = None
panel: pd.DataFrame | None = None
real_supported = False


def refresh() -> None:
    """(Re)load committed reports and the panel; called at the start of every page run."""
    global syn_metrics, syn_audit, real_metrics, panel, real_supported
    syn_metrics = app_data.read_json("synthetic", "metrics.json")
    syn_audit = app_data.read_json("synthetic", "data_audit.json")
    real_metrics = app_data.read_json("real", "metrics.json")
    panel = app_data.read_panel()
    real_supported = bool(real_metrics and real_metrics["selection"]["prediction_supported"])


def badge(part: str) -> None:
    label = {"a": "Part A · synthetic leakage case study", "b": "Part B · real data: Iowa county corn yield"}[part]
    st.markdown(f'<span class="hm-badge hm-{part}">{label}</span>', unsafe_allow_html=True)


def missing(what: str) -> None:
    st.warning(f"{what} is not available in this deployment. Run the pipeline commands in the README "
               "to regenerate it.")


def fig_layout(fig, height=360, title=None):
    fig.update_layout(height=height, margin=dict(t=50 if title else 20, b=10, l=10, r=10), title=title,
                      legend=dict(orientation="h", y=-0.22, title_text=""))
    return fig


# ================================================================== pages
def overview() -> None:
    refresh()
    st.title("🌾 HarvestMetric")
    st.caption("Crop-yield analytics with leakage-safe evaluation · two parts, kept separate")
    st.markdown(
        "**In 30 seconds:** an earlier crop-yield notebook reported R² ≈ 1.00. **Part A** shows that score "
        "came from duplicated rows leaking across the train/test split of a *synthetic* dataset, and that "
        "with leakage removed no model beats the average. **Part B** asks the question properly on "
        "**real public data**: can Iowa county corn yield be forecast on 1 August from the season's "
        "weather so far? It uses a strict chronological test on 2012–2025, strong baselines and calibrated "
        "intervals.")
    a, b = st.columns(2)
    with a:
        badge("a")
        if syn_metrics:
            legacy = syn_metrics["legacy_split_demo"]
            sel = syn_metrics["selection"]
            best = syn_metrics["results"]["at_sowing"][sel["selected_model"]]["test"]["r2"]
            st.metric("Original notebook R² (row-level split)", f"{legacy['row_level_kfold_r2']:.3f}",
                      "invalid: copies in train and test", delta_color="off", delta_arrow="off")
            st.metric("Same model, split by farm", f"{legacy['grouped_by_farm_kfold_r2']:+.2f}",
                      f"best tuned model on held-out farms: R² {best:+.3f}", delta_color="off", delta_arrow="off")
        else:
            missing("The Part A report")
    with b:
        badge("b")
        if real_metrics:
            sel = real_metrics["selection"]
            res = real_metrics["results"]
            m, t = res[sel["selected_model"]]["test"], res["County trend"]["test"]
            gain = sel["vs_baselines_year_bootstrap"]["County trend"]
            st.metric("Test RMSE 2012–2025 (bu/ac)", f"{m['rmse']:.1f}",
                      f"county-trend baseline {t['rmse']:.1f} · {100 * (1 - m['rmse'] / t['rmse']):.0f}% lower",
                      delta_color="off", delta_arrow="off")
            st.metric("Gain over trend, 95% CI (years resampled)",
                      f"{gain['ci_low']:+.1f} … {gain['ci_high']:+.1f}",
                      "not reliably above 0 → forecasts not enabled" if not real_supported else "above 0",
                      delta_color="off", delta_arrow="off")
        else:
            missing("The Part B report")

    if real_metrics:
        iv = real_metrics["intervals"]
        st.subheader("Bottom line")
        st.markdown(
            f"""
- **Part A:** the original R² was an artefact. The synthetic data has no learnable signal.
- **Part B:** weather-informed gradient boosting cut test RMSE from {t['rmse']:.1f} to {m['rmse']:.1f}
  bu/ac. Most of the gain came in the 2012 drought. But the gain's 95% interval touches zero, and in
  training-period cross-validation the model did **not** beat the trend baseline. By the rule fixed
  before testing, the app therefore shows **evaluation results, not live forecasts**.
- Intervals: 80% / 90% conformal intervals covered {iv['80%']['test_coverage']:.0%} /
  {iv['90%']['test_coverage']:.0%} of test county-years. Coverage drops in shock years.
            """)
        tp = app_data.read_table("real", "test_predictions.csv")
        if tp is not None:
            g = tp.groupby("year").mean(numeric_only=True).reset_index()
            fig = go.Figure()
            fig.add_scatter(x=g.year, y=g[rcfg.TARGET], name="Actual (mean of counties)", mode="lines+markers",
                            line=dict(color="black"))
            fig.add_scatter(x=g.year, y=g["pred_County trend"], name="County-trend baseline", mode="lines",
                            line=dict(color=GREY, dash="dash"))
            fig.add_scatter(x=g.year, y=g[f"pred_{sel['selected_model']}"],
                            name=f"{sel['selected_model']} (1 Aug)", mode="lines+markers", line=dict(color=GREEN))
            fig.update_yaxes(title="bu/acre")
            st.plotly_chart(fig_layout(fig, 330, "Untouched test years: county average, actual vs forecast"))


def case_study() -> None:
    refresh()
    badge("a")
    st.header("Why the original R² ≈ 1.00 was invalid")
    if not syn_metrics:
        missing("The Part A report")
        return
    d, legacy, sel = syn_metrics["data"], syn_metrics["legacy_split_demo"], syn_metrics["selection"]
    st.markdown(
        f"""
The dataset is the public **synthetic** *Smart Farming Sensor Data for Yield Prediction* (Kaggle,
Apache-2.0; described by its publisher as simulated). The original notebook's working copy repeated every
one of its {d['unique_rows']} records {d['original_working_copy_rows'] // d['unique_rows']} times
(verified record-identical to the official file). A row-level split therefore put identical copies on
both sides, and models simply memorised them.
        """)
    demo = pd.DataFrame({"Split": ["Row-level KFold (original)", "Grouped by farm (corrected)"],
                         "R²": [legacy["row_level_kfold_r2"], legacy["grouped_by_farm_kfold_r2"]]})
    fig = px.bar(demo, x="R²", y="Split", orientation="h", text_auto="+.3f", color="Split",
                 color_discrete_sequence=[RED, GREEN])
    fig.update_layout(showlegend=False, yaxis_title="")
    st.plotly_chart(fig_layout(fig, 220, "Same random forest, same 5,000 rows, two splits"))

    st.subheader("Corrected evaluation (100 untouched farms)")
    task = st.radio("Feature set", ["at_sowing", "full_season"], horizontal=True,
                    format_func=lambda k: {"at_sowing": "At sowing (forecast)",
                                           "full_season": "Full season (diagnostic)"}[k])
    rows = [{"Model": n, "CV R²": r["cv"]["r2"], "Test R²": r["test"]["r2"], "Test MAE (kg/ha)": r["test"]["mae"],
             "Test RMSE (kg/ha)": r["test"]["rmse"]} for n, r in syn_metrics["results"][task].items()]
    st.dataframe(pd.DataFrame(rows).style.format({"CV R²": "{:+.3f}", "Test R²": "{:+.3f}",
                                                  "Test MAE (kg/ha)": "{:,.0f}", "Test RMSE (kg/ha)": "{:,.0f}"}),
                 hide_index=True)
    p = sel["permutation_test"]
    st.markdown(f"Permutation test for the selected model ({sel['selected_model']}): p = {p['p_value']:.2f}. "
                f"Test R² 95% CI {sel['test_r2_95ci'][0]:+.3f} … {sel['test_r2_95ci'][1]:+.3f}. "
                "**No prediction form**: no model beats predicting the mean.")
    if syn_audit:
        with st.expander("Data audit: signs the data is synthetic"):
            for f in syn_audit["findings"]:
                st.markdown(f"- {f}")
    c1, c2 = st.columns(2)
    crop = app_data.read_table("synthetic", "test_by_crop.csv")
    band = app_data.read_table("synthetic", "test_by_yield_band.csv")
    if crop is not None:
        c1.markdown("**Test error by crop**")
        c1.dataframe(crop.round({"r2": 3, "mae": 0, "rmse": 0, "bias": 0}), hide_index=True)
    if band is not None:
        c2.markdown("**Test error by actual-yield band**: predictions collapse to the mean")
        c2.dataframe(band.drop(columns="r2").round(0), hide_index=True)


@st.cache_data(show_spinner=False)
def _simulate(n_farms: int, copies: int, signal: float, seed: int):
    from .synthetic.simulator import simulate
    return simulate(n_farms=n_farms, copies=copies, signal=signal, seed=seed)


def simulator() -> None:
    refresh()
    badge("a")
    st.header("Leakage simulator")
    st.markdown("Generate a fresh dataset, duplicate each record, and compare a row-level split with a split "
                "that keeps each farm on one side. The data is generated here, so no external file is used.")
    c1, c2, c3 = st.columns(3)
    n_farms = c1.slider("Distinct farms", 100, 600, 300, 50)
    copies = c2.slider("Copies of each record", 1, 20, 10)
    signal = c3.slider("True signal (share of variance explainable)", 0.0, 0.8, 0.0, 0.1)
    r = _simulate(n_farms, copies, signal, 0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Row-level KFold R²", f"{r.row_level_r2:+.3f}")
    m2.metric("Grouped KFold R²", f"{r.grouped_r2:+.3f}")
    m3.metric("Best achievable R² (oracle)", f"{r.true_signal_r2:+.3f}")
    vals = (r.row_level_r2, r.grouped_r2, r.true_signal_r2)
    fig = go.Figure(go.Bar(x=["Row-level", "Grouped", "Oracle"], y=vals, marker_color=[RED, GREEN, GREY],
                           text=[f"{v:+.2f}" for v in vals]))
    fig.update_yaxes(range=[min(-0.3, r.grouped_r2 - 0.1), 1.05], title="R²")
    st.plotly_chart(fig_layout(fig, 300))
    gap = r.row_level_r2 - r.grouped_r2
    if copies > 1 and gap > 0.2:
        st.error(f"Leakage: the row-level split overstates R² by {gap:.2f}, even above the oracle "
                 "ceiling. It is memorising duplicated records.")
    else:
        st.success("No material gap: without duplicates, row-level and grouped splits agree.")


def _pick(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def explore_real() -> None:
    refresh()
    badge("b")
    st.header("Explore Iowa county corn yields and weather")
    if panel is None:
        missing("The Part B panel (data/processed/iowa_corn_county_panel.csv)")
        return
    obs = panel[panel[rcfg.TARGET].notna()].copy()
    obs["dev_pct"] = 100 * (obs[rcfg.TARGET] - obs.trend_pred) / obs.trend_pred
    c1, c2 = st.columns([2, 1])
    years = sorted(obs.year.unique())
    year = c2.slider("Season", int(years[0]), int(years[-1]), 2012 if 2012 in years else int(years[-1]))
    metric = c2.radio("Colour counties by", ["Yield (bu/ac)", "Deviation from pre-season trend (%)"])
    col = rcfg.TARGET if metric.startswith("Yield") else "dev_pct"
    yr = obs[obs.year == year]
    c2.caption(f"{len(yr)} counties published by NASS in {year}. Unpublished counties are not shown.")
    if yr.empty:
        c1.info("No published county yields for this season.")
    else:
        fig = px.scatter_geo(yr, lat="lat", lon="lon", color=col, hover_name="county", scope="usa",
                             color_continuous_scale="RdYlGn",
                             range_color=(None if col == rcfg.TARGET else (-50, 50)),
                             hover_data={"district": True, rcfg.TARGET: ":.1f", "dev_pct": ":.1f",
                                         "lat": False, "lon": False})
        fig.update_traces(marker=dict(size=12, line=dict(width=0.5, color="white")))
        fig.update_geos(scope="north america", projection_type="mercator", lataxis_range=[40.3, 43.6],
                        lonaxis_range=[-96.8, -89.9], showsubunits=True, subunitcolor="#999",
                        showcountries=False, showland=True, landcolor="#f4f4f1", showlakes=False)
        fig.update_layout(coloraxis_colorbar_title_text="bu/ac" if col == rcfg.TARGET else "% vs trend")
        c1.plotly_chart(fig_layout(fig, 400, f"Iowa counties, {year}"))

    counties = sorted(obs.county.unique())
    county = st.selectbox("Highlight a county", counties, index=_pick(counties, "STORY"))
    state = obs.groupby("year")[rcfg.TARGET].mean()
    cty = obs[obs.county == county].set_index("year")
    fig = go.Figure()
    fig.add_scatter(x=state.index, y=state.values, name="Iowa county average", line=dict(color=GREY))
    fig.add_scatter(x=cty.index, y=cty[rcfg.TARGET], name=f"{county}: published yield", mode="lines+markers",
                    line=dict(color=GREEN))
    fig.add_scatter(x=cty.index, y=cty.trend_pred, name=f"{county}: pre-season trend",
                    line=dict(color=GOLD, dash="dot"))
    fig.add_vrect(x0=rcfg.TEST_START - 0.5, x1=rcfg.LAST_YIELD_YEAR + 0.5, fillcolor=GREEN, opacity=0.06,
                  annotation_text="test years", annotation_position="top left")
    fig.update_yaxes(title="bu/acre")
    st.plotly_chart(fig_layout(fig, 340, "Yield over time"))

    labels: dict[str, str] = {"edd30_jun_jul": "Extreme heat: Jun–Jul degree-days above 30 °C",
              "gdd_may_jul": "Growing degree-days May–Jul", "prcp_jun": "June rainfall (mm)",
              "prcp_jul": "July rainfall (mm)", "prcp_apr_may": "Apr–May rainfall (mm)",
              "tmax_jul": "Mean July max temperature (°C)"}
    x: str = st.selectbox("Weather variable (season to 31 July)", list(labels), format_func=lambda k: labels[k])
    r = float(np.corrcoef(obs[x], obs["dev_pct"])[0, 1])
    period = np.where(obs.year >= rcfg.TEST_START, "2012–2025", "1981–2011")
    fig = px.scatter(obs.assign(period=period), x=x, y="dev_pct", color="period", opacity=0.35,
                     labels={x: labels[x], "dev_pct": "Yield deviation from trend (%)"},
                     color_discrete_map={"2012–2025": GREEN, "1981–2011": GREY})
    st.plotly_chart(fig_layout(fig, 360, f"Association, not causation · Pearson r = {r:+.2f}"))


def evaluation_real() -> None:
    refresh()
    badge("b")
    st.header("Forecast evaluation")
    if not real_metrics:
        missing("The Part B report")
        return
    task, data, sel = real_metrics["task"], real_metrics["data"], real_metrics["selection"]
    st.info(f"**Task.** Unit: {task['unit']}. Target: {task['target']}. Forecast date: {task['forecast_date']}. "
            f"Information used: {task['information']}. Training {data['train_years'][0]}–{data['train_years'][1]} "
            f"({data['train_rows']:,} county-years); untouched test {data['test_years'][0]}–{data['test_years'][1]} "
            f"({data['test_rows']:,} county-years).")
    rows = [{"Model": n, "CV RMSE": r["cv"]["rmse"], "CV R²": r["cv"]["r2"], "Test MAE": r["test"]["mae"],
             "Test RMSE": r["test"]["rmse"], "Test R²": r["test"]["r2"],
             "Tuned": json.dumps(r["params"]) if r["params"] else "—"} for n, r in real_metrics["results"].items()]
    st.markdown("**1 · Time-based test** (bu/acre; CV = expanding-window validation inside 1981–2011)")
    fmt: dict[Any, Any] = {c: "{:.2f}" for c in ["CV RMSE", "Test MAE", "Test RMSE"]}
    fmt |= {"CV R²": "{:+.3f}", "Test R²": "{:+.3f}"}
    st.dataframe(pd.DataFrame(rows).style.format(fmt), hide_index=True)
    vs = pd.DataFrame([{"Baseline": b, "RMSE gain (bu/ac)": v["rmse_gain"], "95% CI low": v["ci_low"],
                        "95% CI high": v["ci_high"], "Share of resamples better": v["share_resamples_better"]}
                       for b, v in sel["vs_baselines_year_bootstrap"].items()])
    st.markdown(f"**Selected by CV: {sel['selected_model']}.** Gain over each baseline on the test years "
                "(bootstrap resampling whole years, since counties in one season share its weather):")
    vs_fmt: dict[Any, Any] = {c: "{:+.2f}" for c in ["RMSE gain (bu/ac)", "95% CI low", "95% CI high"]}
    vs_fmt["Share of resamples better"] = "{:.1%}"
    st.dataframe(vs.style.format(vs_fmt), hide_index=True)
    res = real_metrics["results"]
    st.markdown(f"**Forecast gate: {'passed' if real_supported else 'not passed'}.** Rule fixed before testing: "
                f"{sel['gate']}. In CV the selected model's RMSE ({res[sel['selected_model']]['cv']['rmse']:.2f}) "
                f"was worse than the trend baseline's ({res['County trend']['cv']['rmse']:.2f}), and the test "
                "gain's interval reaches zero.")

    by_year = app_data.read_table("real", "test_by_year.csv")
    if by_year is not None:
        fig = go.Figure()
        fig.add_bar(x=by_year.year, y=by_year.trend_rmse, name="County trend", marker_color=GREY)
        fig.add_bar(x=by_year.year, y=by_year.model_rmse, name=sel["selected_model"], marker_color=GREEN)
        fig.update_layout(barmode="group", yaxis_title="RMSE (bu/ac)")
        st.plotly_chart(fig_layout(fig, 320, "Test RMSE by year"))
        st.markdown("**Prediction intervals** (split conformal, calibrated on time-ordered CV residuals)")
        iv = real_metrics["intervals"]
        c1, c2 = st.columns(2)
        c1.metric("80% interval: coverage on test", f"{iv['80%']['test_coverage']:.1%}",
                  f"± {iv['80%']['half_width_bu_ac']:.1f} bu/ac", delta_color="off", delta_arrow="off")
        c2.metric("90% interval: coverage on test", f"{iv['90%']['test_coverage']:.1%}",
                  f"± {iv['90%']['half_width_bu_ac']:.1f} bu/ac", delta_color="off", delta_arrow="off")
        fig = go.Figure()
        fig.add_scatter(x=by_year.year, y=by_year.in80, name="80% interval", mode="lines+markers",
                        line=dict(color=GREEN))
        fig.add_scatter(x=by_year.year, y=by_year.in90, name="90% interval", mode="lines+markers",
                        line=dict(color=GOLD))
        fig.add_hline(y=0.8, line_dash="dot", line_color=GREEN)
        fig.add_hline(y=0.9, line_dash="dot", line_color=GOLD)
        fig.update_yaxes(title="share of counties covered", range=[0.5, 1.02], tickformat=".0%")
        st.plotly_chart(fig_layout(fig, 300, "Coverage by test year: one weather shock hits every county at once"))

    roll = real_metrics["rolling_origin_refit"]
    st.markdown(f"**Rolling-origin check** (refit every year on all earlier years): RMSE {roll['model']['rmse']:.2f} "
                f"vs trend {roll['trend']['rmse']:.2f}; gain 95% CI {roll['vs_trend_year_bootstrap']['ci_low']:+.2f} "
                f"… {roll['vs_trend_year_bootstrap']['ci_high']:+.2f}.")

    st.divider()
    geo, g = app_data.read_table("real", "geographic_check.csv"), real_metrics["geographic_check"]
    st.markdown("**2 · Geographic generalisation (separate check)**: leave one NASS crop-reporting district out, "
                "within training years only. The same seasons appear on both sides, so this measures transfer "
                "across space, not forecasting of new seasons.")
    if geo is not None:
        fig = go.Figure()
        fig.add_bar(y=geo.held_out_district, x=geo.trend_rmse, name="County trend", orientation="h",
                    marker_color=GREY)
        fig.add_bar(y=geo.held_out_district, x=geo.model_rmse, name=sel["selected_model"], orientation="h",
                    marker_color=GREEN)
        fig.update_layout(barmode="group", xaxis_title="RMSE (bu/ac)")
        st.plotly_chart(fig_layout(fig, 380, f"Held-out district RMSE · model better in "
                                              f"{g['districts_where_model_beats_trend']} of 9 districts"))


def errors_real() -> None:
    refresh()
    badge("b")
    st.header("Error analysis and backtest viewer")
    tp = app_data.read_table("real", "test_predictions.csv")
    if tp is None or not real_metrics:
        missing("The Part B test predictions")
        return
    sel = real_metrics["selection"]["selected_model"]
    pc = f"pred_{sel}"
    c1 = c2 = st.container()
    wg = app_data.read_table("real", "test_by_weather_group.csv")
    sf = app_data.read_table("real", "test_by_shortfall.csv")
    show = {"n": "n", "model_rmse": "Model RMSE", "trend_rmse": "Trend RMSE", "model_bias": "Model bias",
            "trend_bias": "Trend bias"}
    if wg is not None:
        c1.markdown("**By weather group** (defined from weather only; thresholds from training years)")
        c1.dataframe(wg[["weather_group", *show]].rename(columns=show | {"weather_group": "Weather group"})
                     .round(1), hide_index=True)
    if sf is not None:
        c2.markdown("**By outcome** (defined from the actual yield, descriptive only)")
        c2.dataframe(sf[["shortfall", *show]].rename(columns=show | {"shortfall": "Outcome group"}).round(1),
                     hide_index=True)
    st.caption("Bias = forecast − actual. Positive bias means the forecast was too high.")
    st.markdown("""
**Known failure modes**
- **2012 drought:** the model cut the trend baseline's over-forecast from about +46 to +15 bu/ac,
  but still over-forecast.
- **2020:** the 10 August derecho flattened central-Iowa corn *after* the forecast date. No information
  available on 1 August could anticipate it, so errors were large.
- **Unpublished counties:** NASS did not publish every county after 2014. Those county-years have
  no target and are not scored.
    """)
    err = tp[pc] - tp[rcfg.TARGET]
    worst = tp.assign(err=err).loc[err.abs().sort_values(ascending=False).index]
    with st.expander("Largest test errors (county-years)"):
        st.dataframe(worst[["year", "county", "district", rcfg.TARGET, pc, "pred_County trend", "err",
                            "weather_group"]].head(15).round(1), hide_index=True)

    st.subheader("Backtest viewer")
    st.caption("Forecasts the evaluated model (trained on 1981–2011) would have issued on 1 August of each "
               "test year, with 80% intervals. These are evaluation results, not live forecasts.")
    counties = sorted(tp.county.unique())
    county = st.selectbox("County", counties, index=_pick(counties, "STORY"))
    c = tp[tp.county == county]
    fig = go.Figure()
    fig.add_scatter(x=c.year, y=c.hi80, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=c.year, y=c.lo80, mode="lines", fill="tonexty", fillcolor="rgba(47,125,79,0.15)",
                    line=dict(width=0), name="80% interval")
    fig.add_scatter(x=c.year, y=c[pc], name="Model forecast (1 Aug)", line=dict(color=GREEN))
    fig.add_scatter(x=c.year, y=c["pred_County trend"], name="Trend baseline", line=dict(color=GREY, dash="dash"))
    fig.add_scatter(x=c.year, y=c[rcfg.TARGET], name="Published yield", mode="markers",
                    marker=dict(color="black", size=8))
    fig.update_yaxes(title="bu/acre")
    st.plotly_chart(fig_layout(fig, 360, county.title()))


def scenarios_real() -> None:
    refresh()
    badge("b")
    st.header("Scenarios: how did yields respond to season conditions?")
    if panel is None:
        missing("The Part B panel (data/processed/iowa_corn_county_panel.csv)")
        return
    from .real.analysis import compare_by_year_bootstrap, with_deviation
    d = with_deviation(panel)
    st.markdown("Compare the yield deviation from each county's pre-season trend between two groups of historical "
                "county-years. Intervals resample whole seasons. These are **associations in observed data**, "
                "not causal effects or forecasts.")
    presets = {
        "Extreme June–July heat vs the rest": ("edd30_jun_jul", 0.9, "above"),
        "Dry June–July vs the rest": ("prcp_jun_jul", 0.1, "below"),
        "Very wet June–July vs the rest": ("prcp_jun_jul", 0.9, "above"),
        "Cool July (low max temperature) vs the rest": ("tmax_jul", 0.1, "below"),
    }
    choice = st.selectbox("Scenario", list(presets))
    var, q0, side = presets[choice]
    q = st.slider("Percentile threshold", 0.05, 0.95, q0, 0.05,
                  help="Threshold computed over all county-years 1981–2025")
    thr = d[var].quantile(q)
    group = d[var] > thr if side == "above" else d[var] < thr
    period = st.radio("Period", ["All years 1981–2025", "1981–2011", "2012–2025"], horizontal=True)
    if period != "All years 1981–2025":
        lo, hi = (1981, 2011) if period == "1981–2011" else (2012, 2025)
        keep = d.year.between(lo, hi)
        d, group = d[keep], group[keep]
    try:
        c = compare_by_year_bootstrap(d, group, ~group)
    except ValueError as exc:
        st.error(f"{exc} Move the threshold toward the middle, or choose a longer period.")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Group ({side} P{int(round(q * 100))})", f"{c.mean_a:+.1f}%",
              f"{c.n_a} county-years, {c.years_a} seasons", delta_color="off", delta_arrow="off")
    m2.metric("Rest", f"{c.mean_b:+.1f}%", f"{c.n_b} county-years", delta_color="off", delta_arrow="off")
    m3.metric("Difference", f"{c.diff:+.1f} pts", f"95% CI {c.ci_low:+.1f} … {c.ci_high:+.1f}",
              delta_color="off", delta_arrow="off")
    if c.conclusive:
        st.success("The interval excludes zero for this historical comparison.")
    else:
        st.info("The interval includes zero: no clear difference in this data.")
    fig = px.box(d.assign(group=np.where(group, "Group", "Rest")), x="group", y="dev_pct", color="group",
                 points=False, color_discrete_map={"Group": GOLD, "Rest": GREY},
                 labels={"dev_pct": "Deviation from trend (%)"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig_layout(fig, 340))


def forecasts_real() -> None:
    refresh()
    badge("b")
    st.header("2026 forecasts")
    fc = app_data.read_table("real", "forecast_2026.csv")
    if fc is None:
        st.info("No forecasts are published because the evaluation gate did not pass.")
        return
    st.dataframe(fc.round(1), hide_index=True)


def about() -> None:
    refresh()
    st.header("Data, licences and limitations")
    st.markdown(
        """
**Part A data:** *Smart Farming Sensor Data for Yield Prediction* by Atharva Soundankar (Kaggle,
Apache-2.0). It is synthetic. This app shows only aggregate results; the file is not redistributed.

**Part B data**, processed panel committed to the repository:
- USDA NASS Quick Stats: Iowa county corn-for-grain yield, bu/acre. CC0 1.0 per the data.gov catalog.
  *This product uses NASS data but is not endorsed or certified by NASS.*
- NASA POWER daily point data (T2M_MAX, T2M_MIN, PRECTOTCORR) at each county's Census internal point.
  *These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the
  NASA Earth Science/Applied Science Program.*
- US Census Bureau 2024 Gazetteer county internal points (US government work).

**Limitations**
- Weather is sampled at one point per county from a grid of about 0.5° × 0.625°, so neighbouring
  counties often share a grid cell. That approximates neither field conditions nor where corn is grown.
- Targets are NASS's *current* published estimates. Values available in real time may have been revised since.
- Counties NASS did not publish (2015 onward) are missing, and not at random.
- The weather–yield relationship is not stable across eras. Several late-1990s and 2002 seasons had high
  heat at the weather point yet above-trend yields. That is why training-period CV and test results
  disagree. The cause (management, hybrids, or point-weather error) was not established.
- The 14 test years are few: year-level uncertainty dominates.

Full documentation: `docs/` and the technical report `docs/report/HarvestMetric_technical_report.pdf`.
        """)


def forecasts_enabled() -> bool:
    refresh()
    return real_supported
