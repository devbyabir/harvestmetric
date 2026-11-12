"""HarvestMetric — Crop Yield Analytics (Streamlit entry point).

Run locally:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from harvestmetric import config
from harvestmetric.analytics import SCENARIOS, apply_filter, compare_groups, describe_filter
from harvestmetric.data import load_dataset

st.set_page_config(page_title="HarvestMetric — Crop Yield Analytics", page_icon="🌾", layout="wide")

ACCENT = "#2f7d4f"
MUTED = "#9aa5a0"
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .hm-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
                 font-weight:600; background:#fdecc8; color:#7a4b00; margin-bottom:0.5rem;}
      .hm-note {font-size:0.9rem; opacity:0.8;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame | None, str | None]:
    try:
        df, _ = load_dataset()
        return df, None
    except FileNotFoundError as exc:
        return None, str(exc)


def read_json(path) -> dict | None:
    """Small report files: read fresh each run so retraining shows up without a restart."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def read_table(name: str) -> pd.DataFrame | None:
    path = config.TABLES_DIR / name
    return pd.read_csv(path) if path.exists() else None


df, data_error = get_data()
metrics = read_json(config.METRICS_PATH)
audit = read_json(config.AUDIT_PATH)
supported = bool(metrics and metrics["selection"]["prediction_supported"])

# ------------------------------------------------------------------ header
st.title("🌾 HarvestMetric")
st.caption("Crop Yield Analytics · leakage-safe evaluation of a public smart-farming dataset")
mode = "Prediction demo" if supported else "Interactive analytics demo — no yield predictions"
st.markdown(f'<span class="hm-badge">{mode}</span>', unsafe_allow_html=True)
st.markdown(
    "The dataset is **synthetic** (its publisher describes it as simulated). "
    "When farms are kept out of training, no model forecasts yield better than "
    "the average. This app shows the data, the corrected evaluation, and why it "
    "gives **no yield predictions**."
)
if data_error:
    st.warning(
        "Raw dataset not found, so the exploration and scenario tabs are disabled. "
        "Evaluation results below come from the saved reports. "
        "See *docs/DATA_PROVENANCE.md* for how to get the data."
    )

tabs = st.tabs(["Overview", "Explore data", "Data audit", "Model evaluation", "Scenarios", "Limitations & sources"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    if metrics:
        sel = metrics["selection"]
        res = metrics["results"]["at_sowing"]
        best, base = res[sel["selected_model"]]["test"], res[sel["best_baseline"]]["test"]
        legacy = metrics.get("legacy_split_demo", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique farms", f'{metrics["data"]["unique_rows"]:,}',
                  f'{metrics["data"]["raw_rows"]:,} rows before de-dup', delta_color="off", delta_arrow="off")
        c2.metric("Original notebook R²", f'{legacy.get("row_level_kfold_r2", float("nan")):.3f}',
                  "invalid: copies in train & test", delta_color="off", delta_arrow="off")
        c3.metric(f'Corrected test R² ({sel["selected_model"]})', f'{best["r2"]:+.3f}',
                  f'95% CI {sel["test_r2_95ci"][0]:+.3f} … {sel["test_r2_95ci"][1]:+.3f}', delta_color="off", delta_arrow="off")
        c4.metric("Test MAE vs baseline", f'{best["mae"]:,.0f} kg/ha',
                  f'baseline {base["mae"]:,.0f} kg/ha', delta_color="off", delta_arrow="off")

        st.subheader("What changed from the original analysis")
        st.markdown(
            f"""
- **Duplicates.** The CSV holds {metrics['data']['raw_rows']:,} rows but only
  {metrics['data']['unique_rows']} distinct ones (each repeated 10×). The original
  row-level split put identical copies in both training and test data, so its
  near-perfect R² measured memorisation.
- **Grouped hold-out.** {metrics['data']['test_farms']} whole farms are held out
  once as the test set; tuning uses nested, farm-grouped cross-validation on the other
  {metrics['data']['train_farms']}.
- **Defined prediction point.** The forecast happens on the *sowing date*, so
  season totals, in-season sensor readings, harvest date and growing duration are excluded.
- **Result.** With the copies kept apart, the same model type scores
  R² = {legacy.get('grouped_by_farm_kfold_r2', float('nan')):.2f}. The selected model is no
  better than chance (permutation p = {sel['permutation_test']['p_value']:.2f}).
            """
        )
    else:
        st.info("Run `python -m harvestmetric.train` to generate evaluation reports.")

# ------------------------------------------------------------------ explore
with tabs[1]:
    if df is None:
        st.info("Exploration needs the raw dataset.")
    else:
        f1, f2, f3 = st.columns(3)
        crops = f1.multiselect("Crop", config.CATEGORIES["crop_type"], default=config.CATEGORIES["crop_type"])
        regions = f2.multiselect("Region (label as given)", config.CATEGORIES["region"],
                                 default=config.CATEGORIES["region"])
        irrig = f3.multiselect("Irrigation", config.CATEGORIES["irrigation_type"],
                               default=config.CATEGORIES["irrigation_type"])
        view = df[df.crop_type.isin(crops) & df.region.isin(regions) & df.irrigation_type.isin(irrig)]
        st.caption(f"{len(view)} of {len(df)} unique farms selected · yield in kg/ha")
        if len(view) < 5:
            st.error("Fewer than 5 farms match these filters. Select more crops, regions or irrigation types.")
        else:
            a, b = st.columns(2)
            fig = px.box(view, x="crop_type", y=config.TARGET, color="crop_type", points="all",
                         labels={config.TARGET: "Yield (kg/ha)", "crop_type": "Crop"},
                         category_orders={"crop_type": config.CATEGORIES["crop_type"]})
            fig.update_layout(showlegend=False, title="Yield by crop", height=380, margin=dict(t=50, b=10))
            a.plotly_chart(fig)

            numeric = [c for c in config.NUMERIC_RANGES if c not in (config.TARGET,)]
            xcol = b.selectbox("Compare yield against", numeric, index=numeric.index("rainfall_mm"),
                               format_func=lambda c: f"{c}  [{config.UNITS.get(c, '')}]")
            r = view[[xcol, config.TARGET]].corr().iloc[0, 1]
            fig = px.scatter(view, x=xcol, y=config.TARGET, color="crop_type", opacity=0.7,
                             labels={config.TARGET: "Yield (kg/ha)", xcol: f"{xcol} ({config.UNITS.get(xcol, '')})"})
            fig.update_layout(title=f"Yield vs {xcol} · Pearson r = {r:+.3f}", height=380,
                              margin=dict(t=50, b=10), legend_title_text="")
            b.plotly_chart(fig)

            fig = px.histogram(view, x=config.TARGET, nbins=40, color_discrete_sequence=[ACCENT],
                               labels={config.TARGET: "Yield (kg/ha)"})
            fig.update_layout(title="Yield distribution (flat ≈ uniform 2000–6000 kg/ha)", height=300,
                              margin=dict(t=50, b=10), bargap=0.05)
            st.plotly_chart(fig)
            with st.expander("Show the selected farms"):
                st.dataframe(view.drop(columns=["sowing_month"]), hide_index=True)

# ------------------------------------------------------------------ audit
with tabs[2]:
    if not audit:
        st.info("Run `python -m harvestmetric.audit` to generate the audit report.")
    else:
        st.subheader("Signs that the data is synthetic")
        for f in audit["findings"]:
            st.markdown(f"- {f}")
        st.markdown(f"**Conclusion:** {audit['conclusion']}")
        g1, g2 = st.columns(2)
        geo = pd.DataFrame([{"region": k, **{kk: str(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}}
                            for k, v in audit["geography"].items()])
        g1.markdown("**Coordinates vs region label**")
        g1.dataframe(geo.rename(columns={"share_inside_plausible_box": "share in real region"}),
                     hide_index=True)
        ks = pd.Series(audit["uniform_ks_pvalues"]).rename("KS p-value vs uniform").to_frame()
        corr = pd.Series(audit["pearson_with_target"]).rename("r with yield")
        g2.markdown("**Distribution and correlation checks**")
        g2.dataframe(ks.join(corr).round(3))
        if df is not None:
            fig = px.scatter_geo(df, lat="latitude", lon="longitude", color="region", opacity=0.75,
                                 projection="natural earth", title="Where the coordinates actually point")
            fig.update_layout(height=420, margin=dict(t=50, b=0, l=0, r=0), legend_title_text="Region label")
            st.plotly_chart(fig)
            st.caption("Points labelled USA and East Africa sit in South Asia, the same box as the India labels.")

# ------------------------------------------------------------------ evaluation
with tabs[3]:
    if not metrics:
        st.info("No evaluation report found.")
    else:
        sel = metrics["selection"]
        task = st.radio("Feature set", ["at_sowing", "full_season"], horizontal=True,
                        format_func=lambda k: {"at_sowing": "At sowing (the forecasting task)",
                                               "full_season": "Full season (diagnostic upper bound)"}[k])
        st.caption(metrics["feature_sets"][task]["description"] + " Features: "
                   + ", ".join(metrics["feature_sets"][task]["features"]))
        rows = []
        for name, r in metrics["results"][task].items():
            rows.append({"Model": name, "CV R²": r["cv"]["r2"], "CV RMSE": r["cv"]["rmse"],
                         "Test R²": r["test"]["r2"], "Test MAE (kg/ha)": r["test"]["mae"],
                         "Test RMSE (kg/ha)": r["test"]["rmse"], "Tuned params": json.dumps(r["best_params"]) or ""})
        table = pd.DataFrame(rows)
        st.dataframe(table.style.format({"CV R²": "{:+.3f}", "Test R²": "{:+.3f}", "CV RMSE": "{:,.0f}",
                                         "Test MAE (kg/ha)": "{:,.0f}", "Test RMSE (kg/ha)": "{:,.0f}"}),
                     hide_index=True)

        fig = go.Figure()
        fig.add_bar(x=table["Model"], y=table["CV R²"], name="Nested grouped CV (train farms)", marker_color=MUTED)
        fig.add_bar(x=table["Model"], y=table["Test R²"], name="Held-out test farms", marker_color=ACCENT)
        legacy = metrics.get("legacy_split_demo", {})
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(barmode="group", height=360, yaxis_title="R²", title="R² by model (0 = predicting the mean)",
                          margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig)

        p = sel["permutation_test"]
        st.markdown(
            f"""
**Selected model:** {sel['selected_model']} (rule: {sel['rule']}).
Permutation test: CV R² {p['cv_r2']:+.3f} vs. shuffled-label R² 95th percentile {p['null_r2_95pct']:+.3f}
→ p = {p['p_value']:.2f}. RMSE reduction vs {sel['best_baseline']}:
{sel['rmse_vs_baseline']['rmse_reduction_mean']:+.1f} kg/ha (95% CI {sel['rmse_vs_baseline']['ci_low']:+.1f}
… {sel['rmse_vs_baseline']['ci_high']:+.1f}).
**Prediction gate: {'passed' if supported else 'not passed'}.** {sel['gate']}.
            """
        )
        c1, c2 = st.columns(2)
        by_crop, by_band = read_table("test_by_crop.csv"), read_table("test_by_yield_band.csv")
        if by_crop is not None:
            c1.markdown("**Test error by crop** (selected model)")
            c1.dataframe(by_crop.round({"r2": 3, "mae": 0, "rmse": 0, "bias": 0}), hide_index=True)
        if by_band is not None:
            c2.markdown("**Test error by actual-yield band**")
            c2.dataframe(by_band.drop(columns="r2").round(0), hide_index=True)
            st.caption("Failure pattern: predictions bunch near the average "
                       f"(spread {sel['prediction_spread']['test_pred_std']:.0f} vs actual "
                       f"{sel['prediction_spread']['test_actual_std']:.0f} kg/ha), so low-yield farms are "
                       "over-predicted and high-yield farms under-predicted, by about 1,500 kg/ha "
                       "at the extremes. R² within a narrow band is not meaningful, so it is omitted.")
        preds = read_table("test_predictions.csv")
        if preds is not None:
            fig = px.scatter(preds, x=config.TARGET, y="predicted", color="crop_type",
                             labels={config.TARGET: "Actual yield (kg/ha)", "predicted": "Predicted (kg/ha)"})
            fig.add_shape(type="line", x0=2000, y0=2000, x1=6000, y1=6000, line=dict(dash="dash", color="gray"))
            fig.update_layout(height=380, title="Held-out farms: actual vs predicted", margin=dict(t=50, b=10),
                              legend_title_text="")
            st.plotly_chart(fig)

        if legacy and "row_level_kfold_r2" in legacy:
            st.markdown("**Why the original R² was invalid**: same model, same file, two ways of splitting it")
            demo = pd.DataFrame({"Split": ["Row-level KFold (original notebook)", "Grouped by farm"],
                                 "R²": [legacy["row_level_kfold_r2"], legacy["grouped_by_farm_kfold_r2"]]})
            fig = px.bar(demo, x="R²", y="Split", orientation="h", text_auto="+.3f",
                         color="Split", color_discrete_sequence=["#c0504d", ACCENT])
            fig.update_layout(height=220, showlegend=False, margin=dict(t=10, b=10), yaxis_title="")
            st.plotly_chart(fig)

        if supported:
            st.subheader("Prediction")
            st.info("Prediction form enabled because the evaluation gate passed.")
            # Intentionally minimal: only rendered if a future dataset supports it.
            from harvestmetric.inference import InputError, predict
            with st.form("predict"):
                rec = {c: st.selectbox(c, config.CATEGORIES[c]) for c in
                       ("region", "crop_type", "irrigation_type", "fertilizer_type")}
                rec["soil_pH"] = st.number_input("soil_pH", 5.5, 7.5, 6.5)
                rec["latitude"] = st.number_input("latitude (° N)", 10.0, 35.0, 22.0)
                rec["longitude"] = st.number_input("longitude (° E)", 70.0, 90.0, 80.0)
                rec["sowing_month"] = st.number_input("sowing month", 1, 12, 2)
                if st.form_submit_button("Predict"):
                    try:
                        st.success(f"{predict(rec)[0]:,.0f} kg/ha")
                    except InputError as exc:
                        st.error(str(exc))

# ------------------------------------------------------------------ scenarios
with tabs[4]:
    if df is None:
        st.info("Scenarios need the raw dataset.")
    else:
        st.markdown("Compare average yield between two groups of farms. Each result has a "
                    "bootstrap 95% interval. These comparisons describe *this synthetic dataset*. "
                    "They are not agronomic advice.")
        choice = st.selectbox("Sample scenario", [*SCENARIOS, "Custom…"])
        if choice == "Custom…":
            cols = st.columns(2)
            specs = []
            for side, col in zip(("A", "B"), cols):
                col.markdown(f"**Group {side}**")
                spec = {}
                for field in ("crop_type", "region", "irrigation_type", "fertilizer_type", "crop_disease_status"):
                    v = col.selectbox(field.replace("_", " "), ["Any", *config.CATEGORIES[field]], key=f"{side}{field}")
                    if v != "Any":
                        spec[field] = v
                specs.append(spec)
            spec_a, spec_b = specs
        else:
            spec_a, spec_b = SCENARIOS[choice]
        ga, gb = apply_filter(df, spec_a), apply_filter(df, spec_b)
        try:
            cmp = compare_groups(ga[config.TARGET], gb[config.TARGET])
        except ValueError as exc:
            st.error(f"{exc} Group A has {len(ga)} farm(s), group B has {len(gb)}. Broaden the filters.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric(f"A: {describe_filter(spec_a)}", f"{cmp.mean_a:,.0f} kg/ha", f"n = {cmp.n_a}", delta_color="off", delta_arrow="off")
            m2.metric(f"B: {describe_filter(spec_b)}", f"{cmp.mean_b:,.0f} kg/ha", f"n = {cmp.n_b}", delta_color="off", delta_arrow="off")
            m3.metric("Difference A − B", f"{cmp.diff:+,.0f} kg/ha",
                      f"95% CI {cmp.ci_low:+,.0f} … {cmp.ci_high:+,.0f}", delta_color="off", delta_arrow="off")
            if cmp.conclusive:
                st.success("The interval excludes zero. With many comparisons, expect some of these by chance.")
            else:
                st.info("The interval includes zero: no detectable difference in this data.")
            both = pd.concat([ga.assign(group="A"), gb.assign(group="B")])
            fig = px.box(both, x="group", y=config.TARGET, color="group", points="all",
                            color_discrete_sequence=[ACCENT, MUTED], labels={config.TARGET: "Yield (kg/ha)"})
            fig.update_layout(height=380, showlegend=False, margin=dict(t=20, b=10))
            st.plotly_chart(fig)

# ------------------------------------------------------------------ limitations
with tabs[5]:
    st.markdown(
        """
### Limitations
- **Synthetic data.** The publisher describes it as simulated. Coordinates contradict
  region labels and the columns behave like independent uniform draws. No result here
  describes real farms.
- **No forecasting skill.** Under a farm-grouped hold-out, no model beats the mean
  baseline, so this app gives **no yield predictions**.
- **Ambiguous measurement timing.** The documentation does not say when soil moisture or
  NDVI were measured. They are treated as in-season and excluded from the at-sowing task.
- **Units.** Pesticide usage is "ml per day" with no area. Temperature, humidity and
  sunlight are season means.
- **Small test set.** 100 held-out farms: intervals are wide. Values are reported with 95% CIs.

### Data source
*Smart Farming Sensor Data for Yield Prediction* by Atharva Soundankar, published on Kaggle
under the Apache License 2.0 (file `Smart_Farming_Crop_Yield_2024.csv`, 500 farms). The
working copy this project started from held the same 500 records repeated 10 times.
See `docs/DATA_PROVENANCE.md` for what has and has not been verified.
        """
    )
