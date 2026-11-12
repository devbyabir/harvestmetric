"""Build the technical report (figures + HTML + PDF) from committed reports.

Run: python docs/report/build_report.py
Needs the optional tools in requirements-report.txt (matplotlib, playwright)
and a Chromium-based browser (Microsoft Edge or `playwright install chromium`).
All numbers are read from reports/; nothing is typed in by hand except prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
R = ROOT / "reports"
GREEN, GREY, RED, GOLD, INK = "#2f7d4f", "#9aa5a0", "#c0504d", "#c9a227", "#222222"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 9})


def load():
    syn = json.loads((R / "synthetic" / "metrics.json").read_text())
    real = json.loads((R / "real" / "metrics.json").read_text())
    t = {n: pd.read_csv(R / "real" / "tables" / f"{n}.csv") for n in
         ["test_predictions", "test_by_year", "geographic_check", "test_by_weather_group", "test_by_shortfall"]}
    return syn, real, t


def figures(syn, real, t) -> None:
    FIG.mkdir(exist_ok=True)
    sel = real["selection"]["selected_model"]

    # Fig 1: Part A
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.45})
    leg = syn["legacy_split_demo"]
    ax[0].barh(["Grouped by farm", "Row-level KFold\n(original)"],
               [leg["grouped_by_farm_kfold_r2"], leg["row_level_kfold_r2"]], color=[GREEN, RED])
    for i, v in enumerate([leg["grouped_by_farm_kfold_r2"], leg["row_level_kfold_r2"]]):
        if v > 0.3:
            ax[0].text(v / 2, i, f"{v:+.3f}", va="center", ha="center", color="white", fontweight="bold")
        else:
            ax[0].text(0.03, i, f"{v:+.3f}", va="center", ha="left", color=INK, fontweight="bold")
    ax[0].set_xlim(-0.35, 1.05)
    ax[0].axvline(0, color=INK, lw=0.8)
    ax[0].set_title("(a) Same model and data, two splits")
    ax[0].set_xlabel("5-fold R²")
    res = syn["results"]["at_sowing"]
    names = list(res)
    ax[1].bar(range(len(names)), [res[n]["test"]["r2"] for n in names], color=GREY)
    ax[1].axhline(0, color=INK, lw=0.8)
    ax[1].set_xticks(range(len(names)), [n.replace(" ", "\n", 1) for n in names], fontsize=7)
    ax[1].set_ylabel("Test R² (100 farms)")
    ax[1].set_title("(b) Corrected: no model beats the mean")
    ax[1].set_ylim(-0.06, 0.02)
    fig.savefig(FIG / "fig1_partA.png")
    plt.close(fig)

    # Fig 2: Part B test years
    tp, by = t["test_predictions"], t["test_by_year"]
    g = tp.groupby("year").mean(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax[0].plot(g.index, g["yield_bu_ac"], "o-", color=INK, ms=3, label="Actual (county mean)")
    ax[0].plot(g.index, g["pred_County trend"], "--", color=GREY, label="County-trend baseline")
    ax[0].plot(g.index, g[f"pred_{sel}"], "o-", color=GREEN, ms=3, label=f"{sel}")
    ax[0].set_ylabel("bu/acre")
    ax[0].set_title("(a) Test years, forecast on 1 August")
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    w = 0.38
    ax[1].bar(by.year - w / 2, by.trend_rmse, w, color=GREY, label="County trend")
    ax[1].bar(by.year + w / 2, by.model_rmse, w, color=GREEN, label=sel)
    ax[1].set_ylabel("RMSE (bu/acre)")
    ax[1].set_title("(b) Error by test year")
    ax[1].legend(fontsize=6.5, frameon=False)
    for a in ax:
        a.set_xticks(range(2012, 2026, 2))
    fig.savefig(FIG / "fig2_partB_test.png")
    plt.close(fig)

    # Fig 3: intervals + geographic
    geo = t["geographic_check"].sort_values("trend_rmse")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.42})
    ax[0].plot(by.year, by.in80, "o-", color=GREEN, ms=3, label="80% interval")
    ax[0].plot(by.year, by.in90, "o-", color=GOLD, ms=3, label="90% interval")
    ax[0].axhline(0.8, color=GREEN, ls=":", lw=1)
    ax[0].axhline(0.9, color=GOLD, ls=":", lw=1)
    ax[0].set_ylim(0.6, 1.02)
    ax[0].set_ylabel("Share of counties covered")
    ax[0].set_title("(a) Interval coverage by test year")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax[0].set_xticks(range(2012, 2026, 2))
    y = range(len(geo))
    ax[1].barh([i + 0.2 for i in y], geo.trend_rmse, 0.4, color=GREY, label="County trend")
    ax[1].barh([i - 0.2 for i in y], geo.model_rmse, 0.4, color=GREEN, label=sel)
    ax[1].set_yticks(list(y), [d.title() for d in geo.held_out_district], fontsize=7)
    ax[1].set_xlabel("RMSE (bu/acre), held-out district")
    ax[1].set_title("(b) Held-out districts (tuned in-fold)")
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    fig.savefig(FIG / "fig3_intervals_geo.png")
    plt.close(fig)


def table(rows, header, cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def ci(lo: float, hi: float) -> str:
    return f"[{lo:+.2f}, {hi:+.2f}]"


def html(syn, real, t) -> str:
    sel = real["selection"]
    res = real["results"]
    m, tr = res[sel["selected_model"]], res["County trend"]
    vs = sel["vs_baselines_year_bootstrap"]
    iv, geo, roll = real["intervals"], real["geographic_check"], real["rolling_origin_refit"]
    leg, ssel = syn["legacy_split_demo"], syn["selection"]
    sres = syn["results"]["at_sowing"]
    wg = t["test_by_weather_group"].set_index("weather_group")
    sf = t["test_by_shortfall"].set_index("shortfall")
    by = t["test_by_year"].set_index("year")
    heat = wg.loc[[i for i in wg.index if i.startswith("Extreme heat")][0]]
    short = sf.loc[[i for i in sf.index if i.startswith("Yield >15%")][0]]

    model_rows = [[n, f"{r['cv']['rmse']:.1f}", f"{r['test']['mae']:.1f}", f"{r['test']['rmse']:.1f}",
                   f"{r['test']['r2']:+.2f}"] for n, r in res.items()]
    gain_rows = [[b, f"{v['rmse_gain']:+.2f}", ci(v["ci_low"], v["ci_high"])] for b, v in vs.items()]
    syn_rows = [[n, f"{r['cv']['r2']:+.3f}", f"{r['test']['r2']:+.3f}", f"{r['test']['mae']:,.0f}"]
                for n, r in sres.items()]

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>HarvestMetric technical report</title>
<style>
@page {{ size: A4; margin: 16mm 16mm 16mm 16mm; }}
body {{ font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #222; }}
h1 {{ font-size: 17pt; margin: 0 0 2px; }} .sub {{ color: #555; margin: 0 0 8px; font-size: 9pt; }}
h2 {{ break-after: avoid; font-size: 11.5pt; margin: 12px 0 4px; color: #1d5534; border-bottom: 1px solid #cfe3d6; padding-bottom: 2px; }}
h3 {{ font-size: 10pt; margin: 8px 0 2px; }}
p {{ margin: 3px 0 5px; text-align: justify; }} ul {{ margin: 2px 0 6px 16px; padding: 0; }} li {{ margin: 1px 0; }}
table {{ border-collapse: collapse; margin: 4px 0 6px; font-size: 8.6pt; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 2px 5px; text-align: right; }}
th:first-child, td:first-child, table.text th, table.text td {{ text-align: left; }} thead th {{ background: #f1f6f2; }}
.abstract {{ background: #f6f8f6; border-left: 3px solid #2f7d4f; padding: 6px 9px; margin: 6px 0 8px; }}
figure {{ margin: 6px 0 8px; page-break-inside: avoid; }} figure img {{ width: 100%; }}
figcaption {{ font-size: 8.3pt; color: #444; }} .two {{ display: flex; gap: 14px; }} .two > div {{ flex: 1; }}
.small {{ font-size: 8.2pt; color: #444; }} .ref p {{ font-size: 8.2pt; margin: 1px 0; text-align: left; }}
.keep {{ page-break-inside: avoid; }}
</style></head><body>
<h1>HarvestMetric: leakage-safe evaluation of crop-yield models</h1>
<p class="sub">Abir Ahmmed · Technical report · November 2026 · github.com/devbyabir/harvestmetric</p>

<div class="abstract"><b>Abstract.</b> An earlier crop-yield notebook reported R² ≈ 1.00. I show that the
score came from duplicated records leaking across a row-level train/test split of a synthetic dataset.
With farms kept apart, the same model scores R² = {leg['grouped_by_farm_kfold_r2']:.2f}, and no tuned model
beats the mean (permutation p = {ssel['permutation_test']['p_value']:.2f}). I then pose a defensible forecasting
question on real public data: Iowa county corn-for-grain yield, forecast on 1 August from the county's past
yields and season-to-date weather. On untouched test years 2012–2025, gradient boosting on trend
deviations lowered RMSE from {tr['test']['rmse']:.1f} to {m['test']['rmse']:.1f} bu/acre against a
county-trend baseline. However, the gain's year-resampled 95% interval reaches zero
({ci(vs['County trend']['ci_low'], vs['County trend']['ci_high'])}), and in training-period
cross-validation the model did not beat the baseline. Under an acceptance rule set in the code before the test
scores were computed (not independently preregistered), the result is reported
as promising but not established, and no live forecasts are published.</div>

<h2>1 Research questions</h2>
<p><b>Q1</b> (Part A): Was the original R² ≈ 1.00 evidence of forecasting ability? <b>Q2</b> (Part B): On real
observational data, does season-to-date weather improve a 1 August forecast of county corn yield beyond
simple baselines built from the county's own yield history? Does any improvement hold across years and
across regions?</p>

<h2>2 Data and licensing</h2>
{table([
 ["A", "Smart Farming Sensor Data for Yield Prediction (A. Soundankar, Kaggle)", "Apache-2.0; synthetic", "Not redistributed; aggregates only"],
 ["B", "USDA NASS Quick Stats: Iowa county corn-for-grain yield, bu/acre, 1926–2025", "CC0 1.0", "Processed panel committed"],
 ["B", "NASA POWER daily point API v2.10.3: T2M_MAX, T2M_MIN, PRECTOTCORR, 1981–2026", "Citation required", "Features in panel"],
 ["B", "US Census Bureau 2024 Gazetteer: county internal points", "US Gov. work (17 U.S.C. §105)", "Coordinates in panel"],
], ["Part", "Source", "Terms", "In repository"], cls="text")}
<p class="small">The original working copy (5,000 rows) was verified record-identical to the official Kaggle file
(500 rows): dates were reformatted and every record repeated 10 times. Retrieval dates, query parameters and
SHA-256 checksums are in <i>data/processed/manifest.json</i>. Raw downloads are re-created by
<i>python -m harvestmetric.real.acquire</i>.</p>

<h2>3 Part A: the leakage case study</h2>
<p>With each record present ten times, a row-level <i>KFold</i> places near-certain copies of every test row in
training, and flexible models memorise them. I removed exact duplicates, held out 100 farms (20%,
stratified by crop) once, and tuned five models with nested, farm-grouped CV on the remaining 400. The
forecast point is the sowing date. Season totals, in-season sensor readings, harvest date and growing
duration are excluded. A permutation test and bootstrap intervals judge skill. An audit found the data's
coordinates contradict its region labels and its columns behave as independent uniform draws. The
publisher describes it as simulated.</p>
<figure><img src="figures/fig1_partA.png"><figcaption><b>Figure 1.</b> (a) The original split's R² is an
artefact of duplication. (b) Under a farm-grouped hold-out, test R² is ≈ 0 for every model.</figcaption></figure>
<div class="two"><div>{table(syn_rows, ["Model (at sowing)", "CV R²", "Test R²", "Test MAE kg/ha"])}</div>
<div><p><b>Answer to Q1: no.</b> Test R² for the CV-selected model is {sres[ssel['selected_model']]['test']['r2']:+.3f}
(95% CI [{ssel['test_r2_95ci'][0]:+.3f}, {ssel['test_r2_95ci'][1]:+.3f}]). Predictions collapse to the mean, so
low- and high-yield farms are missed by about 1,500 kg/ha. Valid evaluation shows the data contains no
learnable yield signal, so no prediction feature is offered.</p></div></div>

<h2>4 Part B: forecasting task</h2>
<p><b>Unit:</b> Iowa county × year (99 counties). <b>Target:</b> NASS final corn-for-grain yield (bu/acre;
1 bu/ac ≈ 0.0627 t/ha). <b>Forecast date:</b> 1 August of year t. <b>Information allowed:</b> the county's
published yields for t−30…t−1, and NASA POWER weather from 1 April to 31 July of year t at the county's
Census internal point. <b>Features:</b> a county linear trend fitted to the previous 30 years, last
year's deviation from that trend (flagged when unpublished), May–July growing degree-days (10/30 °C),
June–July degree-days above 30 °C, April–May, June and July precipitation, and mean July maximum
temperature. <b>Leakage controls</b>, enforced in code and tests: weather after 31 July cannot enter a feature,
history uses only years before t, no feature uses another county's same-year yield, and identifiers
are never inputs. Combined "other counties" rows and withheld values are dropped. Counties not published
after 2014 are left unscored rather than imputed.</p>

<h2>5 Validation design and models</h2>
<ul>
<li><b>Chronological test:</b> train 1981–2011 ({real['data']['train_rows']:,} county-years); untouched test
2012–2025 ({real['data']['test_rows']:,}). The test years were scored once, after model selection.</li>
<li><b>Tuning:</b> expanding-window CV inside the training years (validation blocks 1997–99, …, 2009–11,
each fitted only on earlier years). Selection used CV RMSE.</li>
<li><b>Baselines:</b> the county's 30-year mean, its previous-year yield, and its 30-year linear trend.</li>
<li><b>Models:</b> Ridge and histogram gradient boosting, both predicting the deviation from the county trend.
Ridge captures linear weather effects; boosting allows heat and drought thresholds.</li>
<li><b>Uncertainty:</b> empirical 80%/90% intervals, ± an upper quantile of the selected model's time-ordered
CV residuals. Those residuals also drove model selection and are correlated within seasons, so these are
not split-conformal intervals and carry no coverage guarantee; coverage is checked on the test years. Gains are
tested with a bootstrap that resamples whole <i>years</i>, because one season's weather shock hits all
counties at once.</li>
<li><b>Acceptance gate</b> (written into the code before the test scores were computed, by the author's
account; not an independently dated preregistration): forecasts are published only if the model beats the
best baseline in CV
<i>and</i> its test RMSE gain over the strongest baseline has a 95% interval above zero.</li>
<li><b>Geographic check (separate):</b> leave one of NASS's nine crop-reporting districts out, within 1981–2011.
Model family and hyperparameters are re-selected inside each fold by the same expanding-window CV on the
other eight districts.</li>
</ul>

<h2>6 Results</h2>
<div class="two keep"><div>{table(model_rows, ["Model", "CV RMSE", "Test MAE", "Test RMSE", "Test R²"])}
<p class="small">bu/acre; test = 2012–2025. Selected by CV: {sel['selected_model']}.</p></div>
<div>{table(gain_rows, ["Gain of selected model over", "RMSE gain", "95% CI (years)"])}
<p class="small">Positive = selected model better. Rolling annual refit: RMSE {roll['model']['rmse']:.1f} vs
{roll['trend']['rmse']:.1f}, CI {ci(roll['vs_trend_year_bootstrap']['ci_low'],
roll['vs_trend_year_bootstrap']['ci_high'])}.</p></div></div>
<figure><img src="figures/fig2_partB_test.png"><figcaption><b>Figure 2.</b> (a) County-average actual yield and
1 August forecasts. (b) RMSE by test year: most of the gain comes from 2012.</figcaption></figure>
<p>The model clearly beats the historical-mean and previous-year baselines. Its gain over the county trend
({vs['County trend']['rmse_gain']:+.1f} bu/acre, {100 * (1 - m['test']['rmse'] / tr['test']['rmse']):.0f}% lower RMSE)
is positive in {100 * vs['County trend']['share_resamples_better']:.1f}% of year resamples, but the interval reaches
zero. In CV the model was <i>worse</i> than the trend ({m['cv']['rmse']:.1f} vs {tr['cv']['rmse']:.1f}). The early folds
learned from only three disaster years (1983, 1988, 1993) and over-predicted heat losses in 1997, 1999 and
2002, when yields were above trend despite high heat. The weather–yield relationship is not stable across
eras, so the gate is <b>not passed</b>. The empirical intervals covered {iv['80%']['test_coverage']:.1%} (80%) and
{iv['90%']['test_coverage']:.1%} (90%) of test county-years overall (half-widths ±{iv['80%']['half_width_bu_ac']:.0f} and
±{iv['90%']['half_width_bu_ac']:.0f} bu/acre), but only {by.loc[2012, 'in80']:.0%} and {by.loc[2020, 'in80']:.0%}
at 80% in 2012 and 2020. Errors within a shock year move together, and overall coverage above nominal is not
a guarantee. In the geographic check, tuned inside each fold, the model
beat the trend in {geo['districts_where_model_beats_trend']} of 9 held-out districts (RMSE {geo['model']['rmse']:.1f} vs
{geo['trend']['rmse']:.1f}). The same seasons appear on both sides there, so this shows spatial transfer, not
skill on new seasons.</p>
<figure><img src="figures/fig3_intervals_geo.png"><figcaption><b>Figure 3.</b> (a) Interval coverage collapses in
shock years (2012 drought, 2020 derecho). (b) Leave-one-district-out RMSE, 1981–2011; model and hyperparameters re-tuned without the held-out district. Intervals are empirical, with no coverage guarantee.</figcaption></figure>

<h2>7 Error analysis</h2>
<ul>
<li><b>Extreme heat</b> (June–July degree-days above the training 90th percentile; defined from weather only):
model RMSE {heat['model_rmse']:.1f} vs trend {heat['trend_rmse']:.1f}; bias +{heat['model_bias']:.1f} vs +{heat['trend_bias']:.1f}.
Weather helps most where the trend fails most.</li>
<li><b>Large shortfalls</b> (yield &gt;15% below trend; outcome-defined, descriptive only, n = {int(short['n'])}):
still under-forecast severity, bias +{short['model_bias']:.1f} bu/acre (trend +{short['trend_bias']:.1f}).</li>
<li><b>2020:</b> the 10 August derecho damaged central-Iowa corn after the forecast date. No 1 August
information could anticipate it (bias +{by.loc[2020, 'model_bias']:.1f} bu/acre).</li>
<li>Permutation importance and SHAP-style explanations were <b>not</b> produced, because the model did not pass
validation. Associations in this data are not causal effects.</li>
</ul>

<h2>8 Limitations</h2>
<p>Weather is taken at a single point per county from a ~0.5° × 0.625° grid. Neighbouring counties often share
a cell, and the point ignores where corn is grown. Targets are NASS's current estimates, which can differ from
values published in real time. Unpublished counties after 2014 are missing not at random. Fourteen test
years give wide intervals. The model set was deliberately small, and richer inputs (soil, planting dates,
satellite vegetation indices) were out of scope. Part A uses synthetic data and says nothing about real crops.</p>

<h2>9 Conclusion</h2>
<p>The original R² ≈ 1.00 was leakage. On real data, a carefully time-separated evaluation finds that
season-to-date weather <i>probably</i> improves 1 August county yield forecasts in drought years, but the
evidence does not meet the acceptance rule set in the code: CV across 1997–2011 disagrees, and the test gain is
not reliably above zero. The contribution is a reproducible, leakage-tested pipeline that reports this
honestly, rather than a headline accuracy number.</p>

<p class="small"><b>Authorship note.</b> The original exploratory notebook was developed during an internship
together with a colleague and is not included. This rebuild, data pipeline, evaluation and report were
produced by the author with the assistance of an AI coding assistant.</p>

<h2>References</h2>
<div class="ref">
<p>Kaufman, S., Rosset, S., Perlich, C., &amp; Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. <i>ACM TKDD</i>, 6(4), 15.</p>
<p>Lobell, D. B., Roberts, M. J., Schlenker, W., et al. (2014). Greater sensitivity to drought accompanies maize yield increase in the U.S. Midwest. <i>Science</i>, 344(6183), 516–519.</p>
<p>Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. <i>Ecography</i>, 40(8), 913–929.</p>
<p>Schlenker, W., &amp; Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. <i>PNAS</i>, 106(37), 15594–15598.</p>
<p>Soundankar, A. (2025). Smart Farming Sensor Data for Yield Prediction [dataset]. Kaggle. Apache-2.0.</p>
<p>USDA National Agricultural Statistics Service. Quick Stats database, quickstats.nass.usda.gov (retrieved 12 Nov 2026).</p>
<p>NASA Langley Research Center POWER Project, daily point API v2.10.3, power.larc.nasa.gov (retrieved 12 Nov 2026). These data were obtained from the NASA LaRC POWER Project funded through the NASA Earth Science/Applied Science Program.</p>
<p>US Census Bureau. 2024 Gazetteer Files: counties (retrieved 12 Nov 2026).</p>
<p>NOAA National Centers for Environmental Information. Billion-Dollar Weather and Climate Disasters: Central US derecho, 10 August 2020.</p>
</div>
</body></html>"""


def pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:  # noqa: BLE001 - fall back to Playwright's bundled Chromium
            browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.pdf(path=str(pdf_path), format="A4", print_background=True,
                 margin={"top": "14mm", "bottom": "14mm", "left": "14mm", "right": "14mm"})
        browser.close()


def main() -> None:
    syn, real, t = load()
    figures(syn, real, t)
    out_html = OUT / "HarvestMetric_technical_report.html"
    out_html.write_text(html(syn, real, t), encoding="utf-8")
    if "--no-pdf" not in sys.argv:
        pdf(out_html, OUT / "HarvestMetric_technical_report.pdf")
    print(f"Wrote {out_html.relative_to(ROOT)} and the PDF")


if __name__ == "__main__":
    main()
