# CV bullets, statement-of-purpose paragraph and honest framing

## Who did what (keep this distinct)
1. **Original internship work:** an exploratory notebook built **with a friend/colleague**. It reported
   R² ≈ 1.00. You may say you *co-developed* it. Do not claim sole authorship, and do not cite its R².
2. **The rebuild (this repository):** the audit of that notebook (Part A) and the new real-data forecasting
   study (Part B), directed and reviewed by you.
3. **AI assistance:** the rebuild was implemented with an AI coding assistant. Say so if an application or
   interviewer asks about tools or AI use. Before you list the project, make sure you can explain every
   decision without notes (see [INTERVIEW_DEFENSE.md](INTERVIEW_DEFENSE.md)).

## Supportable CV bullets
Add the repository link only once it is public, and a demo link only once it is actually deployed.

> **HarvestMetric: leakage-safe crop-yield forecasting** · Python, scikit-learn, Streamlit · *github.com/devbyabir/harvestmetric*
> - Audited a crop-yield model from my internship (co-developed with a colleague) and showed its R² ≈ 1.00
>   came from data leakage. Each record appeared 10 times, so a row-level split put copies in train and test.
>   With farm-grouped validation the same model scored R² −0.21, and no model beat the mean.
> - Built a reproducible pipeline on public USDA NASS county yields (99 Iowa counties) and NASA POWER
>   weather to forecast corn yield on 1 August. It uses an untouched chronological test (2012–2025),
>   expanding-window tuning, three baselines and empirical prediction intervals checked by year.
> - Weather-informed gradient boosting lowered test RMSE by 19% against a county-trend baseline (17.6 vs
>   21.8 bu/acre), mainly in the 2012 drought. A year-level bootstrap and training-period CV did not confirm
>   the gain, so I reported it as unestablished and kept forecasts disabled.
> - Packaged the work as a tested Python package (80+ tests covering leakage, time splits, joins and the
>   app; ruff, mypy and CI), an interactive Streamlit app, and a 4-page technical report.

**One-line version:**
> Exposed train/test leakage behind an R² of 1.00 in an internship crop-yield model, then built a
> time-separated forecasting study on USDA/NASA data with strong baselines, year-level uncertainty and
> an acceptance rule written into the code before testing (Python, scikit-learn).

### Not supportable: do not write
- "Achieved R² of 0.99/1.00", "99% accurate", "predicts crop yield from IoT sensors".
- "Deployed live demo" (until a real URL exists), "used by farmers/hospitals", or any hospital link.
- "Real farm sensor data" for Part A (it is synthetic).
- "Sole author" of the internship notebook.
- "Pre-registered" (there is no independently dated preregistration) or "conformal / guaranteed
  coverage" (the intervals are empirical, from CV residuals also used for model selection).
- "Model significantly outperforms baselines". True for two baselines, not established for the strongest one.

## Statement-of-purpose paragraph (≈150 words)
> During an internship, a colleague and I built a crop-yield model that scored an R² of almost 1.0. Revisiting
> it later, I found the score was an artefact: every record appeared ten times, so the test set was full of
> copies of the training data. With the leak removed, the model was no better than predicting the average.
> That experience shaped how I approach machine learning. I rebuilt the project around a question that can
> be tested honestly: can season-to-date weather improve an August forecast of Iowa county corn yield? Using
> public USDA and NASA data, I trained on 1981–2011 and tested once on 2012–2025, against strong baselines
> and with empirical prediction intervals. The model looked better on the test years, but year-level uncertainty and
> cross-validation did not confirm it, so I reported it as unestablished. I want to study how to build
> models whose evaluation can be trusted, especially for temporal and spatial data.

Edit it into your own voice. Admissions readers notice generic text.
