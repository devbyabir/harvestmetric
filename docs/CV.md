# CV wording

## What is and is not supportable

**Supportable** (the repository shows it):

- You audited an earlier crop-yield notebook and found that its near-perfect R² came
  from duplicated rows leaking across a row-level train/test split.
- You rebuilt the pipeline with farm-grouped, nested cross-validation, an untouched
  test set, a defined prediction point, baselines, a permutation test and bootstrap
  intervals.
- You showed that the dataset (public, synthetic) supports no yield forecast, and
  shipped an analytics app instead of a misleading predictor.
- You wrote a tested Python package and a Streamlit dashboard.

**Not supportable:**

- "Built a crop-yield prediction model with 99% / 1.00 R² accuracy". The figure is an artefact of leakage.
- "Predicts crop yield from IoT sensor data". No valid predictive skill was found.
- "Real-world farm data" or any hospital link. The data is synthetic.
- "Sole author of the original analysis". A friend contributed, and the split of work is undocumented.
- "Deployed live demo" before an actual URL exists.

## Suggested bullets

Use these once the repository is public. Add the live-demo link only after it is deployed.

> **HarvestMetric: Crop Yield Analytics** (Python, scikit-learn, Streamlit) · *github.com/devbyabir/harvestmetric-crop-yield*
> - Audited an earlier crop-yield model and found that its R² of 1.00 came from data
>   leakage: each of 500 records appeared 10 times, so a row-level split put copies in both train and test.
> - Rebuilt the evaluation with farm-grouped nested cross-validation, an untouched 100-farm
>   test set and an explicit at-sowing prediction point. With leakage removed, the same
>   model type scored R² −0.21. No model beat a mean baseline (permutation p = 0.55).
> - Showed with a statistical audit that the public dataset is synthetic (coordinates
>   contradict regions, features are independent uniform draws). Delivered an
>   honest analytics dashboard in Streamlit instead of a misleading predictor.
> - Packaged validation, features, training, evaluation and inference as a tested Python
>   module (43 pytest tests covering leakage, input validation and inference).

Shorter, one line:

> Found and fixed train/test leakage that inflated a crop-yield model's R² from about −0.2 to
> 1.00. Rebuilt it with grouped nested CV and permutation testing, and shipped a Streamlit
> analytics dashboard (Python, scikit-learn).

## About AI assistance

This rebuild was produced with an AI coding assistant working at your direction. Your
role was defining the goals and constraints, reviewing and accepting the work, and
owning the publication decisions. Before listing it:

- Read the code and docs until you can explain every design decision without notes
  (grouped split, nested CV, permutation test, why the notebook's R² was wrong).
- If an application or interview asks about AI use, say so plainly.
- Where you want stronger ownership, extend the project yourself, e.g. apply the same
  protocol to a real, openly licensed yield dataset.
