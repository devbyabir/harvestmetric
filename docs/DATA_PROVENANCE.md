# Data provenance and publication rights

## Summary

| Question | Status |
|---|---|
| Where does the data come from? | **High-confidence match** to a public Kaggle dataset (details below). Not byte-verified. |
| Is it real farm data? | **No.** The publisher describes it as simulated, and the statistical audit agrees. |
| Is it hospital or patient data? | **No evidence of any link.** No file mentions hospitals or patients, and every column is agricultural. |
| Licence of the public source | Apache License 2.0, as listed on Kaggle (checked 2026-11-12) |
| Is the local CSV cleared for publication? | **Not yet.** It is excluded from the repository until you complete the steps below. |

## Verified facts

These were checked on 2026-11-12 against Kaggle's public dataset metadata
(`https://www.kaggle.com/api/v1/datasets/view/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction`):

- **Title:** "Smart Farming Sensor Data for Yield Prediction"
- **Creator / owner:** Atharva Soundankar (`atharvasoundankar`)
- **Licence:** Apache 2.0
- **File:** `Smart_Farming_Crop_Yield_2024.csv`, 83,584 bytes, created 2025-04-15
- **Description (paraphrased):** it simulates IoT- and satellite-based smart-farming
  operations for 500 farms in India, the USA and Africa. Its column list matches this
  project's CSV column for column, including documented ranges (lat 10-35, lon 70-90,
  NDVI 0.3-0.9, soil pH 5.5-7.5) and a "random in-cycle timestamp".

Facts verified locally:

- The local `final_dataset_crop_yield.csv` has 5,000 rows, made of 500 distinct records
  each repeated 10 times. Its SHA-256 is `d3fbc283f6a9d3eb06b302267d4cfd7db8bc07ef4133c30edfde67cb8d101847`.
- Its 500 distinct records have the same schema, farm count and value ranges as the
  Kaggle description.

## Inferences (not verified)

- The local file is very likely the Kaggle file stacked 10 times. The de-duplicated
  content is 83,490 bytes (LF line endings), close to but not equal to Kaggle's 83,584.
  The gap may come from formatting differences (e.g. date format or line endings).
  Without downloading the Kaggle file, the records cannot be confirmed identical.
- Who duplicated the rows, and why, is unknown. The notebook downloaded the CSV from a
  Google Drive file ID whose owner is not recorded.

## Steps to clear the data for publication

1. Sign in to Kaggle and download `Smart_Farming_Crop_Yield_2024.csv` from the dataset
   page above. Read the licence shown there yourself.
2. Save it as `data/raw/Smart_Farming_Crop_Yield_2024.csv`.
3. Check it matches the copy the analysis used:
   ```bash
   python -m harvestmetric.provenance data/raw/Smart_Farming_Crop_Yield_2024.csv final_dataset_crop_yield.csv
   ```
   Exit code 0 and `identical_records: True` means the records match.
4. Rerun `python -m harvestmetric.audit` and `python -m harvestmetric.train`. With the
   Kaggle file, `raw_rows` will be 500 and the legacy-split demonstration is skipped.
   Because rows are sorted by `farm_id` before splitting, all other results should
   match the saved reports exactly.
5. Only to deploy on Streamlit Community Cloud: delete the `data/raw/*.csv` line from
   `.gitignore` and commit the file together with `NOTICE`, which carries the
   attribution Apache 2.0 requires. Otherwise keep the CSV out of the repository. The
   code and saved reports work without it.

Never publish `final_dataset_crop_yield.csv` itself: it is an unverified, modified copy.

## How results may be described

- Say "a public **synthetic** smart-farming dataset (Kaggle, Apache-2.0)".
- Do not describe it as real sensor, farm, field-trial or hospital data.
- Do not claim the findings say anything about real crops.
