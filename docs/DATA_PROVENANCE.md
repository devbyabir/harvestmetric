# Data provenance and publication rights

All checks below were made on **12 November 2026**. Exact URLs, query parameters, retrieval
times and SHA-256 checksums for Part B are in [`data/processed/manifest.json`](../data/processed/manifest.json).

## Summary

| Data | Used in | Source and terms | In the repository |
|---|---|---|---|
| Smart Farming Sensor Data for Yield Prediction | Part A | Atharva Soundankar, Kaggle; **Apache-2.0**; publisher says it is *simulated* | **No** (aggregate results only) |
| Original working copy `final_dataset_crop_yield.csv` | Part A (verification only) | Derived from the Kaggle file (verified below) | **No**: stays local |
| USDA NASS Quick Stats, Iowa county corn-for-grain yield | Part B | US Department of Agriculture; **CC0 1.0** (data.gov catalog) | Processed panel only |
| NASA POWER daily point data | Part B | NASA Langley Research Center; free use, **citation required**, notification of redistribution requested | Derived features in the panel |
| US Census Bureau 2024 Gazetteer (counties) | Part B | US federal government work, not subject to US copyright (17 U.S.C. §105) | Coordinates in the panel |

No source or file mentions hospitals, patients or health data. Neither dataset is hospital data.

---

## Part A: synthetic dataset

**Verified facts**
- Kaggle's public metadata API lists the title *Smart Farming Sensor Data for Yield Prediction*, owner
  `atharvasoundankar` (Atharva Soundankar), licence **Apache 2.0**, and file
  `Smart_Farming_Crop_Yield_2024.csv` (83,584 bytes, created 2025-04-15).
- The description says the data **"simulates"** IoT and satellite smart-farming operations for 500 farms.
- The official file was downloaded through Kaggle's public download endpoint without sign-in
  (`python -m harvestmetric.synthetic.acquire`). SHA-256:
  `49b1f2451c9c9f67dcfe1b5aafbbaa77f30385231a4baf7c13c2b2f92d77940d`.
- **The local working copy is derived from it, verified at record level.** I normalised both files
  (exact duplicates removed, dates converted to ISO) and compared every cell
  (`python -m harvestmetric.synthetic.provenance`). Result: 500 vs 500 records, same columns,
  **0 differing cells**. The working copy differs only in date format (dd/mm/yyyy instead of
  yyyy-mm-dd) and in repeating every record exactly 10 times, in the same order. Its SHA-256 is
  `d3fbc283f6a9d3eb06b302267d4cfd7db8bc07ef4133c30edfde67cb8d101847`. Because of those changes the two
  files are **not byte-identical**.

**Unknown:** who reformatted and duplicated the file, and why. The original notebook downloaded it from an
unrecorded Google Drive file ID.

**Publication decision.** Apache-2.0 would permit redistributing the official file with its licence and a
NOTICE. Nothing in the project needs it, so this repository does **not** redistribute it. The app shows Part A
only through committed aggregate results and a simulator that generates its own data. Reproducing Part A
downloads the official file. The local working copy is never published.

## Part B: real observational data

### USDA NASS Quick Stats: county corn yield (target and yield history)
- **What:** `CORN, GRAIN - YIELD, MEASURED IN BU / ACRE`, `SURVEY`, `COUNTY` level, `IOWA`, all years
  (1926–2025 at retrieval).
- **Access:** the Quick Stats web interface's spreadsheet export, which needs no key. The pipeline encodes
  the query (`/uuid/encode?...`) and downloads `/data/spreadsheet/<uuid>.csv`. The query parameters are
  in `harvestmetric/real/config.py` (`NASS_QUERY`) and in the manifest. The documented Quick Stats API
  needs a free key and returns at most 50,000 records. The key route was not needed; if the web export
  changes, request a key and query the same parameters.
- **Terms:** catalog.data.gov lists NASS Quick Stats under **CC0 1.0** (publisher: National Agricultural
  Statistics Service). The NASS API terms ask services to state *"This product uses the NASS API but is not
  endorsed or certified by NASS"*. We did not use the API, but the app and README carry an equivalent
  non-endorsement line.
- **Units:** bushels per harvested acre (56 lb bushel at 15.5% moisture; × 0.0627 ≈ t/ha).
- **Handling:**
  - "OTHER (COMBINED) COUNTIES" and "OTHER COUNTIES" rows have no county code and are dropped.
  - Withheld values such as `(D)` are treated as missing.
  - Duplicate county-years raise an error, and values must lie within 1–350 bu/ac. The minimum observed,
    3.0 bu/ac in 1936, is a genuine Dust Bowl value.
  - From 2015, NASS did not publish every county (73–97 of 99 per year). Those county-years have no target
    and are **not imputed or scored**.
- **Revisions:** Quick Stats serves the *current* published estimates. A forecaster in year t would have
  seen the then-current value for year t−1, which may differ slightly. This is a documented limitation.

### NASA POWER daily weather (season-to-date features)
- **What:** `T2M_MAX`, `T2M_MIN` (°C) and `PRECTOTCORR` (mm/day), daily, from the **POWER daily point API
  v2.10.3** (community `AG`, sources MERRA-2/POWER), 1981-01-01 to 2026-07-31. One request per county,
  at its Census internal point.
- **Terms:** no formal use restrictions. Citation is required, and NASA asks to be notified of
  redistribution and of publications. Required acknowledgement: *"These data were obtained from the NASA
  Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science
  Program."* The repository commits only derived seasonal features, not the raw daily series.
- **Handling:** fill value −999 becomes missing, and a season with more than 2% missing days raises an error
  (none occurred). Features use only days on or before 31 July.

### US Census Bureau 2024 Gazetteer
- **What:** county `GEOID` (FIPS), name, and internal-point latitude/longitude (`INTPTLAT`, `INTPTLONG`)
  for Iowa's 99 counties, from `2024_Gaz_counties_national.zip`.
- **Terms:** US federal government work, not subject to US copyright (17 U.S.C. §105). This was not stated
  as a licence on the download page; it follows from the statute.

### Spatial join and its limits
Counties are joined by 5-digit FIPS (`19` + NASS county ANSI). Weather is taken at one point per county
from a grid of roughly 0.5° × 0.625°, so neighbouring counties often share a grid cell. The point is the
county's geographic internal point, not the centroid of its cornfields. County-scale weather is therefore
approximate, and extremes are smoothed.

## Credits required when redistributing
- NASS: non-endorsement statement as above.
- NASA POWER: the acknowledgement sentence above, plus the API version and access date.
- Kaggle dataset (only if you ever redistribute it): Apache-2.0 licence text and NOTICE naming Atharva Soundankar.

## How to describe the data
- Part A: "a public **synthetic** smart-farming dataset (Kaggle, Apache-2.0)". Never "real sensor/farm data".
- Part B: "USDA NASS county yield estimates and NASA POWER reanalysis-based weather". These are official
  survey estimates and modelled weather, not field measurements.
