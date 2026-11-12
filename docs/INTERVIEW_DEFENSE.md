# Interview defence sheet

Be able to answer each question in your own words, without notes. The short answers point to where the
evidence lives.

## Data
1. **Where does each dataset come from, and what licence covers it?**
   - Part A: Kaggle, Apache-2.0, synthetic.
   - Part B: NASS Quick Stats (CC0), NASA POWER (citation required) and Census Gazetteer (US Gov. work).
   - Details: [DATA_PROVENANCE.md](DATA_PROVENANCE.md).
2. **How do you know Part A is synthetic?**
   - The publisher says it "simulates" smart farming.
   - US and African regions have Indian coordinates.
   - The columns behave like independent uniform draws, and yield is uniform on 2000–6000 for every crop.
3. **How did you verify the local CSV came from Kaggle?**
   - I normalised dates, removed duplicates and compared every cell: 0 differences.
   - It is not byte-identical, because of the date format and the 10× repetition.
4. **What is a NASS county yield, and what are its weaknesses?**
   - A survey-based estimate in bu/harvested acre.
   - It gets revised over time.
   - Some counties are withheld or combined, especially after 2014.
5. **Why NASA POWER, and what is wrong with a single point per county?**
   - It is free, global, daily and keyless.
   - But its ~0.5° grid means neighbouring counties share cells, and the point ignores where corn is grown.

## Leakage
6. **Exactly why did the original model score R² ≈ 1.00?** Row-level KFold on data where every record
   appeared 10 times. Test rows had identical copies in training, so the forest memorised them.
7. **Name every leakage path you closed in Part B.**
   - Weather after 31 July.
   - Current or future yields in the history features.
   - Other counties' same-year yields.
   - Identifiers as inputs.
   - Tuning on test years.
   - Choosing the model after seeing the test results.
8. **Is using last year's published yield leakage?** No: it is published before August. The caveat is that
   Quick Stats serves *revised* values.

## Splits and validation
9. **Why a chronological split rather than random KFold?** Forecasts are made for future seasons, and one
   season's weather affects every county. A random split would leak each season's shock into training.
10. **What does the expanding-window CV do?** Each validation block (e.g. 2003–05) is fitted only on earlier
    years. Hyperparameters were chosen by its pooled RMSE.
11. **Why is the geographic check separate, and what does it not show?**
    - It holds out whole NASS districts within 1981–2011.
    - The same seasons appear on both sides, so it shows transfer across space, not skill on new seasons.
12. **Why bootstrap by year instead of by row?** County errors within a year are strongly correlated.
    Row resampling would pretend there are 1,282 independent cases when there are about 14 seasons.

## Baselines and models
13. **Why these three baselines?** Historical mean, last year and county trend are what an analyst would use.
    The trend is the hard one to beat because of genetic and management gains.
14. **Why model the deviation from trend?** Weather explains departures from the long-run trend. Separating
    them stops the trees from having to learn the trend itself.
15. **Why only Ridge and gradient boosting?**
    - Ridge is a linear reference; boosting can capture heat and drought thresholds.
    - More algorithms would add selection noise, not insight.

## Results and uncertainty
16. **Did the model work?**
    - On test it lowered RMSE from 21.8 to 17.6, mainly in the 2012 drought.
    - But the year-bootstrap CI of the gain is [−0.00, +9.46], and in CV the model lost to the trend.
    - So: promising, not established.
17. **Why did CV and test disagree?**
    - The early CV folds learned from only three disaster years and over-predicted heat losses in 1997, 1999
      and 2002.
    - The weather–yield relationship shifted.
    - I did not change anything after seeing the test results.
18. **How were the intervals built, and do they work?**
    - Split conformal on time-ordered CV residuals: ±24.8 (80%) and ±32.7 (90%) bu/ac.
    - Coverage is 86% / 94% overall, but only 68–69% at 80% in 2012 and 2020, because exchangeability breaks
      within shock years.
19. **Why are there no SHAP plots?** Explanations of an unvalidated model invite over-interpretation, and
    associations are not causal effects anyway.
20. **What was the gate, and why fix it in advance?** CV win plus a test gain whose CI is above 0. Fixing it
    first prevents moving the goalposts after seeing results.

## Limitations (say these before you are asked)
- Point weather; revised targets; counties missing after 2014, not at random; only 14 test years.
- One state, one crop.
- Part A says nothing about real crops.

---

## An extension to implement yourself
**Add NASS weekly crop-condition ratings as an August-1 feature and re-run the same pre-registered evaluation.**

- **Why it matters:** condition ratings (% of corn rated good or excellent) are published weekly during the
  season and summarise field conditions that point weather misses. They are a standard input to real
  in-season forecasts.
- **What to do:**
  1. Query Quick Stats for `CORN - CONDITION, MEASURED IN PCT GOOD`/`EXCELLENT`, state and district level,
     using the existing acquisition pattern in `harvestmetric/real/acquire.py`.
  2. Keep only the last week ending **on or before 31 July**.
  3. Join by district and year. Write a test proving no rating after 31 July can enter the feature.
  4. Re-run `python -m harvestmetric.real.train` *without* changing the gate. Report the result whatever it is.
- **What you will learn to explain:**
  - Joining data at different spatial levels (district to county).
  - Handling report dates as a leakage boundary.
  - Whether a stronger signal finally passes the gate, or not.
