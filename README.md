# perri_validation

The purpose of this repository is to take the analyses from the original _Personalized clinical reference intervals for routine precision medical care_ study and make them reproducible on an independent site’s data. Given = standardized input tables, the pipeline reconstructs personalized setpoints locally from a bayesian setpoint model, [*perri*](https://github.com/cnmnzhang/perRI). Then we rerun the major analyses, across clinical cases (diagnosis, pregnancy, and an iron infusion treatment study), with provided formatted visualization functions. 


## What does the data look like?

### Required input tables

Schemas are defined in [`constants/schemas.py`](constants/schemas.py). Column names are referred to by variables set in [`constants/runtime.py`](constants/runtime.py) instead of being hardcoded. 

| Table | File ([click for example](data/examples/)) | Required columns | Used by |
|---|---|---|---|
| Tests | `data/tests.csv` (or `tests.csv.gz`) ([`tests_example.csv`](data/examples/tests_example.csv))| `anon_id, ts, test_code, result_value, sex` | fig5_iron_infusion, fig4_dx_cases, fig3_dx, fig3_hazard |
| Dx | `data/dx.csv` ([`dx_example.csv`](data/examples/dx_example.csv))| `anon_id, icd9, icd10, diagnosis_ts` | dx_incident |
| Demographics | `data/demographics.csv` ([`demographics_example.csv`](data/examples/demographics_example.csv))| `anon_id, sex, birth_date, death_ts` | fig3_dx, fig4_dx_cases, fig3_hazard |
| iron_mar | `data/iron_mar.csv` ([`iron_mar_example.csv`](data/examples/iron_mar_example.csv))| `anon_id, ts` | fig5_iron_infusion|
| pregnancy_labs | `data/pregnancy_labs.csv` ([`pregnancy_labs_example.csv`](data/examples/pregnancy_labs_example.csv))| `anon_id, ts, test_code, result_value` | fig4_pregnancy |
| pregnancy_outcomes_and_demogs | `data/pregnancy_outcomes_and_demogs.csv` ([`pregnancy_outcomes_and_demogs_example.csv`](data/examples/pregnancy_outcomes_and_demogs_example.csv))| `anon_id, delivery_date, gestational_age, rbc_tf, pih` | fig4_pregnancy |



### Tests
One row per patient per test over time, covering every marker needed. Analysis for a missing marker will be gracefully handled. `tests.csv` may be gzipped by adding `tests.csv.gz` instead of `tests.csv` since file may be large. Scripts will look for `tests.csv` first, then falls back to `tests.csv.gz`. The table will be split into one CSV per marker, once, shared by every analysis. See `build_splits_by_marker` in `## Quickstart`. 

Tests do not have to be isolated. Downstream `build_setpoints` will handle isolation. Deduplication is handled to avoid marking a test as non-isolated, when it is. 

>If your data is already partitioned by marker, you can skip `tests.csv`/`build_splits_by_marker` entirely and drop per-marker CSVs (same columns as Tests, above) straight into `data/cache/splits_by_marker/{test_code}.csv` yourself

**Markers needed in `tests.csv`**, by analysis:
- fig3_hazard: all 43 routine laboratory markers
- fig3_dx: `HB` , `TNEUT` , `ALB`, `ALT`, `MCV`, `P`, `GLU`, `K` 
- fig5_iron_infusion: `HB`
- fig4_dx_cases: `CRE`, `WBC`, `TSH` + **`T4FR` (T4FR isn't assessed as a routine laboratory test, but used to identify incident hypothyroidism)**
- fig4h_multimarker: `CBC` and `BMP` batteries

**Markers by battery.** Also listed in [`constants/marker_lab_config.py`](constants/marker_lab_config.py)`:TESTCODES_LIST`.

| Battery | Markers |
|---|---|
| CBC | HB, HCT, MCH, MCHC, MCV, PLT, RBC, RDWCV, WBC |
| BMP | BUN, CA, CL, CO2, CRE, GLU, IGAP, K, NA |
| WCD | LYMPH, MONOC, TNEUT |
| LFT | ALB, ALK, ALT, AST, BIL, BILD, TP |
| LIPID | CHOL, HDL, LDL, NONHDL, TRIG |
| COAG | PROINR, PROPAT |
| MISC | A1C, FER, HSCRP, LD, MG, P, TSH, VITDT |

```bash
full_marker_list = [
    'HB', 'HCT', 'MCH', 'MCHC', 'MCV', 'PLT', 'RBC', 'RDWCV', 'WBC',
    'BUN', 'CA', 'CL', 'CO2', 'CRE', 'GLU', 'IGAP', 'K', 'NA',
    'LYMPH', 'MONOC', 'TNEUT', 'ALB', 'ALK', 'ALT', 'AST', 'BIL',
    'BILD', 'TP', 'CHOL', 'HDL', 'LDL', 'NONHDL', 'TRIG', 'PROINR',
    'PROPAT', 'A1C', 'FER', 'HSCRP', 'LD', 'MG', 'P', 'TSH', 'VITDT', 'T4FR'
]
```

### Dx
One row per patient per date per ICD9/ICD10 code, where multiple codes can map to the same diagnosis. Usually from EHR, both are filled, but processing scripts handles if one is null. These scripts also handle duplicates.

### Pregnancy
Pregnancy tables were originally built from a separate cohort, so pregnancy analysis (`fig4_pregnancy`) don't interact with other fig scripts. 

### Iron infusion
Data came from medication administration record tables. Input data must come pre-filtered to the IV route and iron sucrose formulation. It's possible that a patient gets multiple administrations (~1-5 closely spaced doses) over a treatment course, and patients can get multiple treatment courses over their lifetime. This is handled in the analysis. 

Fig5 writes the cohort-level data underlying panels B/D/E/F to `iv_iron_cohort.csv` and
the aligned per-measurement data underlying panel C to `fig5_trajectory_data.csv`. The
swimmer panel's cohort, laboratory, and infusion source tables are retained under
`iv_iron_bundle/`.




## What each analysis script produces

| script | purpose | question | notes|outputs | 
|---|---|---|---|---|
|[`splits_by_marker`](scripts/build_splits_by_marker.py)|creates dependencies||Splits `Tests` table into one file per marker present in `tests.csv`. Prerequisite for `fig3_dx`/`fig3_hazard`/`fig4_dx_cases`/`fig5_iron_infusion`|`data/cache/splits_by_marker/{marker}.csv`|
|[`setpoints_by_marker`](scripts/build_setpoints.py)|creates dependencies||Uses previous outputs and fits all 43 markers in `TESTCODES_LIST`, filtered down to patients who have at least 3 setpoints for the given marker file (hence, `_m3`). Setpoints are calculated using the `perri` package, pinned to a GitHub tag in `requirements.txt`|`data/cache/sp_df_{marker}_full_m3.csv`|
|[`dx_incident`](scripts/build_dx_incident.py)|creates dependencies||Uses `Dx` to derive the first diagnosis per patient for a given set of diagnoses, where multiple codes can match to the same diagnosis. We match on ICD10, with ICD9 fall back, using a hardcoded mapping in [`constants/icd_config.py`](constants/icd_config.py). Prerequisite for `fig3_dx` and `fig4_dx_cases`|`data/outputs/dx_incident/dx_incident.csv`|
|[`fig3_hazard`](scripts/run_fig3_hazard.py)|figure analyses|Do personalized setpoint characteristics reproduce associations with mortality?|for each of the 43 markers, using the setpoint caches, fits a mortality Cox regression on the patient's personal setpoint (mu/sigma/cv, adjusted for age and sex). a. `fig3a_hr_by_model.svg` using each patient's 5th setpoint, and b. `fig3b_hr_by_baseline` using the setpoint from only their 1st through 5th isolated measurement, to see how hazard ratios stabilize as more data accumulates | `data/outputs/fig3_hazard/*` |
|[`fig3_dx`](scripts/run_fig3_dx.py)|figure analyses|Do patient-specific laboratory phenotypes stratify subsequent diagnosis risk?|uses `dx_incident`'s output, for each of 8 diagnoses, patients not yet diagnosed as of their 5th personal setpoint estimate are split by whether that setpoint falls above/below a sex-specific 25th or 75th population percentile cutoff, and KM survival curves are fit per group| `data/outputs/fig3_dx/*`|
|[`fig4_dx_cases`](scripts/run_fig4_dx_cases.py)|figure analyses|Does perRI add clinically meaningful reclassification beyond popRI?|for each marker-outcome pair (**CRE** -> acute kidney injury, **WBC** -> leukemia, **TSH** -> hypothyroidism), builds a cohort anchored on a patient's setpoint and produces one combined figure with a row per outcome: a representative patient trajectory, the odds-ratio forest plot, Kaplan-Meier curves, and a reclassification confusion matrix heatmap.| Each case saves the plotted data to `fig4_dx_cases_<case>_ors.csv`,`fig4_dx_cases_<case>_km_data.csv`, and `fig4_dx_cases_<case>_heatmap_t1.csv`|
|[`fig4h_multimarker`](scripts/run_fig4h_multimarker.py)|figure analyses|Does personalized interpretation across whole panels stratify incident disease?|for each of two batteries (**CBC**, **BMP**), builds a cohort of patients with a complete same-day marker panel post-washout, bins patients by how many markers fall outside their personal (perRI) vs population (popRI) reference interval, and reports 1-year incident-diagnosis rates per bin|`data/outputs/fig4h_multimarker/*`. UWM ground-truth counts/rates for comparison are in [`data/UWM/fig4h/fig4h_ground_truth.csv`](data/UWM/fig4h/fig4h_ground_truth.csv)|
|[`fig4_pregnancy`](scripts/run_fig4_pregnancy.py)|figure analyses|Do pre-conception personalized baselines improve interpretation during pregnancy?|for two marker-outcome pairs (**WBC** -> pre-eclampsia, **HCT** -> received a transfusion), fits each patient's pre-conception personal setpoint, and produces a 2x3 panel figure|`data/outputs/fig4_pregnancy/*`|
|[`fig5_iron_infusion`](scripts/run_fig5_iron_infusion.py)|figure analyses|Does deviation from personal Hb baseline relate to treatment response?|identifies each patient's first IV iron infusion course, pairs it with pre/post-treatment **HB** lab values and a personal setpoint, and produces a 6-panel figure covering treatment timing, dose-response, lab trajectories around treatment, and response by baseline severity|`data/outputs/fig5_iron_infusion/*`|

`fig3_hazard` (fig3a/b), `fig3_dx`, and `fig4h_multimarker` require each patient's **5th** isolated setpoint specifically (`INDEX_COL == 5`, via [`utils/clinical/run_clinical.py`](utils/clinical/run_clinical.py)'s `get_one_setpoint` personalized-logic branch, or a direct index filter in `fig4h_multimarker`) — a stricter bar than the `_m3` fit gate `setpoints_by_marker` applies (a patient just needs >=3 isolated measurements to be fit at all). `fig4_dx_cases`, `fig4_pregnancy`, and `fig5_iron_infusion` use whichever setpoint is relevant to their own anchor/presenting-event logic instead of a fixed index, so any patient meeting the `_m3` fit gate is eligible.


## Adapting to a different laboratory

New sites must review the hard coded **Population reference intervals (popRI)**. Edit `pop_ri` in [`constants/marker_lab_config.py`](constants/marker_lab_config.py)'s `MARKER_CONFIG` dict directly. Specifically, within each marker entry:
  `"pop_ri": {"F": (lower, upper), "M": (lower, upper)}` 

The use for this is
  1. Fitting setpoints require grid bounds derived from the popRI. See [`utils/setpoints.py`](utils/setpoints.py)`::compute_sp_df`/ `_params_override`/`_grid_bounds_from_pop_ri`
  
  2. Clinical reclassification confusion matrices and Kaplan-Meier logic require pop_ri bounds. 
  
An exception is one-sided markers (only HDL, which has an infinite upper bound, where only "too low" is clinically abnormal). These markers are auto-detected, and `_params_override` will calculate an empirical reference interval.


## Quickstart

```bash
pip install -r requirements.txt

# Run everything in the necessary and preferred order:
python -m run_all --analysis all
time python -m run_all

# Or one analysis at a time
#### dependencies
python -m scripts.build_splits_by_marker
python -m scripts.build_setpoints
python -m scripts.build_dx_incident
#### Analysis
python -m scripts.run_fig3_hazard
python -m scripts.run_fig3_dx
python -m scripts.run_fig4_dx_cases
python -m scripts.run_fig4h_multimarker
python -m scripts.run_fig5_iron_infusion
#### Analysis with no dependencies
python -m scripts.run_fig4_pregnancy

# rerun marker dependencies for a single marker
python -m scripts.build_splits_by_marker --marker TSH
python -m scripts.build_setpoints --marker TSH

# rerun analysis, but won't recompute dependencies (dx_incident, sp_df)
python -m scripts.run_fig3_dx --force
```

Pass `--force` to any script to recompute (ignoring any existing cache for that run). Recomputing for an analysis script will not recompute dependencies. 

### Parallelization

Edit `N_JOBS` in [`constants/runtime.py`](constants/runtime.py) to change the worker count (default 3; set to 1 to force
serial, e.g. for debugging). Per the comment there: benchmarked on a 128-core machine, 16 workers
gave ~6x speedup over serial, but going higher was flat-to-worse since each joblib worker pays its
own numba JIT warmup for perri's `@njit`-compiled `bayesian()` fit on first call.

- Used in splits_by_marker splitting. Once the master is read (~13 minutes or a 1.75GB gzipped file), 
writing each marker's split CSV is independent I/O-bound work, so it runs across a `ThreadPoolExecutor` (up to N_JOBS workers) instead of one file at a time.
- Used in setpoint computation so `compute_sp_df` to fit patients in parallel. 

### Side-by-side Validation
See [`scripts/diagnostics/validate_fig3_hazard.py`](scripts/diagnostics/validate_fig3_hazard.py).
```bash
python -m scripts.diagnostics.validate_fig3_hazard
```