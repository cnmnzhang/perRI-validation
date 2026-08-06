# perri_validation

Replicating analyses from the _Personalized clinical reference intervals for routine precision medical care_ study across 43 routine laboratory tests. Upon calculating personalized reference ranges from a bayesian setpoint model, we perform mortality and morbidity analysis, look at rates to incident diagnosis across clinical cases (diagnosis, pregnancy, and an iron infusion treatment study).

## Required input tables

Schemas are defined in `constants/schemas.py`. Column names are referred to by variables set in `constants/runtime.py` instead of being hardcoded. 

| Table | File ([click for example](data/examples/)) | Required columns | Used by |
|---|---|---|---|
| Tests | `data/tests.csv` (or `tests.csv.gz`) ([`tests_example.csv`](data/examples/tests_example.csv))| `anon_id, ts, test_code, result_value, sex` | fig5_iron_infusion, fig4_dx_cases, fig3_dx, fig3_hazard |
| Dx | `data/dx.csv` ([`dx_example.csv`](data/examples/dx_example.csv))| `anon_id, icd9, icd10, date` | dx_incident |
| Demographics | `data/demographics.csv` ([`demographics_example.csv`](data/examples/demographics_example.csv))| `anon_id, sex, birth_date, death_ts` | fig3_dx, fig4_dx_cases, fig3_hazard |
| iron_mar | `data/iron_mar.csv` ([`iron_mar_example.csv`](data/examples/iron_mar_example.csv))| `anon_id, ts` | fig5_iron_infusion|
| pregnancy_labs | `data/pregnancy_labs.csv` ([`pregnancy_labs_example.csv`](data/examples/pregnancy_labs_example.csv))| `anon_id, ts, test_code, result_value` | fig4_pregnancy |
| pregnancy_outcomes_and_demogs | `data/pregnancy_outcomes_and_demogs.csv` ([`pregnancy_outcomes_and_demogs_example.csv`](data/examples/pregnancy_outcomes_and_demogs_example.csv))| `anon_id, delivery_date, gestational_age, rbc_tf, pih` | fig4_pregnancy |



### Tests
One row per patient per test over time, covering every marker needed. Analysis for a missing marker will be gracefully handled. `tests.csv` may be gzipped by adding `tests.csv.gz` instead of `tests.csv` since file may be large. Scripts will look for `tests.csv` first, then falls back to `tests.csv.gz`. The table will be split into one CSV per marker, once, shared by every analysis. See `run_tests_by_marker` in `## Quickstart`. Tests do not have to be isolated. Downstream `run_setpoints_by_marker` will handle isolation. Deduplication is handled to avoid marking a test as non-isolated, when it is. 

**Markers needed in `tests.csv`**, by analysis:
- fig3_hazard: all 43 routine laboratory markers
- fig3_dx: `HB` , `TNEUT` , `ALB`, `ALT`, `MCV`, `P`, `GLU`, `K` 
- fig5_iron_infusion: `HB`
- fig4_dx_cases: `CRE`, `WBC`, `TSH` + **`T4FR` (T4FR isn't assessed as a routine laboratory test, but used to identify incident hypothyroidism)**

**Markers by battery.** Also listed in `constants/marker_lab_config.py:TESTCODES_LIST`.

| Battery | Markers |
|---|---|
| CBC | HB, HCT, MCH, MCHC, MCV, PLT, RBC, RDWCV, WBC |
| BMP | BUN, CA, CL, CO2, CRE, GLU, IGAP, K, NA |
| WCD | LYMPH, MONOC, TNEUT |
| LFT | ALB, ALK, ALT, AST, BIL, BILD, TP |
| LIPID | CHOL, HDL, LDL, NONHDL, TRIG |
| COAG | PROINR, PROPAT |
| MISC | A1C, FER, HSCRP, LD, MG, P, TSH, VITDT |

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


## Quickstart

```bash
pip install -r requirements.txt

# Run everything in the necessary and preferred order:
python -m perri_validation.run_all --analysis all

# Or one analysis at a time
#### dependencies
python -m perri_validation.scripts.run_tests_by_marker
python -m perri_validation.scripts.run_setpoints_by_marker
python -m perri_validation.scripts.run_dx_incident
#### Analysis
python -m perri_validation.scripts.run_fig3_hazard
python -m perri_validation.scripts.run_fig3_dx
python -m perri_validation.scripts.run_fig4_dx_cases
python -m perri_validation.scripts.run_fig5_iron_infusion
#### Analysis with no dependencies
python -m perri_validation.scripts.run_fig4_pregnancy

# rerun marker dependencies for a single marker
python -m perri_validation.scripts.run_tests_by_marker --marker TSH
python -m perri_validation.scripts.run_setpoints_by_marker --marker TSH

# rerun analysis, but won't recompute dependencies (dx_incident, sp_df)
python -m perri_validation.scripts.run_fig3_dx --force
```

Pass `--force` to any script to recompute (ignoring any existing cache for that run). Recomputing for an analysis script will not recompute dependencies. 



## What each analysis script produces

| script | purpose | notes|outputs | 
|---|---|---|---|
|`tests_by_marker`|creates dependencies|Splits `Tests` table into one file per marker present in `tests.csv`. Prerequisite for `fig3_dx`/`fig3_hazard`/`fig4_dx_cases`/`fig5_iron_infusion`|`data/cache/tests_by_marker/{marker}.csv`|
|`setpoints_by_marker`|creates dependencies|Uses previous outputs and fits all 43 markers in `TESTCODES_LIST`, filtered down to patients who have at least 3 setpoints for the given marker file (hence, `_m3`). Setpoints are calculated using the `perri` package, pinned to a GitHub tag in `requirements.txt`|`data/cache/sp_df_{marker}_full_m3.csv`|
|`dx_incident`|creates dependencies|Uses `Dx` to derive the first diagnosis per patient for a given set of diagnoses, where multiple codes can match to the same diagnosis. We match on ICD10, with ICD9 fall back, using a hardcoded mapping in `icd_config.py`. Prerequisite for `fig3_dx` and `fig4_dx_cases`|`outputs/dx_incident/dx_incident.csv`|
|`fig3_hazard`|figure analyses|for each of the 43 markers, using the setpoint caches, fits a mortality Cox regression on the patient's personal setpoint (mu/sigma/cv, adjusted for age and sex). a. `fig3a_hr_by_model.svg` using each patient's 5th setpoint, and b. `fig3b_hr_by_baseline` using the setpoint from only their 1st through 5th isolated measurement, to see how hazard ratios stabilize as more data accumulates | `outputs/fig3_hazard/*` |
|`fig3_dx`|figure analyses|uses `dx_incident`'s output, for each of 8 diagnoses, patients not yet diagnosed as of their 3rd personal setpoint estimate are split by whether that setpoint falls above/below a sex-specific 25th or 75th population percentile cutoff, and KM survival curves are fit per group| `outputs/fig4_dx_cases/*`|
|`fig4_dx_cases`|figure analyses|for each marker-outcome pair (**CRE** -> acute kidney injury, **WBC** -> leukemia, **TSH** -> hypothyroidism), builds a cohort anchored on a patient's setpoint and produces one combined figure with a row per outcome: a representative patient trajectory, the odds-ratio forest plot, Kaplan-Meier curves, and a reclassification confusion matrix heatmap.| Each case saves the plotted data to `fig4_dx_cases_<case>_ors.csv`,`fig4_dx_cases_<case>_km_data.csv`, and `fig4_dx_cases_<case>_heatmap_t1.csv`|
|`fig4_pregnancy`|figure analyses|for two marker-outcome pairs (**WBC** -> pre-eclampsia, **HCT** -> received a transfusion), fits each patient's pre-conception personal setpoint, and produces a 2x3 panel figure|`outputs/fig4_pregnancy/*`|
|`fig5_iron_infusion`|figure analyses|identifies each patient's first IV iron infusion course, pairs it with pre/post-treatment **HB** lab values and a personal setpoint, and produces a 6-panel figure covering treatment timing, dose-response, lab trajectories around treatment, and response by baseline severity|`outputs/fig5_iron_infusion/*`|


## Adapting to a different laboratory

New sites must review the hard coded **Population reference intervals (popRI)**. Edit `pop_ri` in `constants/marker_lab_config.py`'s `MARKER_CONFIG` dict directly. Specifically, within each marker entry:
  `"pop_ri": {"F": (lower, upper), "M": (lower, upper)}` 

The use for this is
  1. Fitting setpoints require grid bounds derived from the popRI. See `utils/setpoints.py::compute_sp_df`/ `_params_override`/`_grid_bounds_from_pop_ri`
  
  2. Clinical reclassification confusion matrices and Kaplan-Meier logic require pop_ri bounds. 
  
  An exception is one-sided markers (only HDL, which has an infinite upper bound, where only "too low" is
  clinically abnormal). Because `pop_ri`'s formula needs a finite width, we call `_params_override` 
  which computes an empirical (patch_lower, patch_upper)
  reference interval. This falls back to `perri`'s default bundled `min_mu`/`max_mu`/`max_sigma`
  if the input `Tests` population has no isolated data at all to compute a patch from.


## Parallelization

Edit `N_JOBS` in `constants/runtime.py` to change the worker count (default 3; set to 1 to force
serial, e.g. for debugging). Per the comment there: benchmarked on a 128-core machine, 16 workers
gave ~6x speedup over serial, but going higher was flat-to-worse since each joblib worker pays its
own numba JIT warmup for perri's `@njit`-compiled `bayesian()` fit on first call.

- Used in tests_by_marker splitting. Once the master is read (~13 minutes or a 1.75GB gzipped file), 
writing each marker's split CSV is independent I/O-bound work, so it runs across a `ThreadPoolExecutor` (up to N_JOBS workers) instead of one file at a time.
- Used in setpoint computation so `compute_sp_df` to fit patients in parallel. 
