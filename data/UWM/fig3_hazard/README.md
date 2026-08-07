# Ground truth: fig3a/fig3b hazard ratios

Reference values from `bayesian-setpoint-inference`, the source pipeline `perri-validation`
replicates, to diff `outputs/fig3_hazard/*.csv` against as the vendored/replication code changes.

## Source

Both files are derived from `bayesian-setpoint-inference/data/figures/v_obj2/fig3baseline.csv`
(model=="bayesian" rows only), generated 2026-07-29. `v_obj2` is the pipeline's current
`DATA_VERSION` (`config/opt_config.py`) -- the version whose artifacts are cited as current.

- `fig3b_hr_by_baseline.csv` -- the bayesian-only subset, as-is. Same schema as
  `outputs/fig3_hazard/fig3b_hr_by_baseline.csv`.
- `fig3a_hr_by_model.csv` -- derived from the `baseline_index == 5` subset, reshaped to
  `fig3a_hr_by_model.csv`'s schema (`hr`/`ci_lower`/`ci_upper` -> `exp(coef)`/`exp(coef) lower
  95%`/`exp(coef) upper 95%`). fig3a and fig3b's baseline_index=5 row are the same computation
  (`get_one_setpoint(..., use_personalized_logic=True, min_isolated=5)`) in both
  `bayesian-setpoint-inference/scripts/figures/fig3.py:_build_hr_baseline_df` and
  `perri-validation/scripts/run_fig3_hazard.py:build_hr_by_model` -- there's no separate fig3a
  source artifact to pull from (the live pipeline's own `cox_summary.pkl` for `v_obj2` isn't
  materialized on disk locally), so this is the correct way to get it rather than a stand-in.

No `p` column in the fig3a ground truth (not present in the fig3baseline.csv source) -- only
`exp(coef)`/CI/`n` are comparable there.

## Known, expected differences (not drift)

- `n` will not match exactly even with correct code: cohort composition depends on the exact
  Tests/Demographics tables each side runs against.
- `model` column: source `fig3baseline.csv` also has `gmm` rows (dropped here) --
  perri-validation only ever fits `bayesian` (see `scoping_v2.md`).

## Regenerating

If `bayesian-setpoint-inference`'s `DATA_VERSION` or `data/figures/<version>/fig3baseline.csv`
changes, regenerate both files from the new `fig3baseline.csv` the same way (bayesian-only rows;
fig3a = baseline_index==5 subset, reshaped).
