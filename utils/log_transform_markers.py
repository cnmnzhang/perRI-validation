"""Per-marker log-space fitting, vendored from bayesian-setpoint-inference's
utils/setpoints_runner.py:run_patient_from_dict + utils/get.py:get_log_model_constants.

The live pipeline fits some right-skewed markers in log-space (log-transform the
raw measurements, fit, then back-transform mu/sigma) rather than raw units, per
marker, when doing so gave lower fitting error -- recorded as a `log_transform`
column in its best_trials table, not a static property of the marker (a marker
could in principle flip between DATA_VERSIONs/tuning runs).

Resolved by reading bayesian-setpoint-inference's
data/share/best_trials_merged_v_obj2.csv, where every sex row for FER, GLU,
HSCRP, TRIG, and TSH has log_transform=True.

TSH is a live open question, not a settled one -- a direct comparison against
data/UWM/'s actual TSH.pkl snapshot found its extreme/outlier raw measurements
clamp at mu=11.9, sigma=1.173469, exactly the *raw*-space grid bounds (perri's
own bundled bayesian_hyperparameters.csv has these same numbers for TSH), not
the ~221 raw-space clamp that log-transforming pop_ri (0.4, 5.0) before
deriving the grid would produce -- and fitting TSH without log-space treatment
dropped the mu diff against data/UWM/ from mean 0.35 (max 176) to mean 0.004
(max 4.27, 90.6% of rows within 0.01) at the setpoint level. One plausible
explanation: generate_sp_df_from_params reads its `log_transform` flag via
`hp_row.iloc[0].get("log_transform", "")` from whichever params_df it's given,
and data/share/bayesian_hyperparameters_v_obj2.csv (which does have matching
raw-space min_mu/max_mu/sigma columns) has no `log_transform` column at all --
so whatever generated this particular TSH.pkl snapshot may have silently
defaulted to raw-space, regardless of what best_trials_merged records.
TSH was put back to raw-space (removed from this set) after a second direct
check, at the fig4_dx_cases hypothyroidism outcome's cv odds ratio: raw-space
gives 1.2004 against ground truth's 1.1989 (+0.1%, odds ratio/CI/x_mean/x_std
all matching to 2-3 decimals), while log-transform reproduces the original
+26.6% divergence (1.5181) this whole investigation started from. fig3_hazard's
HR-level check was less conclusive either way (mu HR 1.147 raw vs 1.173 log vs
ground truth's 1.113; cv HR 0.929 raw vs 1.024 log vs ground truth's 0.980 --
raw closer on both, but neither exact), but fig4's near-exact match under
raw-space settled it. See setpoints.py's `_cache_name_for`/
`_canonical_cache_name` `_log` suffix, which lets both the log-transformed and
raw-space TSH fits stay cached side by side so toggling TSH in/out of this set
for comparison doesn't require re-fitting from scratch each time.
FER/GLU/HSCRP/TRIG haven't been re-checked against this same direct-comparison
method; if their own residuals ever look TSH-sized, re-run this check for them too.

Baked in as a static list for the same reason utils/hp_sex_selector.py's
SEX_STRATIFIED_MARKERS is: this package targets one fixed, published set of
figures, not future re-tuning runs. Re-derive this list if
bayesian-setpoint-inference's DATA_VERSION or best_trials selection changes.

Note this is a different (and non-overlapping in origin, though overlapping in
membership -- FER and TRIG are both) set from constants/marker_lab_config.py's
static "log_transform" flag, which is unused here on purpose: that flag marks
markers as *candidates* the live pipeline's tuning sweep considered for log-space
fitting, not which ones actually won for the current DATA_VERSION.
"""

LOG_TRANSFORM_MARKERS = frozenset({"FER", "GLU", "HSCRP", "TRIG"})


def is_log_transform(test_code: str) -> bool:
    """Return True if this marker should be fit in log-space (measurements
    log-transformed before fitting, mu/sigma back-transformed after)."""
    return test_code in LOG_TRANSFORM_MARKERS
