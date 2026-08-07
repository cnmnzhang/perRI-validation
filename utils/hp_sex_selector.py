"""Per-marker sex-stratified hyperparameter selection, vendored from
bayesian-setpoint-inference's utils/hp_sex_selector.py.

The live pipeline decides, per marker, whether to fit with pooled ("ALL") or
sex-specific ("F"/"M") hyperparameters via `is_sex_stratified()`, which merges
config/sex_stratified_markers.json's biological defaults (sex-specific
reference intervals -- e.g. HB, HCT) with a convergence assessment's
per-marker override (config/sex_stratified_markers_{SUFFIX}_{DATA_VERSION}.json,
produced by scripts/diagnostics/assess_lambda_convergence.py). That merge is
baked in here as a static list rather than re-vendoring the convergence-sweep
machinery, since perri-validation targets one fixed, published set of figures
(SUFFIX="local", DATA_VERSION="v_obj2" -- the pipeline's current default and
what generates its live fig3 artifacts) rather than re-running future sweeps.

Resolved by calling bayesian-setpoint-inference's own
utils.hp_sex_selector.is_sex_stratified() over all 43 markers under that
config: WBC, HB, HCT, RBC, ALB, ALT, ALK, TRIG, HDL, TSH, FER. Re-derive this
list (and re-check it hasn't changed) if bayesian-setpoint-inference's
DATA_VERSION or its sex_stratified_markers*.json configs are ever updated.
"""

SEX_STRATIFIED_MARKERS = frozenset({"WBC", "HB", "HCT", "RBC", "ALB", "ALT", "ALK", "TRIG", "HDL", "TSH", "FER"})


def is_sex_stratified(test_code: str) -> bool:
    """Return True if this marker should be fit with per-sex (F/M) hyperparameters
    instead of the pooled ALL row."""
    return test_code in SEX_STRATIFIED_MARKERS
