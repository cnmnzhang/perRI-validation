"""Setpoint computation via the perri package.

The Bayesian model's grid bounds (min_mu/max_mu/max_sigma) are derived live
from this repo's own pop_ri (constants/marker_config.py), not taken
unchanged from perri's bundled hyperparameters -- see _params_override(). Only
log_lambda_ still comes from perri's tuned defaults; overriding it needs a
perri-level change (see README.md's "Adapting to a different population").

A handful of markers (SEX_STRATIFIED_MARKERS below) are fit with sex-specific
("F"/"M") hyperparameters instead of the pooled "ALL" row, matching the live
pipeline -- see compute_sp_df's _compute().

A handful of markers (is_log_transform below) are fit in log-space instead of
raw units -- see _fit_batch_for_group().
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from perri.isolation import filter_isolated

from constants.runtime import DEFAULT_MIN_MEASUREMENTS, ID_COL, INDEX_COL, MEASUREMENT_COL, MODEL_COL, MU, N_JOBS, SEX_COL, SIGMA, TEST_CODE_COL, TS_COL
from utils.cache import cache_or_compute as _cache_or_compute
from utils.cache import hash_dataframe
from utils.clinical.get import popRI
from utils.io import load_tests_marker_subset
from perri import fit_batch, get_default_params
from perri import is_log_transform as _perri_is_log_transform

SP_DF_COLUMNS = [ID_COL, TEST_CODE_COL, MODEL_COL, TS_COL, MU, SIGMA, MEASUREMENT_COL, SEX_COL, INDEX_COL]

SMALL_VALUE = 0.001

# Per-marker sex-stratified hyperparameter selection, vendored from bayesian-setpoint-
# inference's utils/hp_sex_selector.py. The live pipeline decides, per marker, whether to
# fit with pooled ("ALL") or sex-specific ("F"/"M") hyperparameters via is_sex_stratified(),
# which merges config/sex_stratified_markers.json's biological defaults (sex-specific
# reference intervals -- e.g. HB, HCT) with a convergence assessment's per-marker override
# (config/sex_stratified_markers_{SUFFIX}_{DATA_VERSION}.json, produced by
# scripts/diagnostics/assess_lambda_convergence.py). That merge is baked in here as a
# static set rather than re-vendoring the convergence-sweep machinery, since perri-validation
# targets one fixed, published set of figures (SUFFIX="local", DATA_VERSION="v_obj2" -- the
# pipeline's current default and what generates its live fig3 artifacts) rather than
# re-running future sweeps. Resolved by calling bayesian-setpoint-inference's own
# utils.hp_sex_selector.is_sex_stratified() over all 43 markers under that config. Re-derive
# this set (and re-check it hasn't changed) if bayesian-setpoint-inference's DATA_VERSION or
# its sex_stratified_markers*.json configs are ever updated.
SEX_STRATIFIED_MARKERS = frozenset({"WBC", "HB", "HCT", "RBC", "ALB", "ALT", "ALK", "TRIG", "HDL", "TSH", "FER"})


def is_sex_stratified(test_code: str) -> bool:
    """Return True if this marker should be fit with per-sex (F/M) hyperparameters
    instead of the pooled ALL row."""
    return test_code in SEX_STRATIFIED_MARKERS


# No local override: perri v0.3.0's per-marker log_transform default (including TSH)
# was checked directly against this population's own ground truth (fig4_dx_cases
# hypothyroidism outcome's mu/cv odds ratios) using a fresh vendored ground-truth pull
# (data/UWM/, regenerated after bayesian-setpoint-inference's log-transform hyperparameter
# update) and matches almost exactly: log-space gives mu=1.1117/cv=1.4441 vs ground
# truth's mu=1.1116/cv=1.4365 (+0.01%/+0.53%), while raw-space diverges (+12.2%/-16.4%).
# An earlier version of this override forced TSH to raw-space based on a STALE ground-
# truth snapshot (predating that hyperparameter update, where the live pipeline itself
# still used raw-space TSH) -- that override is gone now that the ground truth has been
# refreshed and confirms perri's log-space default is correct.
is_log_transform = _perri_is_log_transform

# Empirical reference-interval fallback for one-sided markers (e.g. HDL, pop_ri
# upper bound = inf): the 95% reference interval computed from this population's
# own isolated measurements, in place of a clinical pop_ri that has no finite bound.
_LOWER_PCTL = 2.5
_UPPER_PCTL = 97.5

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def cache_or_compute(name: str, compute_fn, force: bool = False, file_format: str = "auto"):
    """cache_or_compute rooted at data/cache/: `name` is a filename
    (e.g. "sp_df_HB_<hash>_m5.csv") rather than a full path."""
    return _cache_or_compute(CACHE_DIR / name, compute_fn, force=force, file_format=file_format)


def _grid_bounds_from_pop_ri(low: float, high: float, log_space: bool = False) -> dict:
    """Derive min_mu/max_mu/min_sigma/max_sigma from a population reference interval.

    Reproduces the original pipeline's utils/get.py:expanded_ref_interval + popRR
    formula (verified against perri's bundled hyperparameters -- e.g. HB's pop_ri
    (11.5, 18.0) reproduces min_mu=1.75/max_mu=27.75/max_sigma=1.658 exactly, GLU's
    (62, 125) reproduces min_mu=0.001/max_mu=219.5/max_sigma=16.071 exactly):
      - the mu grid spans the pop_ri midpoint +/- 2x the pop_ri width
      - max_sigma is the pop_ri width scaled down by 2*1.96 (half the 95% CI multiplier)
      - min_sigma is always a small constant, not derived from pop_ri

    This is what makes editing pop_ri in constants/marker_config.py actually
    change what perri fits with -- see README.md's "Adapting to a different population".

    log_space=True means (low, high) are already log-space bounds (see
    _params_override) and min_mu is a log-space location parameter -- expected to
    go negative for any marker whose raw magnitude is below 1 (log(0.1) < 0), so
    unlike the raw-space case it is NOT floored at SMALL_VALUE. Matches the live
    pipeline's utils/get.py:get_log_model_constants.
    """
    ref_size = high - low
    center = (high + low) / 2
    min_mu = center - 2 * ref_size
    if not log_space:
        min_mu = max(SMALL_VALUE, min_mu)
    max_mu = center + 2 * ref_size
    max_sigma = ref_size / (2.0 * 1.96)
    return {"min_mu": min_mu, "max_mu": max_mu, "min_sigma": SMALL_VALUE, "max_sigma": max_sigma}


def _isolated_values(df: pd.DataFrame) -> np.ndarray:
    """This population's own isolated measurements (the same isolation filter
    perri's fit_batch applies internally, min_gap_days=90), NaNs dropped.

    `df` is already filtered to one test_code (and, for a sex-stratified group,
    to one sex) as compute_sp_df does before calling this.
    """
    isolated_values = [filter_isolated(g[MEASUREMENT_COL].to_numpy(), g[TS_COL])[0] for _, g in df.groupby(ID_COL)]
    all_values = np.concatenate(isolated_values) if isolated_values else np.array([])
    return all_values[~np.isnan(all_values)]


def _compute_popri_patch(df: pd.DataFrame) -> "tuple[float, float] | None":
    """Empirical (patch_lower, patch_upper) reference interval for a one-sided marker,
    computed from this population's own isolated measurements rather than from a
    clinical pop_ri that has no finite bound to derive grid bounds from.

    Returns None if there's no isolated data to compute a percentile from, or if the
    population is too small/degenerate for p2.5/p97.5 to span a positive width (e.g.
    a single distinct value) -- a zero-width grid divides by zero downstream in
    bayesian_model.build_prior_flat.
    """
    all_values = _isolated_values(df)
    if all_values.size == 0:
        return None
    low, high = float(np.percentile(all_values, _LOWER_PCTL)), float(np.percentile(all_values, _UPPER_PCTL))
    return (low, high) if high > low else None


def _compute_log_popri_patch(df: pd.DataFrame) -> "tuple[float, float] | None":
    """Positive-only empirical log-space (patch_lower, patch_upper), computed from
    this population's own isolated measurements -- matching the live pipeline's
    utils.patch_popri.compute_log_patch, which this repo's log-transform markers
    now use as their PREFERRED grid-bound source (see get_log_model_constants),
    ahead of (not just as a fallback for) a finite clinical pop_ri.

    Returns None if there's no isolated positive data to compute a percentile from
    (e.g. an empty or all-nonpositive population), or if the population is too
    small/degenerate for p2.5/p97.5 to span a positive width -- the caller falls
    back to deriving bounds from pop_ri in either case.
    """
    all_values = _isolated_values(df)
    positive = all_values[all_values > 0]
    if positive.size == 0:
        return None
    log_values = np.log(positive)
    low, high = float(np.percentile(log_values, _LOWER_PCTL)), float(np.percentile(log_values, _UPPER_PCTL))
    return (low, high) if high > low else None


def _params_override(test_code: str, df: pd.DataFrame, sex: str = "ALL", log_space: bool = False) -> dict:
    """Override one marker's grid parameters using the editable ``pop_ri`` read
    from ``constants/marker_lab_config.py``; retain perri's tuned ``log_lambda_``.

    For one-sided markers (pop_ri with an infinite bound, e.g. HDL), pop_ri can't
    define a finite grid width, so both bounds are replaced with an empirical
    (patch_lower, patch_upper) reference interval computed from `df`'s own isolated
    measurements (see _compute_popri_patch) -- falling back to perri's bundled
    min_mu/max_mu/min_sigma/max_sigma unchanged only if that population has no
    isolated data to compute a patch from.

    log_space=True (for markers where is_log_transform (this module) is True) prefers
    the positive-only empirical log-space patch (_compute_log_popri_patch) over a
    log-transformed pop_ri, matching the live pipeline's utils/get.py:
    get_log_model_constants now preferring utils.patch_popri's empirical log p2.5/
    p97.5 over a clinical-pop_ri-derived fallback. Only when that patch is
    unavailable (no isolated positive data for this population) do we fall back to
    log-transforming pop_ri directly -- pop_ri lower bounds are sometimes exactly 0
    (e.g. HSCRP, BILD), where log(low) is undefined; rather than clipping at an
    arbitrary raw-space floor (1e-6, which logs to ~-13.8 and has nothing to do with
    the marker's real smallest values -- confirmed to break BILD/HSCRP's fitted mu
    and CV on real data), anchor log_lo symmetrically at log_hi - 4.0, the same
    +/-4-log-unit span _grid_bounds_from_pop_ri already implies via its 2x-ref_size
    expansion.
    """
    params = get_default_params(test_code, sex=sex)
    if log_space:
        log_bounds = _compute_log_popri_patch(df)
        if log_bounds is not None:
            params.update(_grid_bounds_from_pop_ri(*log_bounds, log_space=True))
            return params
    low, high = popRI(sex=sex, test_code=test_code)
    bounds = (low, high) if math.isfinite(low) and math.isfinite(high) else _compute_popri_patch(df)
    if bounds is None:
        return params
    low, high = bounds
    if log_space:
        log_hi = math.log(max(high, 1e-6))
        log_lo = math.log(low) if low > 0 else log_hi - 4.0
        low, high = log_lo, log_hi
    params.update(_grid_bounds_from_pop_ri(low, high, log_space=log_space))
    return params


def _fit_batch_for_group(df: pd.DataFrame, test_code: str, sex_label: str, min_measurements: int, n_jobs: int) -> pd.DataFrame:
    """fit_batch for one (test_code, sex_label) group, transparently handling
    log-space fitting for markers where is_log_transform (this module) is True.

    perri itself handles log-space fitting internally (log-transforms
    measurements before fitting, back-transforms mu/sigma after via the exact
    log-normal formula, and always echoes back the original raw `value`) -- see
    perri.fit_batch's log_transform parameter. We still resolve log_space here
    (rather than leaving log_transform=None for perri.fit_batch to resolve on
    its own) so _params_override can compute grid bounds in the matching space.
    """
    log_space = is_log_transform(test_code)
    return fit_batch(
        df,
        value_col=MEASUREMENT_COL,
        timestamp_col=TS_COL,
        patient_id_col=ID_COL,
        test_code=test_code,
        sex=sex_label,
        params=_params_override(test_code, df, sex=sex_label, log_space=log_space),
        log_transform=log_space,
        min_measurements=min_measurements,
        n_jobs=n_jobs,
    )


def _cache_name_for(df_filtered_to_test_code: pd.DataFrame, test_code: str, min_measurements: int) -> str:
    """The cache filename compute_sp_df would use for this exact (already test_code-filtered)
    population -- shared by compute_sp_df itself and is_fitted (a cheap existence check with
    no fitting involved, e.g. so a caller can decide whether to reuse an already-fit
    population instead of fitting its own smaller one -- see fig5_iron_infusion, which prefers
    filtering an already-cached full-population HB fit over a fresh cohort-only fit, but only
    when that full-population fit actually exists; otherwise it still fits the cohort directly
    rather than forcing the much larger full-population fit itself).

    Cache filename embeds a content hash of the exact population being fit (+
    min_measurements), not just test_code -- different callers fitting the same marker on
    different populations (e.g. fig5_iron_infusion's cohort-filtered HB vs. fig3_hazard's
    full-population HB) get genuinely different cache files instead of fighting over one
    `sp_df_HB.csv`, which previously caused real cache thrashing (and, worse, real cached
    results getting overwritten by unrelated smoke-test runs sharing the same path).

    Also embeds a `_log` suffix when is_log_transform(test_code) (this module) is True -- so a
    marker flipping in/out of log-transformed status upstream (e.g. while investigating
    whether it should be log-transformed) produces a distinct cache file instead of
    silently overwriting the other variant's fit, letting both be kept around and compared.
    """
    population_hash = hash_dataframe(df_filtered_to_test_code)[:16]
    log_suffix = "_log" if is_log_transform(test_code) else ""
    return f"sp_df_{test_code}_{population_hash}_m{min_measurements}{log_suffix}.csv"


def is_fitted(tests_df: pd.DataFrame, test_code: str, min_measurements: int = DEFAULT_MIN_MEASUREMENTS) -> bool:
    """Whether compute_sp_df(tests_df, test_code, min_measurements=...) is already cached --
    without fitting anything. `tests_df` need not be pre-filtered to test_code."""
    df = tests_df[tests_df[TEST_CODE_COL] == test_code]
    if df.empty:
        return False
    return (CACHE_DIR / _cache_name_for(df, test_code, min_measurements)).exists()


def _full_population_cache_name(test_code: str, min_measurements: int) -> str:
    """The cache filename for a marker's full, unfiltered population -- no content-hash
    fingerprint, just test_code + min_measurements. Only valid when the caller trusts
    tests_df (when supplied) is that marker's whole splits_by_marker split with no cohort
    filtering; unlike _cache_name_for's fingerprint, this can't detect on its own that
    tests.csv changed underneath it -- delete data/cache/, or pass force=True, after it does.

    The point: checking is_fitted_full_population needs only test_code, not a loaded/hashed
    DataFrame, so a caller fitting many markers (fit_markers) can skip loading a
    marker's multi-hundred-MB CSV split entirely when it's already fitted.

    Also embeds a `_log` suffix when is_log_transform(test_code) -- see _cache_name_for's
    docstring for why.
    """
    log_suffix = "_log" if is_log_transform(test_code) else ""
    return f"sp_df_{test_code}_full_m{min_measurements}{log_suffix}.csv"


def is_fitted_full_population(test_code: str, min_measurements: int = DEFAULT_MIN_MEASUREMENTS) -> bool:
    """Whether the marker's full-population cache file already exists --
    no tests_df needed at all, unlike is_fitted."""
    return (CACHE_DIR / _full_population_cache_name(test_code, min_measurements)).exists()


def fit_markers(input_dir, test_codes: list, *, force: bool = False, min_measurements: int = DEFAULT_MIN_MEASUREMENTS, label: str = "setpoints") -> pd.DataFrame:
    """Fits every marker in test_codes via compute_sp_df, in order, skipping (not aborting
    the whole run for) a marker with no/insufficient data or an unexpected fit failure (a bad
    pop_ri, a malformed row, ...). Shared by scripts.build_setpoints (explicit pre-fit
    stage covering all of TESTCODES_LIST), run_fig3_hazard, and run_fig3_dx.

    Loads a marker's Tests split off disk only when its full_population cache isn't already there,
    instead of loading every marker's full split up front. Lets a fully-warm cache (e.g. from
    a prior build_setpoints) skip reading every multi-hundred-MB per-marker CSV just
    to recheck a cache that's already known to exist -- the tradeoff being that a stale
    full_population cache isn't auto-detected the way a fingerprinted one is (see `_full_population_cache_name`).
    """
    from utils.io import load_tests_marker_subset

    sp_frames = []
    for i, test_code in enumerate(test_codes, 1):
        t0 = time.time()
        already_fitted = not force and is_fitted_full_population(test_code, min_measurements)
        tests_df = None if already_fitted else load_tests_marker_subset(input_dir, test_codes=[test_code])
        try:
            sp = compute_sp_df(tests_df, test_code=test_code, force=force, min_measurements=min_measurements, full_population=True)
        except Exception as exc:
            print(f"{label}: [{i}/{len(test_codes)}] {test_code}: SKIPPED, fit failed: {exc}")
            continue
        n_fitted = sp[ID_COL].nunique() if not sp.empty else 0
        candidates = f"{tests_df[ID_COL].nunique():,} candidate patients" if tests_df is not None else "cached"
        print(f"{label}: [{i}/{len(test_codes)}] {test_code}: {candidates} -> {n_fitted:,} fitted ({time.time() - t0:.1f}s)")
        sp_frames.append(sp)
    non_empty = [f for f in sp_frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=SP_DF_COLUMNS)
    return pd.concat(non_empty, ignore_index=True)


def compute_sp_df(
    tests_df: "pd.DataFrame | None",
    test_code: str,
    *,
    min_measurements: int = DEFAULT_MIN_MEASUREMENTS,
    force: bool = False,
    n_jobs: int = N_JOBS,
    full_population: bool = False,
    input_dir: "str | Path | None" = None,
) -> pd.DataFrame:
    """Compute setpoints for one marker from a generic Tests table, via perri.

    This is the expensive step (a Bayesian filter fit per patient) — the result
    is cached at data/cache/sp_df_<test_code>_<population_hash>_m<N>.csv,
    where population_hash is a content hash of the exact filtered input rows.
    Cases: 
    - Re-running an analysis (or iterating on downstream cohort/plotting code)
    doesn't refit the model unless the input Tests rows for that marker actually
    changed
    - fitting the same marker on two different populations (e.g. a full
    population vs. a cohort-filtered subset) produces two independent cache
    files rather than one clobbering the other.


    Parameters
    ----------
    tests_df : Tests table (anon_id, ts, test_code, result_value, sex), already
        filtered or not — this function filters to `test_code` internally. May be
        left as None when `full_population=True` and the cache already has this marker
        (see `full_population` below) — it's only read on a cache miss.
    test_code : marker to fit (e.g. "HB", "WBC").
    force : recompute even if a cached result exists.
    full_population : use the marker's full-population, fingerprint-free cache file (see
        `_full_population_cache_name`) instead of hashing tests_df. Only valid when
        tests_df (when supplied) is that marker's whole, unfiltered population --
        see `fit_markers`, which is the intended caller.
    input_dir : where to load `test_code`'s per-marker split from (via
        load_tests_marker_subset) on a full_population cache miss when `tests_df` is
        None. None resolves to the repo's default `data/` dir (see
        `splits_by_marker_dir`). Ignored when `tests_df` is supplied or the
        full_population cache already hits.

    Returns
    -------
    DataFrame shaped like the internal pipeline's sp_df: columns
    [anon_id, test_code, model, ts, mu, sigma, result_value, sex, index].
    """
    if full_population:
        cache_name = _full_population_cache_name(test_code, min_measurements)
    else:
        df = tests_df[tests_df[TEST_CODE_COL] == test_code]
        if df.empty:
            return pd.DataFrame(columns=SP_DF_COLUMNS)
        cache_name = _cache_name_for(df, test_code, min_measurements)

    def _compute() -> pd.DataFrame:
        nonlocal tests_df
        if tests_df is None:
            tests_df = load_tests_marker_subset(input_dir, test_codes=[test_code])
        df = tests_df[tests_df[TEST_CODE_COL] == test_code].copy()
        if df.empty:
            return pd.DataFrame(columns=SP_DF_COLUMNS)

        # fit_batch's `sex` is applied uniformly to every patient in `df` -- for
        # sex-stratified markers it must be called once per sex on a sex-filtered
        # slice (per fit_batch's own docstring), not once for the whole population.
        if is_sex_stratified(test_code):
            fitted_frames = []
            for sex_label in ("F", "M"):
                sex_df = df[df[SEX_COL] == sex_label]
                if sex_df.empty:
                    continue
                fitted_frames.append(_fit_batch_for_group(sex_df, test_code, sex_label, min_measurements, n_jobs))
            fitted = pd.concat(fitted_frames, ignore_index=True) if fitted_frames else pd.DataFrame()
        else:
            fitted = _fit_batch_for_group(df, test_code, "ALL", min_measurements, n_jobs)
        if fitted.empty:
            return pd.DataFrame(columns=SP_DF_COLUMNS)

        # Rename perri's output columns to match the internal pipeline's sp_df schema
        out = fitted.rename(
            columns={
                "patient_id": ID_COL,
                "timestamp": TS_COL,
                "value": MEASUREMENT_COL,
                "measurement_index": INDEX_COL,
            }
        )
        sex_lookup = df[[ID_COL, "sex"]].drop_duplicates(ID_COL)
        out = out.merge(sex_lookup, on=ID_COL, how="left")

        out[TEST_CODE_COL] = test_code
        out["model"] = "bayesian"
        return out[SP_DF_COLUMNS]

    out = cache_or_compute(
        cache_name,
        _compute,
        force=force,
        file_format="csv",
    )
    # cache_or_compute round-trips through CSV on a cache hit, which loses dtypes
    # (ts comes back as a string) 
    # test_code is set unconditionally to avoid turning the marker "NA" (sodium) into NaN
    out[ID_COL] = out[ID_COL].astype(str)
    out[TS_COL] = pd.to_datetime(out[TS_COL])
    out[TEST_CODE_COL] = test_code
    return out
