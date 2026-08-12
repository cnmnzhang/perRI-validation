"""Setpoint computation via the perri package.

The Bayesian model's grid bounds (min_mu/max_mu/max_sigma) are derived live
from this repo's own pop_ri (constants/marker_config.py), not taken
unchanged from perri's bundled hyperparameters -- see _params_override(). Only
log_lambda_ still comes from perri's tuned defaults; overriding it needs a
perri-level change (see README.md's "Adapting to a different population").

A handful of markers (utils/hp_sex_selector.SEX_STRATIFIED_MARKERS) are fit
with sex-specific ("F"/"M") hyperparameters instead of the pooled "ALL" row,
matching the live pipeline -- see compute_sp_df's _compute().

A handful of markers (utils/log_transform_markers.LOG_TRANSFORM_MARKERS) are fit
in log-space instead of raw units -- see _fit_batch_for_group().
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
from utils.hp_sex_selector import is_sex_stratified
from utils.io import load_tests_marker_subset
from utils.log_transform_markers import is_log_transform
from perri import fit_batch, get_default_params

SP_DF_COLUMNS = [ID_COL, TEST_CODE_COL, MODEL_COL, TS_COL, MU, SIGMA, MEASUREMENT_COL, SEX_COL, INDEX_COL]

SMALL_VALUE = 0.001

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


def _grid_bounds_from_pop_ri(low: float, high: float) -> dict:
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
    """
    ref_size = high - low
    center = (high + low) / 2
    min_mu = max(SMALL_VALUE, center - 2 * ref_size)
    max_mu = center + 2 * ref_size
    max_sigma = ref_size / (2.0 * 1.96)
    return {"min_mu": min_mu, "max_mu": max_mu, "min_sigma": SMALL_VALUE, "max_sigma": max_sigma}


def _compute_popri_patch(df: pd.DataFrame) -> "tuple[float, float] | None":
    """Empirical (patch_lower, patch_upper) reference interval for a one-sided marker,
    computed from this population's own isolated measurements (the same isolation
    filter perri's fit_batch applies internally, min_gap_days=90) rather than from
    a clinical pop_ri that has no finite bound to derive grid bounds from.

    `df` is already filtered to one test_code (as compute_sp_df does before calling
    this). Returns None if there's no isolated data to compute a percentile from.
    """
    isolated_values = [filter_isolated(g[MEASUREMENT_COL].to_numpy(), g[TS_COL])[0] for _, g in df.groupby(ID_COL)]
    all_values = np.concatenate(isolated_values) if isolated_values else np.array([])
    all_values = all_values[~np.isnan(all_values)]
    if all_values.size == 0:
        return None
    return float(np.percentile(all_values, _LOWER_PCTL)), float(np.percentile(all_values, _UPPER_PCTL))


def _params_override(test_code: str, df: pd.DataFrame, sex: str = "ALL", log_space: bool = False) -> dict:
    """Override one marker's grid parameters using the editable ``pop_ri`` read
    from ``constants/marker_lab_config.py``; retain perri's tuned ``log_lambda_``.

    For one-sided markers (pop_ri with an infinite bound, e.g. HDL), pop_ri can't
    define a finite grid width, so both bounds are replaced with an empirical
    (patch_lower, patch_upper) reference interval computed from `df`'s own isolated
    measurements (see _compute_popri_patch) -- falling back to perri's bundled
    min_mu/max_mu/min_sigma/max_sigma unchanged only if that population has no
    isolated data to compute a patch from.

    log_space=True (for utils.log_transform_markers.LOG_TRANSFORM_MARKERS) log-
    transforms the resolved (low, high) bounds before deriving the grid, matching
    the live pipeline's utils/get.py:get_log_model_constants -- clipped at 1e-6
    the same way it clips there, since pop_ri lower bounds are sometimes exactly 0.
    """
    params = get_default_params(test_code, sex=sex)
    low, high = popRI(sex=sex, test_code=test_code)
    bounds = (low, high) if math.isfinite(low) and math.isfinite(high) else _compute_popri_patch(df)
    if bounds is None:
        return params
    low, high = bounds
    if log_space:
        low, high = math.log(max(low, 1e-6)), math.log(max(high, 1e-6))
    params.update(_grid_bounds_from_pop_ri(low, high))
    return params


def _fit_batch_for_group(df: pd.DataFrame, test_code: str, sex_label: str, min_measurements: int, n_jobs: int) -> pd.DataFrame:
    """fit_batch for one (test_code, sex_label) group, transparently handling
    log-space fitting for utils.log_transform_markers.LOG_TRANSFORM_MARKERS.

    Mirrors the live pipeline's utils/setpoints_runner.py:run_patient_from_dict:
    measurements are log-transformed before fitting (clipped at 1e-6, matching
    _params_override's grid-bound clipping), and mu/sigma are back-transformed via
    the exact log-normal formula after fitting -- mu_raw = exp(mu_log), sigma_raw =
    mu_raw * sqrt(exp(sigma_log^2) - 1). perri's fit_batch echoes back whatever
    `value` it was fit on, so the fitted `value` column (log-space) is swapped back
    for the original raw measurement per (patient, timestamp) -- the live pipeline's
    sp_df always stores raw-unit measurements regardless of which space mu/sigma
    were fit in.
    """
    log_space = is_log_transform(test_code)
    fit_df = df
    if log_space:
        fit_df = df.copy()
        fit_df[MEASUREMENT_COL] = np.log(np.clip(fit_df[MEASUREMENT_COL].to_numpy(), 1e-6, None))
    fitted = fit_batch(
        fit_df,
        value_col=MEASUREMENT_COL,
        timestamp_col=TS_COL,
        patient_id_col=ID_COL,
        test_code=test_code,
        sex=sex_label,
        params=_params_override(test_code, df, sex=sex_label, log_space=log_space),
        min_measurements=min_measurements,
        n_jobs=n_jobs,
    )
    if fitted.empty or not log_space:
        return fitted
    raw_lookup = df.rename(columns={ID_COL: "patient_id", TS_COL: "timestamp", MEASUREMENT_COL: "value"})[["patient_id", "timestamp", "value"]]
    fitted = fitted.drop(columns=["value"]).merge(raw_lookup, on=["patient_id", "timestamp"], how="left")
    fitted["sigma"] = np.exp(fitted["mu"]) * np.sqrt(np.exp(fitted["sigma"] ** 2) - 1)
    fitted["mu"] = np.exp(fitted["mu"])
    return fitted


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

    Also embeds a `_log` suffix when utils.log_transform_markers.is_log_transform(test_code)
    -- so flipping a marker in/out of LOG_TRANSFORM_MARKERS (e.g. while investigating whether
    it should be log-transformed) produces a distinct cache file instead of silently
    overwriting the other variant's fit, letting both be kept around and compared.
    """
    population_hash = hash_dataframe(df_filtered_to_test_code)[:16]
    log_suffix = "_log" if is_log_transform(test_code) else ""
    return f"sp_df_{test_code}_{population_hash}_m{min_measurements}{log_suffix}.csv"


def is_fitted(tests_df: pd.DataFrame, test_code: str, min_measurements: int = DEFAULT_MIN_MEASUREMENTS) -> bool:
    """Whether compute_sp_df(tests_df, test_code, min_measurements=...) is already cached --
    without fitting anything. `tests_df` need not be pre-filtered to test_code."""
    df = tests_df[tests_df["test_code"] == test_code]
    if df.empty:
        return False
    return (CACHE_DIR / _cache_name_for(df, test_code, min_measurements)).exists()


def _canonical_cache_name(test_code: str, min_measurements: int) -> str:
    """The cache filename for a marker's full, unfiltered population -- no content-hash
    fingerprint, just test_code + min_measurements. Only valid when the caller trusts
    tests_df (when supplied) is that marker's whole tests_by_marker split with no cohort
    filtering; unlike _cache_name_for's fingerprint, this can't detect on its own that
    tests.csv changed underneath it -- delete data/cache/, or pass force=True, after it does.

    The point: checking is_fitted_canonical needs only test_code, not a loaded/hashed
    DataFrame, so a caller fitting many markers (fit_markers_lazy) can skip loading a
    marker's multi-hundred-MB CSV split entirely when it's already fitted.

    Also embeds a `_log` suffix when is_log_transform(test_code) -- see _cache_name_for's
    docstring for why.
    """
    log_suffix = "_log" if is_log_transform(test_code) else ""
    return f"sp_df_{test_code}_full_m{min_measurements}{log_suffix}.csv"


def is_fitted_canonical(test_code: str, min_measurements: int = DEFAULT_MIN_MEASUREMENTS) -> bool:
    """Whether the marker's canonical (full-population) cache file already exists --
    no tests_df needed at all, unlike is_fitted."""
    return (CACHE_DIR / _canonical_cache_name(test_code, min_measurements)).exists()


def fit_markers(tests_df: pd.DataFrame, test_codes: list, *, force: bool = False, min_measurements: int = DEFAULT_MIN_MEASUREMENTS, label: str = "setpoints") -> pd.DataFrame:
    """Fits every marker in test_codes via compute_sp_df, in order, skipping (not aborting
    the whole run for) a marker with no/insufficient data or an unexpected fit failure (a bad
    pop_ri, a malformed row, ...). Shared by
    scripts.run_setpoints_by_marker (explicit pre-fit stage covering all of
    TESTCODES_LIST) and run_fig3_hazard (which needs the same 43 markers fit either way,
    whether or not the shared stage already warmed the cache).
    """
    sp_frames = []
    for i, test_code in enumerate(test_codes, 1):
        n_patients = tests_df.loc[tests_df["test_code"] == test_code, ID_COL].nunique()
        t0 = time.time()
        try:
            sp = compute_sp_df(tests_df, test_code=test_code, force=force, min_measurements=min_measurements)
        except Exception as exc:
            print(f"{label}: [{i}/{len(test_codes)}] {test_code}: SKIPPED, fit failed: {exc}")
            continue
        print(f"{label}: [{i}/{len(test_codes)}] {test_code}: {n_patients:,} candidate patients -> {sp[ID_COL].nunique() if not sp.empty else 0:,} fitted ({time.time() - t0:.1f}s)")
        sp_frames.append(sp)
    non_empty = [f for f in sp_frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=SP_DF_COLUMNS)
    return pd.concat(non_empty, ignore_index=True)


def fit_markers_lazy(input_dir, test_codes: list, *, force: bool = False, min_measurements: int = DEFAULT_MIN_MEASUREMENTS, label: str = "setpoints") -> pd.DataFrame:
    """Like fit_markers, but loads a marker's Tests split off disk only when its canonical
    cache isn't already there, instead of loading every marker's full split up front. Lets a
    fully-warm cache (e.g. from a prior run_setpoints_by_marker) skip reading every
    multi-hundred-MB per-marker CSV just to recheck a cache that's already known to exist --
    the tradeoff being that a stale canonical cache isn't auto-detected the way fit_markers'
    fingerprinted one is (see `_canonical_cache_name`).
    """
    from utils.io import load_tests_marker_subset

    sp_frames = []
    for i, test_code in enumerate(test_codes, 1):
        t0 = time.time()
        already_fitted = not force and is_fitted_canonical(test_code, min_measurements)
        tests_df = None if already_fitted else load_tests_marker_subset(input_dir, test_codes=[test_code])
        try:
            sp = compute_sp_df(tests_df, test_code=test_code, force=force, min_measurements=min_measurements, canonical=True)
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
    canonical: bool = False,
    input_dir: "str | Path | None" = None,
) -> pd.DataFrame:
    """Compute setpoints for one marker from a generic Tests table, via perri.

    This is the expensive step (a Bayesian filter fit per patient) — the result
    is cached at data/cache/sp_df_<test_code>_<population_hash>_m<N>.csv,
    where population_hash is a content hash of the exact filtered input rows.
    Re-running an analysis (or iterating on downstream cohort/plotting code)
    doesn't refit the model unless the input Tests rows for that marker actually
    changed; fitting the same marker on two different populations (e.g. a full
    population vs. a cohort-filtered subset) produces two independent cache
    files rather than one clobbering the other.

    Column names are not parameterized: utils/io.py's loaders
    guarantee the Tests table always uses anon_id/ts/test_code/result_value/sex
    (sex is a required column, not merged in from Demographics), so there's
    nothing to vary here.

    Parameters
    ----------
    tests_df : Tests table (anon_id, ts, test_code, result_value, sex), already
        filtered or not — this function filters to `test_code` internally. May be
        left as None when `canonical=True` and the cache already has this marker
        (see `canonical` below) — it's only read on a cache miss.
    test_code : marker to fit (e.g. "HB", "WBC").
    force : recompute even if a cached result exists.
    canonical : use the marker's canonical, fingerprint-free cache file (see
        `_canonical_cache_name`) instead of hashing tests_df. Only valid when
        tests_df (when supplied) is that marker's whole, unfiltered population --
        see `fit_markers_lazy`, which is the intended caller.
    input_dir : where to load `test_code`'s per-marker split from (via
        load_tests_marker_subset) on a canonical cache miss when `tests_df` is
        None. None resolves to the repo's default `data/` dir (see
        `tests_by_marker_dir`). Ignored when `tests_df` is supplied or the
        canonical cache already hits.

    Returns
    -------
    DataFrame shaped like the internal pipeline's sp_df: columns
    [anon_id, test_code, model, ts, mu, sigma, result_value, sex, index].
    """
    if canonical:
        cache_name = _canonical_cache_name(test_code, min_measurements)
    else:
        df = tests_df[tests_df["test_code"] == test_code]
        if df.empty:
            return pd.DataFrame(columns=SP_DF_COLUMNS)
        cache_name = _cache_name_for(df, test_code, min_measurements)

    def _compute() -> pd.DataFrame:
        nonlocal tests_df
        if tests_df is None:
            tests_df = load_tests_marker_subset(input_dir, test_codes=[test_code])
        df = tests_df[tests_df["test_code"] == test_code].copy()
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

        out["test_code"] = test_code
        out["model"] = "bayesian"
        return out[SP_DF_COLUMNS]

    out = cache_or_compute(
        cache_name,
        _compute,
        force=force,
        file_format="csv",
    )
    # cache_or_compute round-trips through CSV on a cache hit, which loses dtypes
    # (ts comes back as a string) -- restore them so callers never have to care
    # whether this was freshly computed or loaded from cache. test_code is set
    # unconditionally (not just re-typed) because pandas' default NA-string
    # sniffing on a bare read_csv (cache_or_compute's csv path has no dtype/
    # na_values guard) turns the marker "NA" (sodium) into a real NaN
    out[ID_COL] = out[ID_COL].astype(str)
    out[TS_COL] = pd.to_datetime(out[TS_COL])
    out["test_code"] = test_code
    return out
