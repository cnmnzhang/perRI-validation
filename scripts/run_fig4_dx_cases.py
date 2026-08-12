"""Fig4 dx cases: aki, leukemia, hypothyroidism.

For each outcome, builds an anchor->presenting cohort from setpoints (computed
via perri, not a pre-built sp_df pickle), fits an age/sex-adjusted logistic
odds-ratio model, classifies patients into the PerRI x PopRI 2x2, and plots
Kaplan-Meier curves for the outcome. Produces ONE combined figure with one row
per outcome and four columns (trajectory, forest, KM, reclassification
heatmap), using utils/visuals_fig4.py's rendering functions
directly (not a simplified re-implementation).

Required inputs: `tests.csv` (CRE, WBC, TSH, T4FR rows), a Demographics table,
and the *derived* Dx table produced by `scripts.run_dx_incident` (`dx_incident.csv`)
— see README.md. Run dx_incident first: `python -m scripts.run_dx_incident`.
This script will raise a clear error naming the expected path if that file
isn't there yet. Reads its markers from the per-marker split built by
`scripts.run_tests_by_marker` -- run that first too (or use
`run_all`, which sequences both automatically); raises a clear FileNotFoundError
with the command to run if it hasn't been built yet.

Run:
    python -m scripts.run_fig4_dx_cases --input-dir data --output-dir outputs/fig4_dx_cases
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.bootstrap import ensure_importable

ensure_importable()

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import numpy as np  # noqa: E402

from utils.io import load_demographics_csv, load_dx_incident, load_tests_marker_subset  # noqa: E402
from utils.logging_utils import tagged_stdout  # noqa: E402
from utils.visuals_shared import save_fig_as_svg  # noqa: E402
from utils.setpoints import compute_sp_df, is_fitted_canonical  # noqa: E402
from utils.cache import cache_or_compute  # noqa: E402
from utils.clinical import get as _get  # noqa: E402
from utils.clinical.get import attach_ref_intervals  # noqa: E402
from constants.fig_config import A4_WIDTH, FIG4_ROW1_HEIGHT  # noqa: E402
from utils.km_exclusive import KMExclusiveInputs  # noqa: E402
from utils.progression.config import OUTCOME_REGISTRY  # noqa: E402
from utils.progression.core import (  # noqa: E402
    compute_popri_continuous,
    load_or_create_analysis_ready_cohort,
)
from utils.progression.logit import fit_logit_and_report  # noqa: E402
from constants.runtime import (  # noqa: E402
    CV_COL,
    DELTA_COL,
    ID_COL,
    INDEX_COL,
    MEASUREMENT_COL,
    MU,
    OUT_PERRI_P95_COL,
    OUT_PERRI_P95_LOWER_COL,
    OUT_PERRI_P95_UPPER_COL,
    OUT_POPRI_COL,
    OUT_POPRI_LOWER_COL,
    OUT_POPRI_UPPER_COL,
    PERRI_Z_SCORE_COL,
    PRESENT_TS_COL,
    PRESENT_VAL_COL,
    SEX_COL,
    SIGMA,
    TS_COL,
)
from utils.clinical.inputs import ProgressionPanelInputs  # noqa: E402
from utils.clinical.run_clinical import build_added_value_tables, build_reclassification_2x2_grids  # noqa: E402
from utils.visuals_fig4 import (  # noqa: E402
    fig4forest_on_ax,
    fig4heatmap_2x2_on_ax,
    fig4km_exclusive_on_ax,
    plot_single_patient_history_on_ax,
)
DEMOGRAPHICS_FILE = "demographics.csv"

# Manually-selected example patients, written by `scripts.run_fig4_dx_cases_case --accept N`.
# Kept outside any single --output-dir so a saved case survives across output-dir choices.
CASES_JSON_PATH = Path(__file__).resolve().parent.parent / "outputs" / "fig4_dx_cases_cases.json"
BASELINE_SORT_N = 3

def _outcome_markers(outcome_cfg) -> list[str]:
    """One outcome's own markers (outcome_cfg.markers, used by compute_sp_df) plus any
    additional marker referenced by a "lab_threshold" OutcomeDefinition (e.g. hypothyroidism's
    t4fr_low uses T4FR, which compute_sp_df never touches -- it's read directly off tests_df
    by flag_outcomes_from_config)."""
    return sorted(set(outcome_cfg.markers) | {out.marker for out in outcome_cfg.outcomes if out.type == "lab_threshold"})


# Every marker any outcome in OUTCOME_REGISTRY needs -- used only for the manifest/docs, not
# for loading (each outcome now loads just its own markers, lazily -- see compute_one_outcome).
FIG4_DX_CASES_MARKERS = sorted({m for cfg in OUTCOME_REGISTRY.values() for m in _outcome_markers(cfg)})


def _direction_columns(lower: bool, upper: bool) -> tuple[str, str]:
    if lower and upper:
        return OUT_PERRI_P95_COL, OUT_POPRI_COL
    if lower:
        return OUT_PERRI_P95_LOWER_COL, OUT_POPRI_LOWER_COL
    return OUT_PERRI_P95_UPPER_COL, OUT_POPRI_UPPER_COL


# ---------------------------------------------------------------------------
# Example-patient case selection (saved-case / group-B / naive-fallback tiers).
# Mirrors bayesian-setpoint-inference's scripts/figures/fig4_prog.py +
# fig4_prog_case.py case-selection machinery; scripts/run_fig4_dx_cases_case.py
# is this repo's counterpart to fig4_prog_case.py for browsing/saving cases.
# ---------------------------------------------------------------------------


def _build_group_masks(df: pd.DataFrame, per_col: str, pop_col: str) -> dict[str, pd.Series]:
    """KM-exclusive 2x2 group masks (mirrors KMExclusiveInputs.from_dataframe's masks)."""
    pop_in, pop_out = df[pop_col] == 0, df[pop_col] == 1
    per_in, per_out = df[per_col] == 0, df[per_col] == 1
    return {"a": pop_in & per_in, "b": pop_in & per_out, "c": pop_out & per_in, "d": pop_out & per_out}


def _load_cases() -> dict:
    if CASES_JSON_PATH.exists():
        return json.loads(CASES_JSON_PATH.read_text())
    return {}


def _save_case(outcome_name: str, patient_id: str) -> None:
    """Write patient_id as the selected case for outcome_name to CASES_JSON_PATH."""
    CASES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    cases[outcome_name] = str(patient_id)
    CASES_JSON_PATH.write_text(json.dumps(cases, indent=2))
    print(f"[case] Saved: {outcome_name} -> {patient_id}\n[case] Written to {CASES_JSON_PATH}")


def _safe_z_sort(df: pd.DataFrame) -> pd.Series:
    if SIGMA in df.columns:
        sigma_safe = pd.to_numeric(df[SIGMA], errors="coerce").replace(0, np.nan)
        return (pd.to_numeric(df[DELTA_COL], errors="coerce") / sigma_safe).abs()
    return pd.to_numeric(df[DELTA_COL], errors="coerce").abs()


def _add_baseline_sort_columns(
    candidates: pd.DataFrame,
    sp_df: pd.DataFrame,
    test_code: str,
    *,
    id_col: str = ID_COL,
    baseline_n: int = BASELINE_SORT_N,
) -> pd.DataFrame:
    """Add baseline-aware ranking columns for case selection.

    Ranking prefers patients whose isolated values immediately before the presenting
    value are all inside PopRI -- this avoids picking extreme low-variance baselines
    solely because they inflate |delta/sigma|.
    """
    out = candidates.copy()
    if out.empty:
        return out

    out["_z_sort"] = _safe_z_sort(out)
    out["_delta_abs_sort"] = pd.to_numeric(out[DELTA_COL], errors="coerce").abs()
    out["_baseline_popri_ok"] = False
    out["_pre_all_popri_ok"] = False
    out["_trajectory_n"] = 0
    out["_event_lag_days"] = np.nan

    if "first_in_window_event" in out.columns and PRESENT_TS_COL in out.columns:
        event_ts = pd.to_datetime(out["first_in_window_event"], errors="coerce")
        presenting_ts = pd.to_datetime(out[PRESENT_TS_COL], errors="coerce")
        out["_event_lag_days"] = (event_ts - presenting_ts).dt.days

    required = {id_col, TS_COL, MEASUREMENT_COL}
    if sp_df is None or not required.issubset(sp_df.columns) or PRESENT_TS_COL not in out.columns:
        return out

    sp_local = sp_df.copy()
    sp_local[id_col] = sp_local[id_col].astype(str)
    sp_local[TS_COL] = pd.to_datetime(sp_local[TS_COL], errors="coerce")
    sp_local = sp_local.sort_values([id_col, TS_COL])
    grouped = {pid: pat for pid, pat in sp_local.groupby(id_col, sort=False)}

    for idx, row in out.iterrows():
        patient_id = str(row[id_col])
        patient_sp = grouped.get(patient_id)
        if patient_sp is None or patient_sp.empty:
            continue

        presenting_ts = pd.to_datetime(row[PRESENT_TS_COL], errors="coerce")
        if pd.isna(presenting_ts):
            continue

        before = patient_sp[patient_sp[TS_COL] < presenting_ts]
        same_day = patient_sp[patient_sp[TS_COL].dt.normalize().eq(presenting_ts.normalize())]
        if not same_day.empty and INDEX_COL in patient_sp.columns:
            presenting_index = same_day.iloc[0].get(INDEX_COL, np.nan)
            if pd.notna(presenting_index):
                before = patient_sp[pd.to_numeric(patient_sp[INDEX_COL], errors="coerce") < float(presenting_index)]

        baseline = pd.to_numeric(before.tail(baseline_n)[MEASUREMENT_COL], errors="coerce").dropna()
        if baseline.empty:
            continue

        sex = row.get(SEX_COL, patient_sp[SEX_COL].iloc[0] if SEX_COL in patient_sp.columns else "ALL")
        if pd.isna(sex):
            sex = "ALL"
        pop_lo, pop_hi = _get.popRI(sex=sex, test_code=test_code)
        pre_values = pd.to_numeric(before[MEASUREMENT_COL], errors="coerce").dropna()
        out.loc[idx, "_trajectory_n"] = int(len(patient_sp))
        out.loc[idx, "_baseline_popri_ok"] = bool(len(baseline) >= baseline_n and ((baseline >= pop_lo) & (baseline <= pop_hi)).all())
        out.loc[idx, "_pre_all_popri_ok"] = bool(len(pre_values) >= baseline_n and ((pre_values >= pop_lo) & (pre_values <= pop_hi)).all())

    return out


def find_candidates(
    presenting_df: pd.DataFrame,
    sp_df: pd.DataFrame,
    outcome_cfg,
    *,
    group: str = "b",
    events_only: bool = True,
) -> pd.DataFrame:
    """Return candidate rows sorted by baseline plausibility, then |delta|.

    Used both by the tiered example-patient selector below and by
    scripts/run_fig4_dx_cases_case.py for interactive browsing.
    """
    per_col, pop_col = _direction_columns(outcome_cfg.flag_below, outcome_cfg.flag_above)
    missing = [c for c in [pop_col, per_col, DELTA_COL] if c not in presenting_df.columns]
    if missing:
        raise KeyError(f"Columns missing from presenting_df: {missing}")

    df = presenting_df.copy()
    if group != "all":
        masks = _build_group_masks(df, per_col, pop_col)
        if group not in masks:
            raise ValueError(f"Unknown group {group!r}.")
        df = df[masks[group]].copy()

    if events_only and "any_in_window" in df.columns:
        df = df[df["any_in_window"] == 1].copy()

    # Keep only "timely" presenters -- patients whose flag appeared at (or shortly
    # after) their first eligible presenting test, not years after eligibility opened.
    if "anchor_ts" in df.columns and PRESENT_TS_COL in df.columns:
        washout_years = float(getattr(outcome_cfg, "washout_years", 0) or 0)
        presenting_min_year = getattr(outcome_cfg, "presenting_min_year", None)
        anchor_ts = pd.to_datetime(df["anchor_ts"])
        pres_ts = pd.to_datetime(df[PRESENT_TS_COL])
        min_eligible = anchor_ts + pd.to_timedelta(washout_years * 365.25, unit="D")
        if presenting_min_year is not None:
            min_eligible = min_eligible.clip(lower=pd.Timestamp(presenting_min_year))
        timely = (pres_ts - min_eligible).dt.days <= 400
        n_before = len(df)
        if timely.any():
            df = df[timely].copy()
        print(f"[find_candidates] {len(df)}/{n_before} retained")

    if df.empty:
        return df

    test_code = outcome_cfg.markers[0]
    df = _add_baseline_sort_columns(df, sp_df, test_code)
    df = df.sort_values(["_baseline_popri_ok", "_delta_abs_sort", "_z_sort"], ascending=[False, False, False])

    if "first_in_window_event" in df.columns and PRESENT_TS_COL in df.columns:
        t_event = pd.to_datetime(df["first_in_window_event"], errors="coerce")
        t_pres = pd.to_datetime(df[PRESENT_TS_COL], errors="coerce")
        df["days_to_event"] = (t_event - t_pres).dt.days
    else:
        df["days_to_event"] = np.nan

    return df.reset_index(drop=True)


def _select_fallback_patient(presenting_df_oo_km: pd.DataFrame, sp_df_test_code: pd.DataFrame):
    """Naive fallback: first event patient, no PerRI/baseline requirement."""
    event_patients = presenting_df_oo_km[presenting_df_oo_km["any_in_window"] == 1]
    if event_patients.empty:
        return None
    example_patient = event_patients.iloc[0]
    patient_id = example_patient[ID_COL]
    patient_sp_data = sp_df_test_code[sp_df_test_code[ID_COL] == patient_id]
    if patient_sp_data.empty:
        return None
    presenting_row = presenting_df_oo_km[presenting_df_oo_km[ID_COL] == patient_id]
    return {"patient_sp_data": patient_sp_data, "presenting_row": presenting_row, "event_row": presenting_row}


def _select_example_patient(
    outcome_name: str,
    presenting_df_oo_km: pd.DataFrame,
    sp_df_test_code: pd.DataFrame,
    outcome_cfg,
):
    """Select the example patient for one outcome's trajectory panel.

    Fallback order:
    1. Saved case from CASES_JSON_PATH (manually selected via run_fig4_dx_cases_case.py --accept).
    2. Best group-B candidate (pop_in & per_out & event), ranked by baseline plausibility then |delta|.
    3. _select_fallback_patient (first event patient, no PerRI requirement).
    """
    cases = _load_cases()
    patient_id = cases.get(outcome_name)
    if patient_id:
        patient_sp_data = sp_df_test_code[sp_df_test_code[ID_COL] == patient_id]
        presenting_row = presenting_df_oo_km[presenting_df_oo_km[ID_COL] == patient_id]
        if not patient_sp_data.empty and not presenting_row.empty:
            print(f"[{outcome_name}] using saved case {patient_id}")
            return {"patient_sp_data": patient_sp_data, "presenting_row": presenting_row, "event_row": presenting_row}
        print(f"[{outcome_name}] saved case {patient_id} not found in data -- falling back")

    try:
        candidates = find_candidates(presenting_df_oo_km, sp_df_test_code, outcome_cfg, group="b", events_only=True)
    except Exception as exc:
        print(f"[{outcome_name}] could not rank group-B candidates ({exc}) -- falling back")
        candidates = pd.DataFrame()

    if not candidates.empty:
        best = candidates.iloc[0]
        patient_id = best[ID_COL]
        patient_sp_data = sp_df_test_code[sp_df_test_code[ID_COL] == patient_id]
        presenting_row = presenting_df_oo_km[presenting_df_oo_km[ID_COL] == patient_id]
        if not patient_sp_data.empty:
            print(f"[{outcome_name}] auto-selected group B case {patient_id}")
            return {"patient_sp_data": patient_sp_data, "presenting_row": presenting_row, "event_row": presenting_row}

    return _select_fallback_patient(presenting_df_oo_km, sp_df_test_code)


def _prepare_forest_model_data(presenting_df_oo_km: pd.DataFrame):
    """Mirrors scripts/figures/fig4_prog.py:_prepare_fig4_forest_model_data."""
    df_model = presenting_df_oo_km.copy()
    outcome_col = "any_in_window"
    covariates = [c for c in ["age_at_presenting", SEX_COL] if c in df_model.columns]
    df_model = df_model.rename(columns={PRESENT_VAL_COL: "presenting"})
    df_model[CV_COL] = df_model[SIGMA] / df_model[MU]
    exposures = [c for c in ["presenting", DELTA_COL, MU, CV_COL] if c in df_model.columns]
    if SEX_COL in df_model.columns:
        df_model[SEX_COL] = df_model[SEX_COL].map({"F": 0, "M": 1})
    return df_model, outcome_col, exposures, covariates


def _fit_forest_model_results(presenting_df_oo_km: pd.DataFrame) -> dict | None:
    df_model, outcome_col, exposures, covariates = _prepare_forest_model_data(presenting_df_oo_km)
    res_df = fit_logit_and_report(outcome=outcome_col, df=df_model, exposures=exposures, covariates=covariates, multivariable_exposures=[], standardize=True)
    if res_df is None or res_df.empty or "model_type" not in res_df.columns:
        print("[fig4_dx_cases] No model results (insufficient events or variation). Skipping forest plot.")
        return None
    model_results = {}
    for pretty_label, sub in res_df.groupby("model_type"):
        model_results[pretty_label] = {"or_table": sub}
    return model_results


def _forest_ors_df(results: dict[str, dict]) -> pd.DataFrame:
    """Flattens every outcome's forest-plot model_results (odds_ratio/ci_lower/ci_upper per
    feature, from fit_logit_and_report) into one tidy table. fig4forest_on_ax plots this same
    data straight from the in-memory model_results dict -- without this, the OR/CI bounds
    behind the combined mosaic's forest column only ever existed for the duration of one run,
    with no way to recover them short of re-fitting."""
    rows = []
    for outcome_name, spec in results.items():
        model_results = spec.get("model_results")
        if not model_results:
            continue
        for sub in model_results.values():
            # sub["or_table"] already has a model_type column (from fit_logit_and_report's
            # uni/multi split, which is also model_results' dict key) -- only outcome is new.
            or_table = sub["or_table"].copy()
            or_table.insert(0, "outcome", outcome_name)
            rows.append(or_table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _km_data_df(km_inputs: KMExclusiveInputs) -> pd.DataFrame:
    """Return the patient-level durations/events and exclusive group used by the KM panel."""
    frames = []
    for group, mask in km_inputs.masks.items():
        sub = km_inputs.km_all.loc[mask].copy()
        if not sub.empty:
            sub.insert(0, "group", group)
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["group", "duration_days", "event_observed", "presenting_ts"])


def _heatmap_tidy_df(spec: dict) -> pd.DataFrame:
    """Return the four count/rate cells rendered in one case's reclassification heatmap."""
    lower, upper = spec["lower"], spec["upper"]
    if lower and not upper:
        abnormal_label, pop_col, per_col, direction = "Low", OUT_POPRI_LOWER_COL, OUT_PERRI_P95_LOWER_COL, "decrease"
    elif upper and not lower:
        abnormal_label, pop_col, per_col, direction = "High", OUT_POPRI_UPPER_COL, OUT_PERRI_P95_UPPER_COL, "increase"
    else:
        abnormal_label, pop_col, per_col, direction = "Abnormal", OUT_POPRI_COL, OUT_PERRI_P95_COL, "two_tailed"

    tables = build_added_value_tables(
        spec["cohort"], pop_col=pop_col, per_col=per_col, outcome_col="any_in_window",
        id_col=ID_COL, ts_col="presenting_ts", verbose=False,
    )
    grids = build_reclassification_2x2_grids(tables["reclassification"].copy(), abnormal_label=abnormal_label, direction=direction)
    rate_df, count_df = grids["rate_df"], grids["count_df"]
    return pd.DataFrame(
        [
            {"perri": per_label, "popri": pop_label, "n": int(count_df.loc[per_label, pop_label]), "event_rate_pct": float(rate_df.loc[per_label, pop_label])}
            for per_label in rate_df.index
            for pop_label in rate_df.columns
        ]
    )


def compute_one_outcome(outcome_name: str, *, input_dir: Path, dx_incident, demographics_df, output_dir: Path, force: bool = False) -> dict:
    """Computes (does not plot) everything one row of the combined mosaic needs.

    Loads this outcome's own markers only, and only when actually needed: sp_df's canonical
    cache (shared with fig3_hazard/run_setpoints_by_marker/fig3_dx) is checked before touching
    tests.csv at all, and the cohort cache below skips tests_df entirely on a hit. sp_df
    itself is still always needed (not just for cohort-building) -- plot_combined's heatmap
    panel reads it directly, regardless of whether the cohort was cached.
    """
    outcome_cfg = OUTCOME_REGISTRY[outcome_name]
    test_code = outcome_cfg.markers[0]
    lower, upper = outcome_cfg.flag_below, outcome_cfg.flag_above
    _, out_popri_col = _direction_columns(lower, upper)

    outcome_markers = _outcome_markers(outcome_cfg)
    tests_df_holder = {}

    def _get_tests_df() -> pd.DataFrame:
        if "value" not in tests_df_holder:
            tests_df_holder["value"] = load_tests_marker_subset(input_dir, test_codes=outcome_markers)
        return tests_df_holder["value"]

    # force (this outcome's own cohort cache, below) never cascades to the shared setpoint
    # dependency -- that's run_setpoints_by_marker's job, not this script's.
    if is_fitted_canonical(test_code):
        sp_df = compute_sp_df(None, test_code=test_code, canonical=True)
    else:
        sp_df = compute_sp_df(_get_tests_df(), test_code=test_code, force=False, canonical=True)
    if sp_df.empty:
        raise ValueError(f"[{outcome_name}] No setpoints computable for test_code={test_code}.")

    def _compute():
        cohort = load_or_create_analysis_ready_cohort(sp_df=sp_df, tests_df=_get_tests_df(), dx_incident=dx_incident, outcome_cfg=outcome_cfg)
        cohort[PERRI_Z_SCORE_COL] = cohort[PERRI_Z_SCORE_COL].clip(-6, 6)
        cohort = attach_ref_intervals(cohort)
        cohort["popri_continuous"] = compute_popri_continuous(cohort, pop_lo="ref_low", pop_hi="ref_high", lower=lower, upper=upper)
        cohort = cohort.merge(demographics_df[[ID_COL, "birth_date"]].drop_duplicates(ID_COL), on=ID_COL, how="left")
        cohort["age_at_presenting"] = (pd.to_datetime(cohort["presenting_ts"]) - pd.to_datetime(cohort["birth_date"])).dt.days / 365.25
        return cohort

    cohort_path = output_dir / f"fig4_dx_cases_{outcome_name}_cohort.csv"
    presenting_df_oo_km = cache_or_compute(cohort_path, _compute, force=force, file_format="csv")

    analysis_window_days = int(outcome_cfg.analysis_window_years * 365.25)
    km_inputs = KMExclusiveInputs.from_dataframe(presenting_df_oo_km, window_days=analysis_window_days, out_popri_col=out_popri_col, lower=lower, upper=upper)
    try:
        model_results = _fit_forest_model_results(presenting_df_oo_km)
    except Exception as exc:
        print(f"[{outcome_name}] forest model fit failed, showing 'no results' in that panel instead: {exc}")
        model_results = None
    example_bundle = _select_example_patient(outcome_name, presenting_df_oo_km, sp_df, outcome_cfg)

    group_summary = []
    for group, mask in km_inputs.masks.items():
        sub = presenting_df_oo_km.loc[mask]
        n = int(len(sub))
        events = int(sub["any_in_window"].sum()) if n else 0
        group_summary.append({"group": group, "n": n, "events": events, "event_rate_pct": 100.0 * events / n if n else None})

    return {
        "outcome_name": outcome_name,
        "outcome_cfg": outcome_cfg,
        "test_code": test_code,
        "lower": lower,
        "upper": upper,
        "cohort": presenting_df_oo_km,
        "sp_df": sp_df,
        "km_inputs": km_inputs,
        "model_results": model_results,
        "example_bundle": example_bundle,
        "cohort_path": cohort_path,
        "manifest": {
            "outcome": outcome_name,
            "test_code": test_code,
            "n_cohort": int(len(presenting_df_oo_km)),
            "n_events": int(presenting_df_oo_km["any_in_window"].sum()),
            "group_summary": group_summary,
        },
    }


def plot_combined(results: dict[str, dict], mosaic_path: Path) -> None:
    """One row per outcome, four columns: trajectory | forest | KM | reclassification heatmap.

    Mirrors scripts/figures/fig4_prog.py:generate_combined.
    """
    specs = list(results.values())
    fig, axes = plt.subplots(
        len(specs),
        4,
        figsize=(A4_WIDTH * 1.1, FIG4_ROW1_HEIGHT * len(specs)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.2, 0.6, 0.80, 0.80], "wspace": 0.6, "hspace": 0.8},
    )

    for row_idx, spec in enumerate(specs):
        legend = row_idx == 0
        example_bundle = spec["example_bundle"]

        ax0 = axes[row_idx, 0]
        if example_bundle is None:
            ax0.text(0.5, 0.5, "No event patients found", ha="center", va="center", transform=ax0.transAxes)
            ax0.set_xlabel("Years")
        else:
            plot_single_patient_history_on_ax(
                ax=ax0,
                patient_sp_data=example_bundle["patient_sp_data"],
                presenting_row=example_bundle["presenting_row"],
                event_row=example_bundle["event_row"],
                test_code=spec["test_code"],
                lower=spec["lower"],
                upper=spec["upper"],
                add_legend=legend,
            )

        ax1 = axes[row_idx, 1]
        if spec["model_results"] is None:
            ax1.text(0.5, 0.5, "No model results", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_axis_off()
        else:
            fig4forest_on_ax(ax=ax1, model_results_dict=spec["model_results"])

        ax2 = axes[row_idx, 2]
        fig4km_exclusive_on_ax(ax=ax2, inputs=spec["km_inputs"], xlabel="Years after presenting test", add_legend=legend)

        ax3 = axes[row_idx, 3]
        inputs = ProgressionPanelInputs(presenting_df=spec["cohort"], outcome_cfg=spec["outcome_cfg"], sp_df=spec["sp_df"])
        fig4heatmap_2x2_on_ax(ax=ax3, inputs=inputs, annotate_n=True)

        axes[row_idx, 0].text(-0.2, 1.2, spec["outcome_name"], ha="left", va="center", fontsize=8, fontweight="bold", transform=axes[row_idx, 0].transAxes)

    save_fig_as_svg(fig, "fig4_dx_cases_combined", mosaic_path)
    plt.close(fig)


def run(*, input_dir: Path, output_dir: Path, dx_incident_path: Path = None, outcomes: list[str] = None, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = outcomes or list(OUTCOME_REGISTRY.keys())
    if dx_incident_path is None:
        dx_incident_path = Path(__file__).resolve().parent.parent / "outputs" / "dx_incident" / "dx_incident.csv"

    dx_incident = load_dx_incident(dx_incident_path)
    demographics_df = load_demographics_csv(input_dir / DEMOGRAPHICS_FILE)

    results = {}
    manifests = {}
    for outcome_name in outcomes:
        print(f"\n=== fig4 progression: {outcome_name} ===")
        try:
            results[outcome_name] = compute_one_outcome(outcome_name, input_dir=input_dir, dx_incident=dx_incident, demographics_df=demographics_df, output_dir=output_dir, force=force)
            manifests[outcome_name] = results[outcome_name]["manifest"]
        except Exception as e:
            print(f"[{outcome_name}] SKIPPED: {e}")
            manifests[outcome_name] = {"outcome": outcome_name, "error": str(e)}

    mosaic_path = output_dir / "fig4_dx_cases_combined.svg"
    if results:
        plot_combined(results, mosaic_path)

    forest_ors_df = _forest_ors_df(results)
    forest_ors_path = output_dir / "fig4_dx_cases_forest_ors.csv"
    forest_ors_df.to_csv(forest_ors_path, index=False)

    case_outputs = []
    for outcome_name, spec in results.items():
        artifacts = {
            f"fig4_dx_cases_{outcome_name}_ors.csv": _forest_ors_df({outcome_name: spec}),
            f"fig4_dx_cases_{outcome_name}_km_data.csv": _km_data_df(spec["km_inputs"]),
            f"fig4_dx_cases_{outcome_name}_heatmap_t1.csv": _heatmap_tidy_df(spec),
        }
        for filename, artifact_df in artifacts.items():
            artifact_df.to_csv(output_dir / filename, index=False)
            case_outputs.append(filename)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "results": manifests,
        "outputs": [mosaic_path.name, forest_ors_path.name] + [f"fig4_dx_cases_{name}_cohort.csv" for name in results] + case_outputs + ["manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")

    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run fig4 dx cases (aki, leukemia, hypothyroidism).")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "fig4_dx_cases")
    parser.add_argument("--dx-incident-path", type=Path, default=None, help="Path to dx_incident.csv from `scripts.run_dx_incident`. Default: outputs/dx_incident/dx_incident.csv")
    parser.add_argument("--outcome", action="append", dest="outcomes", choices=list(OUTCOME_REGISTRY.keys()), help="Restrict to one outcome (repeatable). Default: all.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    with tagged_stdout("fig4_dx_cases"):
        run(input_dir=args.input_dir, output_dir=args.output_dir, dx_incident_path=args.dx_incident_path, outcomes=args.outcomes, force=args.force)
    # print(json.dumps(manifest, indent=2, sort_keys=True, default=str))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
