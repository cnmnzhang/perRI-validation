"""Fig4 progression cohort-building + outcome-flagging core.

Scoped to what the aki/leukemia/hypothyroidism outcomes in
perri_validation/utils/progression/config.py need:
build_eligible_cohort_from_setpoints, flag_outcomes_from_config,
outcome_processing, and the orchestration wrapper load_or_create_analysis_ready_cohort.
Every marker lookup takes an explicit ``tests_df`` argument -- the in-memory
generic Tests table (validation/ never reads raw markers from disk). None of
the three in-scope outcomes use ``age_cutoff``, so that branch is
intentionally omitted (see below). Caching (CSV-on-disk memoization keyed by
a cohort-params hash) goes through ``cache_or_compute``.
"""

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from perri_validation.utils.clinical.inputs import OutcomeConfig
from perri_validation.utils.clinical.run_clinical import add_oo
from perri_validation.constants.runtime import ID_COL, MEASUREMENT_COL, MU, PRESENT_TS_COL, PRESENT_VAL_COL, SIGMA, TEST_CODE_COL, TS_COL


def build_eligible_cohort_from_setpoints(
    sp_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    outcome_cfg: OutcomeConfig,
    mu_col: str = MU,
    sigma_col: str = SIGMA,
    min_points: int = 5,
) -> Tuple[pd.DataFrame, dict]:
    """Builds the core cohort from sp_df using an OutcomeConfig.

    Identifies the anchor row (Nth sp_df row per patient per cohort strategy) and
    records the setpoint mu/sigma from that row. Sex and birth_date are captured
    from the anchor row when present in sp_df. Then loads raw (non-isolated) marker
    data from ``tests_df`` and finds the first presenting measurement strictly after
    anchor_ts (with optional washout).
    """
    cohort_cfg = outcome_cfg.cohort
    min_points = int(cohort_cfg.min_points if cohort_cfg.min_points else min_points)
    first_n = cohort_cfg.first_n  # None = anchor at last eligible measurement
    age_cutoff = cohort_cfg.age_cutoff
    year_cutoff = cohort_cfg.year_cutoff
    test_code = outcome_cfg.markers[0] if outcome_cfg.markers else None
    washout_years = outcome_cfg.washout_years

    if age_cutoff is not None:
        raise NotImplementedError("age_cutoff cohort filtering requires attach_demographics, not implemented here (unused by aki/leukemia/hypothyroidism).")

    active_filters = [k for k, v in [("year_cutoff", year_cutoff), ("age_cutoff", age_cutoff)] if v is not None]
    anchor_desc = f"first_n={first_n}" if first_n is not None else "last"
    print(f"Building cohort for test_code={test_code} | filters={active_filters or 'none'} | anchor={anchor_desc}")

    df = sp_df.copy()
    if test_code and TEST_CODE_COL in df.columns:
        df = df[df[TEST_CODE_COL] == test_code].copy()

    required = {ID_COL, TS_COL, mu_col, sigma_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sp_df is missing required columns: {missing}")

    n_before = len(df)
    df = df.drop_duplicates(subset=[ID_COL, TS_COL])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[cohort builder] Dropped {n_dropped:,} duplicate rows for test_code={test_code} (kept one row per patient-timestamp).")

    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    df = df.sort_values([ID_COL, TS_COL])

    rows = []
    stats = defaultdict(int)

    for pat_id, g in df.groupby(ID_COL, sort=False):
        g = g.sort_values(TS_COL).reset_index(drop=True)

        if len(g) < min_points:
            stats["too_few_overall"] += 1
            continue

        eligible = g.copy()
        eligibility_ts = None

        if year_cutoff is not None:
            cutoff_ts = pd.Timestamp(f"{int(year_cutoff)}-01-01")
            eligible = eligible[eligible[TS_COL] < cutoff_ts]
            eligibility_ts = cutoff_ts

        if len(eligible) < min_points:
            stats["too_few_in_eligible_pool"] += 1
            continue

        if first_n is not None:
            if len(eligible) < first_n:
                stats["too_few_for_n"] += 1
                continue
            anchor_idx = eligible.index[first_n - 1]
        else:
            anchor_idx = eligible.index[-1]

        anchor_row = g.loc[anchor_idx]
        anchor_ts = anchor_row[TS_COL]

        if eligibility_ts is None:
            eligibility_ts = anchor_ts

        mu = anchor_row[mu_col]
        sigma = anchor_row[sigma_col]
        if pd.isna(mu) or pd.isna(sigma):
            stats["invalid_setpoint"] += 1
            continue

        row = {
            ID_COL: pat_id,
            TEST_CODE_COL: test_code if test_code else (anchor_row[TEST_CODE_COL] if TEST_CODE_COL in g.columns else None),
            "anchor_ts": anchor_ts,
            "eligibility_ts": eligibility_ts,
            MU: float(mu),
            SIGMA: float(sigma),
            "ci_lower": float(mu - 1.96 * sigma),
            "ci_upper": float(mu + 1.96 * sigma),
        }
        if "sex" in anchor_row.index:
            row["sex"] = anchor_row["sex"]
        if "birth_date" in anchor_row.index:
            row["birth_date"] = anchor_row["birth_date"]

        rows.append(row)
        stats["included"] += 1

    if not rows:
        print("Warning: No rows were included in the cohort.")
        return pd.DataFrame(), stats

    anchor_df = pd.DataFrame(rows)
    anchor_df = anchor_df.sort_values([c for c in [TEST_CODE_COL, ID_COL, "anchor_ts"] if c in anchor_df.columns]).reset_index(drop=True)

    print("* Cohort summary (before raw data join):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # --- Join raw marker data to find the presenting measurement after washout ---
    raw_df = tests_df[tests_df[TEST_CODE_COL] == test_code].copy()
    raw_df[ID_COL] = raw_df[ID_COL].astype(str)
    raw_df[TS_COL] = pd.to_datetime(raw_df[TS_COL])

    raw_with_anchor = raw_df.merge(anchor_df[[ID_COL, "anchor_ts"]], on=ID_COL, how="inner")
    if washout_years > 0:
        washout_cutoff = raw_with_anchor["anchor_ts"] + pd.DateOffset(years=washout_years)
    else:
        washout_cutoff = raw_with_anchor["anchor_ts"]

    presenting_min_year = outcome_cfg.presenting_min_year
    if presenting_min_year is not None:
        min_ts = pd.Timestamp(presenting_min_year)
        washout_cutoff = washout_cutoff.clip(lower=min_ts)
        print(f"[cohort builder] Presenting floor applied: no presenting value before {min_ts.date()}.")

    post_washout = raw_with_anchor[raw_with_anchor[TS_COL] > washout_cutoff]

    print("[cohort builder] selecting first test after washout cutoff (anchor_ts + washout_years) ...")
    presenting = (
        post_washout.sort_values([ID_COL, TS_COL])
        .groupby(ID_COL)
        .first()
        .reset_index()[[ID_COL, TS_COL, MEASUREMENT_COL]]
        .rename(columns={TS_COL: PRESENT_TS_COL, MEASUREMENT_COL: PRESENT_VAL_COL})
    )

    n_before_join = len(anchor_df)
    anchor_df = anchor_df.merge(presenting, on=ID_COL, how="inner")
    n_dropped_no_present = n_before_join - len(anchor_df)
    if n_dropped_no_present:
        print(f"[cohort builder] Dropped {n_dropped_no_present:,} patients with no presenting measurement after washout (washout_years={washout_years}).")

    print(f"[cohort builder] Final cohort size after raw data join: {len(anchor_df):,}")

    return anchor_df, stats


def _merge_and_flag_in_window(
    presenting_df: pd.DataFrame,
    first_dates_df: pd.DataFrame,
    name: str,
    window_years: float,
    *,
    present_col: str = PRESENT_TS_COL,
    grace_days: int = 14,
) -> pd.DataFrame:
    """Merge earliest event dates and flag events within a window."""
    df = presenting_df.copy()
    date_col = f"{name}_date"
    if date_col not in first_dates_df.columns:
        if first_dates_df.empty:
            df[date_col] = pd.NaT
            df[f"{name}_in_window"] = False
            df[f"{name}_tte_days"] = np.nan
            return df
        else:
            raise ValueError(f"Expected column '{date_col}' in first_dates_df")

    df = df.merge(first_dates_df[[ID_COL, date_col]], on=ID_COL, how="left")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    any_dt = df[date_col]
    win_lo = df[present_col] + pd.Timedelta(days=grace_days)
    win_hi = df[present_col] + pd.Timedelta(days=int(window_years * 365.25))

    df[f"{name}_in_window"] = any_dt.notna() & (any_dt >= win_lo) & (any_dt <= win_hi)
    df[f"{name}_tte_days"] = (any_dt - df[present_col]).dt.days.astype("float")

    return df


def flag_outcomes_from_config(
    presenting_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    outcome_cfg: OutcomeConfig,
    dx_incident: pd.DataFrame,
    analysis_window_years: float,
    value_col: str = MEASUREMENT_COL,
) -> pd.DataFrame:
    """Loops through outcome_cfg.outcomes and flags based on type ("diagnosis" or "lab_threshold")."""
    df = presenting_df.copy()
    df[PRESENT_TS_COL] = pd.to_datetime(df[PRESENT_TS_COL])

    grace_days = outcome_cfg.grace_days

    if dx_incident is not None:
        dx = dx_incident.copy()
        dx["earliest_contact_date"] = pd.to_datetime(dx["earliest_contact_date"])
    else:
        dx = pd.DataFrame(columns=[ID_COL, "diagnosis_name", "earliest_contact_date"])

    for out in outcome_cfg.outcomes:
        name = out.name
        outcome_type = out.type

        first_dates = pd.DataFrame(columns=[ID_COL, f"{name}_date"])

        try:
            if outcome_type == "diagnosis":
                dx_name = out.diagnosis_name
                first_dates = (
                    dx.loc[dx["diagnosis_name"] == dx_name, [ID_COL, "earliest_contact_date"]]
                    .dropna(subset=["earliest_contact_date"])
                    .sort_values([ID_COL, "earliest_contact_date"])
                    .groupby(ID_COL, as_index=False)
                    .first()
                    .rename(columns={"earliest_contact_date": f"{name}_date"})
                )

            elif outcome_type == "lab_threshold":
                marker = out.marker
                lower, upper = out.thresholds

                lab = tests_df[tests_df[TEST_CODE_COL] == marker].copy()
                lab[TS_COL] = pd.to_datetime(lab[TS_COL])
                lab[value_col] = pd.to_numeric(lab[value_col], errors="coerce")

                qualifies = lab[value_col].notna()
                if lower is not None:
                    qualifies &= lab[value_col] >= lower
                if upper is not None:
                    qualifies &= lab[value_col] < upper

                first_dates = lab.loc[qualifies, [ID_COL, TS_COL]].sort_values([ID_COL, TS_COL]).groupby(ID_COL, as_index=False).first().rename(columns={TS_COL: f"{name}_date"})

            else:
                raise NotImplementedError(f"Outcome type '{outcome_type}' not implemented (only 'diagnosis'/'lab_threshold' used by aki/leukemia/hypothyroidism).")

            df = _merge_and_flag_in_window(df, first_dates, name, present_col=PRESENT_TS_COL, window_years=analysis_window_years, grace_days=grace_days)

        except Exception as e:
            print(f"Warning: Failed to flag outcome '{name}'. Error: {e}")
            if f"{name}_date" not in df.columns:
                df[f"{name}_date"] = pd.NaT
                df[f"{name}_in_window"] = False
                df[f"{name}_tte_days"] = np.nan

    return df


def outcome_processing(presenting_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes all individual outcome flags into master any_in_window/first_in_window_event flags."""
    df = presenting_df.copy()

    for col in ["any_progression_event", "any_progression_tte", "any_in_window", "first_in_window_event"]:
        if col in df.columns:
            df = df.drop(columns=col)

    exclude_dates = ["birth_date", PRESENT_TS_COL, "eligibility_ts"]
    all_date_cols = sorted([c for c in df.columns if c.endswith("_date") and c not in exclude_dates])

    for c in all_date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
        df.loc[df[c] == pd.Timestamp(0), c] = pd.NaT

    if not all_date_cols:
        print("Warning: No outcome '_date' columns found.")
        df["any_progression_event"] = pd.NaT
        df["any_progression_tte"] = np.nan
        df["any_in_window"] = False
        df["first_in_window_event"] = pd.NaT
        return df

    df["any_progression_event"] = df[all_date_cols].min(axis=1, skipna=True)
    df["any_progression_tte"] = (df["any_progression_event"] - df[PRESENT_TS_COL]).dt.days

    def root(name):
        return name[: -len("_date")]

    roots = {root(c) for c in all_date_cols}
    existing_inwin_cols = [f"{r}_in_window" for r in roots if f"{r}_in_window" in df.columns]

    if existing_inwin_cols:
        df["any_in_window"] = df[existing_inwin_cols].any(axis=1)
    else:
        df["any_in_window"] = False

    if existing_inwin_cols:
        masked_dates = []
        for r in roots:
            dcol = f"{r}_date"
            icol = f"{r}_in_window"
            if dcol in df.columns and icol in df.columns:
                masked_dates.append(df[dcol].where(df[icol]))

        if masked_dates:
            tmp = pd.concat(masked_dates, axis=1)
            df["first_in_window_event"] = tmp.min(axis=1, skipna=True)
        else:
            df["first_in_window_event"] = pd.NaT
    else:
        df["first_in_window_event"] = pd.NaT

    return df


def _build_analysis_ready_cohort(
    presenting_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    dx_incident: pd.DataFrame,
    outcome_cfg: OutcomeConfig,
    analysis_window_years: float,
) -> pd.DataFrame:
    """Run the full data prep pipeline: OO flags, prevalent exclusion, outcome flagging, summarization.

    Sex and birth_date are expected to already be present in presenting_df (sourced from
    sp_df rows in build_eligible_cohort_from_setpoints).
    """
    print("Running full data preparation pipeline...")

    presenting_df_oo = add_oo(presenting_df, setpoint_col=MU, sigma_col=SIGMA, result_col=PRESENT_VAL_COL, p=0.95)

    for outcome_def in outcome_cfg.outcomes:
        if not outcome_def.exclude_prevalent:
            continue
        if outcome_def.type == "diagnosis":
            dx_name = outcome_def.diagnosis_name
            prev_dates = dx_incident[dx_incident["diagnosis_name"] == dx_name][[ID_COL, "earliest_contact_date"]]
            prev_pats = prev_dates.merge(presenting_df_oo[[ID_COL, PRESENT_TS_COL]], on=ID_COL, how="inner")
            prevalent = prev_pats[prev_pats["earliest_contact_date"] <= prev_pats[PRESENT_TS_COL]][ID_COL].unique()
        elif outcome_def.type == "lab_threshold":
            marker = outcome_def.marker
            thresholds = outcome_def.thresholds
            lab_df = tests_df[tests_df[TEST_CODE_COL] == marker].copy()
            lab_df[ID_COL] = lab_df[ID_COL].astype(str)
            lab_df[TS_COL] = pd.to_datetime(lab_df[TS_COL])
            lo, hi = thresholds
            mask = pd.Series(True, index=lab_df.index)
            if lo is not None:
                mask &= lab_df[MEASUREMENT_COL] >= lo
            if hi is not None:
                mask &= lab_df[MEASUREMENT_COL] < hi
            lab_prev = lab_df[mask].merge(presenting_df_oo[[ID_COL, PRESENT_TS_COL]], on=ID_COL, how="inner")
            prevalent = lab_prev[lab_prev[TS_COL] <= lab_prev[PRESENT_TS_COL]][ID_COL].unique()
        else:
            continue
        n_before = len(presenting_df_oo)
        presenting_df_oo = presenting_df_oo[~presenting_df_oo[ID_COL].isin(prevalent)].copy()
        print(f"[prevalent exclusion] {outcome_def.name}: excluded {n_before - len(presenting_df_oo):,} prevalent cases")

    presenting_df_flagged = flag_outcomes_from_config(
        presenting_df=presenting_df_oo,
        tests_df=tests_df,
        outcome_cfg=outcome_cfg,
        dx_incident=dx_incident,
        analysis_window_years=analysis_window_years,
    )

    return outcome_processing(presenting_df_flagged)


def load_or_create_analysis_ready_cohort(
    sp_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    dx_incident: pd.DataFrame,
    outcome_cfg: OutcomeConfig,
    window_years: Optional[float] = None,
) -> pd.DataFrame:
    """Orchestrates build_eligible_cohort_from_setpoints -> _build_analysis_ready_cohort."""
    final_window_years = window_years if window_years is not None else outcome_cfg.analysis_window_years

    presenting_df, stats = build_eligible_cohort_from_setpoints(sp_df=sp_df, tests_df=tests_df, outcome_cfg=outcome_cfg)

    if presenting_df.empty:
        print("\n--- COHORT BUILDING FAILED ---")
        for k, v in (stats or {}).items():
            print(f"  {k}: {v}")
        raise ValueError("Cohort building resulted in an empty dataframe.")

    return _build_analysis_ready_cohort(
        presenting_df=presenting_df,
        tests_df=tests_df,
        dx_incident=dx_incident,
        outcome_cfg=outcome_cfg,
        analysis_window_years=final_window_years,
    )


def compute_popri_continuous(
    df: pd.DataFrame,
    measurement_col: str = PRESENT_VAL_COL,
    pop_lo: str = "pop_lo",
    pop_hi: str = "pop_hi",
    lower: bool = True,
    upper: bool = True,
):
    """Continuous PopRI deviation score (distance from the relevant boundary/center)."""
    values = df[measurement_col].to_numpy(dtype=float)
    lo = df[pop_lo].to_numpy(dtype=float)
    hi = df[pop_hi].to_numpy(dtype=float)

    if upper and not lower:
        return hi - values
    if lower and not upper:
        return values - lo
    center = (lo + hi) / 2.0
    half_width = np.clip((hi - lo) / 2.0, 1e-9, None)
    return (values - center) / half_width
