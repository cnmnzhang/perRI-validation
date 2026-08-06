"""Cohort-building helpers for iron infusion, outcome flagging, and fig3's KM panel.

add_oo, IV-iron course/cohort builders, reclassification tables,
get_one_setpoint.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from lifelines import KaplanMeierFitter

from perri_validation.utils.clinical.get import popRI
from perri_validation.constants.marker_config import LOG_TRANSFORM_MARKERS
from perri_validation.constants.runtime import (
    DELTA_COL,
    ID_COL,
    INDEX_COL,
    MAX_FIT_DATE,
    MEASUREMENT_COL,
    MU,
    OUT_PERRI_LOWER_SUFFIX,
    OUT_PERRI_P_PREFIX,
    OUT_PERRI_UPPER_SUFFIX,
    OUT_POPRI_COL,
    OUT_POPRI_LOWER_COL,
    OUT_POPRI_UPPER_COL,
    PERRI_Z_SCORE_COL,
    POP_HI_COL,
    POP_LO_COL,
    SEX_COL,
    SIGMA,
    TEST_CODE_COL,
    TS_COL,
)

MACHINE_EPSILON = 1e-6
GAP_BETWEEN_COURSES = 60
MIN_MEASUREMENTS = 5


def add_oo(df: pd.DataFrame, setpoint_col: str = MU, sigma_col: str = SIGMA, result_col: str = MEASUREMENT_COL, p: float = 0.95) -> pd.DataFrame:
    """Adds out-of-range flags based on a normal distribution AND the continuous z-score.

    - perRI_z_score: (continuous) (result - setpoint) / sigma
    - out_perri_pXX : 1 if result is outside the personal reference interval at XX% confidence
    - out_perri_pXX_lower / _upper : directional flags
    - out_popri : 1 if result is outside the population reference interval
    """
    df = df.copy()

    df[PERRI_Z_SCORE_COL] = (df[result_col] - df[setpoint_col]) / df[sigma_col]
    df[PERRI_Z_SCORE_COL] = df[PERRI_Z_SCORE_COL].replace([np.inf, -np.inf], np.nan)

    z_threshold = norm.ppf((1 + p) / 2.0)  # e.g., p=0.95 -> z~1.96
    key = f"{OUT_PERRI_P_PREFIX}{int(p * 100)}"

    df[key] = (df[PERRI_Z_SCORE_COL].abs() > z_threshold).astype(int)
    df[key + OUT_PERRI_LOWER_SUFFIX] = (df[PERRI_Z_SCORE_COL] < -z_threshold).astype(int)
    df[key + OUT_PERRI_UPPER_SUFFIX] = (df[PERRI_Z_SCORE_COL] > z_threshold).astype(int)

    df[[POP_LO_COL, POP_HI_COL]] = df.apply(lambda r: pd.Series(popRI(r[SEX_COL], test_code=r[TEST_CODE_COL])), axis=1)
    df[OUT_POPRI_COL] = (df[result_col] < df[POP_LO_COL]) | (df[result_col] > df[POP_HI_COL])
    df[OUT_POPRI_LOWER_COL] = (df[result_col] < df[POP_LO_COL]).astype(int)
    df[OUT_POPRI_UPPER_COL] = (df[result_col] > df[POP_HI_COL]).astype(int)
    df[OUT_POPRI_COL] = df[OUT_POPRI_COL].astype(int)

    if setpoint_col in df.columns and result_col in df.columns:
        df[DELTA_COL] = df[result_col] - df[setpoint_col]

    return df


def n_ids(df: pd.DataFrame) -> int:
    return 0 if df.empty else int(df[ID_COL].nunique())


def overlap_ids(df1: pd.DataFrame, df2: pd.DataFrame) -> int:
    if df1.empty or df2.empty:
        return 0
    return int(len(set(df1[ID_COL].unique()) & set(df2[ID_COL].unique())))


def identify_first_course(outcome_df: pd.DataFrame, id_col: str = ID_COL, ts_col: str = TS_COL, gap_between_courses: int = GAP_BETWEEN_COURSES):
    """
    Identify IV iron treatment courses per patient and summarize the FIRST course.

    Doses within `gap_between_courses` days of the previous dose (within a patient)
    are grouped into the same course. A course is defined by its first and last dose timestamps.

    Returns
    -------
    first_courses : pd.DataFrame
        One row per patient for their first observed course, with columns:
            [id_col, course_id, course_start, course_end, doses, n_courses, next_course_start]
    iv_events_first : pd.DataFrame
        All rows from outcome_df that belong to the FIRST course for each patient.
    """
    df = outcome_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values([id_col, ts_col])

    df["prev_ts"] = df.groupby(id_col)[ts_col].shift(1)
    df["gap"] = (df[ts_col] - df["prev_ts"]).dt.days
    df["is_new_course"] = df["gap"].isna() | (df["gap"] > gap_between_courses)
    df["course_id"] = df.groupby(id_col)["is_new_course"].cumsum()

    courses = (
        df.groupby([id_col, "course_id"])
        .agg(
            course_start=(ts_col, "min"),
            course_end=(ts_col, "max"),
            doses=(ts_col, "count"),
        )
        .reset_index()
        .sort_values([id_col, "course_start"])
    )

    course_counts = courses.groupby(id_col)["course_id"].nunique()
    courses["n_courses"] = courses[id_col].map(course_counts)
    courses["next_course_start"] = courses.groupby(id_col)["course_start"].shift(-1)

    first_idx = courses.groupby(id_col)["course_start"].idxmin()
    first_courses = courses.loc[first_idx].copy()
    print("--- Course Identification Summary ---")
    print(f"{len(first_courses)} first courses (per patient).")
    print(f"Patients with >1 course total: {(first_courses['n_courses'] > 1).sum()}")

    iv_events_first = df.merge(first_courses[[id_col, "course_id"]], on=[id_col, "course_id"], how="inner")

    return first_courses, iv_events_first


def _select_pre_post_labs(
    presenting_df: pd.DataFrame,
    courses_df: pd.DataFrame,
    pre_days_max: Optional[int] = None,
    post_days_min: Optional[int] = None,
    post_days_max: Optional[int] = None,
    id_col: str = ID_COL,
    ts_col: str = TS_COL,
    meas_col: str = MEASUREMENT_COL,
):
    """Select pre and/or post labs based on which windows are specified.

    If pre_days_max is None, returns empty DataFrame for pre.
    If post_days_min/max are None, returns empty DataFrame for post.
    """
    lab = presenting_df.copy()
    lab[ts_col] = pd.to_datetime(lab[ts_col], errors="coerce").dt.tz_localize(None)
    lab = lab.dropna(subset=[ts_col]).sort_values([id_col, ts_col])

    courses = courses_df.copy()
    courses["course_start"] = pd.to_datetime(courses["course_start"], errors="coerce")
    courses["course_end"] = pd.to_datetime(courses["course_end"], errors="coerce")
    if "next_course_start" in courses.columns:
        courses["next_course_start"] = pd.to_datetime(courses["next_course_start"], errors="coerce")

    merged = courses.merge(lab, on=id_col, how="left")

    merged["days_from_start"] = (merged[ts_col] - merged["course_start"]).dt.days
    merged["days_from_end"] = (merged[ts_col] - merged["course_end"]).dt.days

    pre = pd.DataFrame()
    if pre_days_max is not None:
        pre_mask = (merged["days_from_start"] <= 0) & (merged["days_from_start"] >= -pre_days_max)
        pre = (
            merged[pre_mask]
            .sort_values([id_col, "days_from_start"], ascending=[True, False])
            .groupby(id_col, as_index=False)
            .first()
            .rename(columns={meas_col: "result_pre", ts_col: "result_pre_ts"})
        )
        print(f"PRE labs: {len(pre)} patients within {pre_days_max}d before course_start")

    post = pd.DataFrame()
    if post_days_min is not None and post_days_max is not None:
        post_mask = (merged["days_from_end"] >= post_days_min) & (merged["days_from_end"] <= post_days_max)

        if "next_course_start" in merged.columns:
            post_mask &= merged["next_course_start"].isna() | (merged[ts_col] < merged["next_course_start"])

        post_subset = merged[post_mask].copy()
        midpoint = (post_days_min + post_days_max) / 2
        post_subset["days_from_midpoint"] = (post_subset["days_from_end"] - midpoint).abs()

        post = (
            post_subset.sort_values([id_col, "days_from_midpoint"], ascending=[True, True])
            .groupby(id_col, as_index=False)
            .first()
            .drop(columns=["days_from_midpoint"])
            .rename(columns={meas_col: "result_post", ts_col: "result_post_ts"})
        )
        print(f"POST labs: {len(post)} patients within {post_days_min}-{post_days_max}d after course_end")

    return pre, post


def get_sp_from_courses(
    sp_df: pd.DataFrame,
    courses_df: pd.DataFrame,
    id_col: str = ID_COL,
    ts_col: str = TS_COL,
    *,
    lookback_min_days: int = 365,
    lookback_max_days: int = 365 * 3,
    min_setpoint_n: int = 3,
):
    """Finds the latest setpoint up to lookback_days prior to the start of the first IV course.

    Returns one setpoint per patient: the latest setpoint within
    [course_start - lookback_max_days, course_start - lookback_min_days].
    """
    print(f"--- Setpoint Selection ({lookback_min_days} - {lookback_max_days} days prior to course_start) ---")

    sp = sp_df.copy()
    sp[ts_col] = pd.to_datetime(sp[ts_col], errors="coerce").dt.tz_localize(None)

    courses = courses_df[[id_col, "course_start", "course_end"]].copy()
    courses["course_start"] = pd.to_datetime(courses["course_start"], errors="coerce").dt.tz_localize(None)
    courses["course_end"] = pd.to_datetime(courses["course_end"], errors="coerce").dt.tz_localize(None)

    merged = sp.merge(courses, on=id_col, how="inner")
    merged["days_before_course"] = (merged["course_start"] - merged[ts_col]).dt.days

    merged = merged[(merged["days_before_course"] >= lookback_min_days) & (merged["days_before_course"] <= lookback_max_days)].copy()
    merged = merged.sort_values([id_col, "days_before_course"], ascending=[True, True]).drop_duplicates(subset=id_col, keep="first")

    n_total_courses_df = courses_df[id_col].nunique()
    n_pats = merged[id_col].nunique()
    merged = merged[merged[INDEX_COL] >= min_setpoint_n]

    print(f"{n_pats} (filtered from {n_total_courses_df} IV patients) with setpoints")
    print(f"{len(merged)} patients with >= {min_setpoint_n} setpoints (after filtering)")

    return merged


def build_iv_cohort(
    pre: pd.DataFrame,
    sp_df: pd.DataFrame,
    post: pd.DataFrame,
    id_col: str = ID_COL,
    ts_col: str = TS_COL,
):
    sp = sp_df[[id_col, ts_col, MU, SIGMA, "course_start", "course_end", SEX_COL]].copy()
    sp[ts_col] = pd.to_datetime(sp[ts_col]).dt.tz_localize(None)
    sp = sp.sort_values(ts_col)
    sp = sp.rename(columns={ts_col: "set_ts"})

    pre = pre.copy()
    pre["result_pre_ts"] = pd.to_datetime(pre["result_pre_ts"]).dt.tz_localize(None)
    pre = pre.sort_values("result_pre_ts")

    pre_with_set = pd.merge(pre, sp, on=id_col, how="left")

    n_total = len(pre_with_set)
    pre_with_set = pre_with_set.dropna(subset=[MU, SIGMA])
    n_kept = len(pre_with_set)

    pre_with_set["setpoint_age_days"] = (pre_with_set["result_pre_ts"] - pre_with_set["set_ts"]).dt.days

    print(f"{n_kept}/{n_total} PRE patients with setpoints: ")

    cohort = pre_with_set.merge(post[[id_col, "result_post", "result_post_ts"]], on=id_col, how="inner")
    return cohort


def attach_metrics(cohort: pd.DataFrame) -> pd.DataFrame:
    """Attach drop, response, and timing metrics to the IV cohort."""
    cohort = cohort.copy()
    cohort = add_oo(cohort, result_col="result_pre")
    cohort["drop"] = -cohort[DELTA_COL].copy()
    cohort["response"] = cohort["result_post"] - cohort["result_pre"]

    print(f"Drop: median={cohort['drop'].median():.2f}, IQR=({cohort['drop'].quantile(0.25):.2f}, {cohort['drop'].quantile(0.75):.2f})")
    print(f"Response: median={cohort['response'].median():.2f}, IQR=({cohort['response'].quantile(0.25):.2f}, {cohort['response'].quantile(0.75):.2f})")

    cohort["days_pre"] = (cohort["course_start"] - cohort["set_ts"]).dt.days
    cohort["days_post"] = (cohort["course_end"] - cohort["set_ts"]).dt.days
    cohort["z_pre"] = (cohort["result_pre"] - cohort[MU]) / cohort[SIGMA].clip(lower=MACHINE_EPSILON)
    return cohort


def build_iron_infusion_trajectories(hb_marker: pd.DataFrame, episodes: pd.DataFrame, sp_df_hb: pd.DataFrame, iv_cohort: pd.DataFrame) -> pd.DataFrame:
    """Build per-measurement trajectories around the IV iron course, for the fig5 trajectory panel.

    hb_marker: full (non-isolated) HB Tests rows. episodes: first_courses (course_start/course_end).
    sp_df_hb: isolated HB setpoints. iv_cohort: the final cohort (for presenting_hb/result_pre and
    the id filter).
    """
    # Select the canonical Tests fields explicitly so old per-marker/bundle caches made
    # before loader normalization cannot leak source-only columns (e.g. epic_pat_id) into
    # the exported figure data.
    hb_marker = hb_marker[[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL, SEX_COL]].copy()
    traj = (
        hb_marker.merge(episodes[[ID_COL, "course_start", "course_end"]], on=ID_COL, how="inner")
        .assign(days_from_infusion=lambda x: (x[TS_COL] - x["course_start"]).dt.days)
    )
    traj = traj[traj[ID_COL].isin(iv_cohort[ID_COL])].copy()

    traj["days_pre"] = (traj[TS_COL] - traj["course_start"]).dt.days
    traj["days_post"] = (traj[TS_COL] - traj["course_end"]).dt.days

    presenting = iv_cohort[[ID_COL, "result_pre"]].drop_duplicates(ID_COL)
    traj = traj.merge(presenting, on=ID_COL, how="left")

    traj_final = pd.merge_asof(
        traj.sort_values(TS_COL),
        sp_df_hb[[ID_COL, TS_COL, MU, SIGMA]].sort_values(TS_COL),
        on=TS_COL,
        by=ID_COL,
        direction="backward",
    )
    return traj_final


# ---------------------------------------------------------------------------
# Reclassification (PopRI x PerRI) 2x2 tables — used by the fig4 heatmap panel
# ---------------------------------------------------------------------------

RECLASS_REQUIRED_CELLS = ["A_popN_perN", "B_popN_perY", "C_popY_perN", "D_popY_perY"]

RECLASS_CELL_BY_STATE = {
    (0, 0): "A_popN_perN",
    (1, 0): "B_popN_perY",
    (0, 1): "C_popY_perN",
    (1, 1): "D_popY_perY",
}


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robustly coerce to boolean: True/False, 'true'/'false', 1/0, 1.0/0.0, NaN.

    NaN -> False by design (missing == no event/flag).
    """
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    map_true = {"true", "t", "1", "yes", "y"}
    map_false = {"false", "f", "0", "no", "n", "", "nan", "none", "null"}

    def _coerce(v):
        if pd.isna(v):
            return False
        if isinstance(v, bool):
            return v
        vstr = str(v).strip().lower()
        if vstr in map_true:
            return True
        if vstr in map_false:
            return False
        return True

    return s.map(_coerce).astype(bool)


def build_added_value_tables(
    df,
    *,
    pop_col: str = "out_popri",
    per_col: str = "out_perri_p95",
    outcome_col: str = "any_in_window",
    id_col: str = ID_COL,
    ts_col: str = "presenting_ts",
    verbose: bool = True,
) -> dict:
    """Mode 1 ONLY: outcome_col is a precomputed boolean "*_in_window" flag.

    Returns {"reclassification": DataFrame [n, events, event_rate_pct] indexed by
    cell A/B/C/D, "summary": DataFrame with one row of counts/sensitivity/specificity/PPV/NPV}.

    Definitions: A pop=0,per=0; B pop=0,per=1 (subclinical); C pop=1,per=0; D pop=1,per=1.
    """
    d = df.copy()

    needed = [pop_col, per_col, outcome_col, id_col, ts_col]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d[pop_col] = _to_bool_series(d[pop_col]).astype(int)
    d[per_col] = _to_bool_series(d[per_col]).astype(int)
    d[outcome_col] = _to_bool_series(d[outcome_col])

    before = len(d)
    d = d.dropna(subset=[id_col, ts_col]).copy()
    dropped_missing = before - len(d)

    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    before = len(d)
    d = d.sort_values([id_col, ts_col]).drop_duplicates(subset=[id_col, ts_col], keep="first")
    dropped_dupes = before - len(d)

    y = d[outcome_col].astype(int)

    cells = {
        "A_popN_perN": (d[pop_col].eq(0) & d[per_col].eq(0)),
        "B_popN_perY": (d[pop_col].eq(0) & d[per_col].eq(1)),
        "C_popY_perN": (d[pop_col].eq(1) & d[per_col].eq(0)),
        "D_popY_perY": (d[pop_col].eq(1) & d[per_col].eq(1)),
    }

    rows = []
    for name, mask in cells.items():
        n = int(mask.sum())
        ev = int(y[mask].sum())
        rate = 100.0 * ev / n if n else 0.0
        rows.append({"cell": name, "n": n, "events": ev, "event_rate_pct": round(rate, 2)})
    reclass_df = pd.DataFrame(rows).set_index("cell")

    N = len(d)
    E = int(y.sum())
    overall_rate = round(100.0 * E / N, 2) if N else 0.0

    def sens_spec_counts(flag_col: str):
        tp = int(((d[flag_col] == 1) & (y == 1)).sum())
        fn = int(((d[flag_col] == 0) & (y == 1)).sum())
        tn = int(((d[flag_col] == 0) & (y == 0)).sum())
        fp = int(((d[flag_col] == 1) & (y == 0)).sum())
        return tp, fn, tn, fp

    def sens_spec(flag_col: str):
        tp, fn, tn, fp = sens_spec_counts(flag_col)
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        return {
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "sensitivity_pct": round(100 * sens, 2) if sens == sens else np.nan,
            "specificity_pct": round(100 * spec, 2) if spec == spec else np.nan,
            "ppv_pct": round(100 * ppv, 2) if ppv == ppv else np.nan,
            "npv_pct": round(100 * npv, 2) if npv == npv else np.nan,
        }

    pop_stats = sens_spec(pop_col)
    per_stats = sens_spec(per_col)

    sub_mask = cells["B_popN_perY"]
    B_n = int(sub_mask.sum())
    B_ev = int(y[sub_mask].sum())
    B_rate = round(100.0 * B_ev / B_n, 2) if B_n else 0.0
    B_share = round(100.0 * B_ev / E, 2) if E else 0.0

    summary = pd.DataFrame(
        [
            {
                "total_n": N,
                "total_events": E,
                "overall_event_rate_pct": overall_rate,
                "pop_tp": pop_stats["tp"],
                "pop_fn": pop_stats["fn"],
                "pop_tn": pop_stats["tn"],
                "pop_fp": pop_stats["fp"],
                "pop_sensitivity_pct": pop_stats["sensitivity_pct"],
                "pop_specificity_pct": pop_stats["specificity_pct"],
                "pop_ppv_pct": pop_stats["ppv_pct"],
                "pop_npv_pct": pop_stats["npv_pct"],
                "per_tp": per_stats["tp"],
                "per_fn": per_stats["fn"],
                "per_tn": per_stats["tn"],
                "per_fp": per_stats["fp"],
                "per_sensitivity_pct": per_stats["sensitivity_pct"],
                "per_specificity_pct": per_stats["specificity_pct"],
                "per_ppv_pct": per_stats["ppv_pct"],
                "per_npv_pct": per_stats["npv_pct"],
                "added_events_by_per": B_ev,
                "subclinical_B_n": B_n,
                "subclinical_B_events": B_ev,
                "subclinical_B_event_rate_pct": B_rate,
                "subclinical_B_events_pct_of_all": B_share,
                "dropped_missing_rows": dropped_missing,
                "dropped_duplicate_rows": dropped_dupes,
            }
        ]
    )

    if verbose:
        print(f"Cohort used: n={N:,}, events={E:,} ({overall_rate}%)")

    return {"reclassification": reclass_df, "summary": summary}


def build_reclassification_2x2_grids(
    reclass_df: pd.DataFrame,
    *,
    abnormal_label: str,
    direction: str,
) -> dict:
    """Convert canonical A/B/C/D reclassification rows into ordered 2x2 display grids."""
    missing_cols = [col for col in ["n", "event_rate_pct"] if col not in reclass_df.columns]
    if missing_cols:
        raise KeyError(f"Reclassification table missing required columns: {missing_cols}")

    missing_cells = [cell for cell in RECLASS_REQUIRED_CELLS if cell not in reclass_df.index]
    if missing_cells:
        raise KeyError(f"Reclassification table missing required cells: {missing_cells}")

    if direction == "decrease":
        x_labels = [abnormal_label, "Normal"]
        y_labels = ["Normal", abnormal_label]
        pop_states = [1, 0]
        per_states = [0, 1]
    elif direction in {"increase", "two_tailed"}:
        x_labels = ["Normal", abnormal_label]
        y_labels = [abnormal_label, "Normal"]
        pop_states = [0, 1]
        per_states = [1, 0]
    else:
        raise ValueError(f"Unsupported reclassification direction: {direction}")

    rate_values = []
    count_values = []
    for per_i in per_states:
        rate_row = []
        count_row = []
        for pop_i in pop_states:
            cell_key = RECLASS_CELL_BY_STATE[(per_i, pop_i)]
            rate_row.append(float(reclass_df.loc[cell_key, "event_rate_pct"]))
            count_row.append(int(reclass_df.loc[cell_key, "n"]))
        rate_values.append(rate_row)
        count_values.append(count_row)

    rate_df = pd.DataFrame(rate_values, index=pd.Index(y_labels, name="PerRI"), columns=pd.Index(x_labels, name="PopRI"))
    count_df = pd.DataFrame(count_values, index=rate_df.index, columns=rate_df.columns)
    return {"rate_df": rate_df, "count_df": count_df}


# ---------------------------------------------------------------------------
# fig3 KM panel: setpoint selection + diagnosis-anchored at-risk cohorts
# ---------------------------------------------------------------------------


def get_one_setpoint(filtered_df, use_personalized_logic=True, model="bayesian", min_isolated=MIN_MEASUREMENTS, min_dts=None, max_dts=None):
    """Select one setpoint per (patient, test_code), grouped by test_code.

    If personalized, gets the first setpoint that meets the min_isolated
    criteria; else gets the last setpoint that meets the min_isolated criteria.
    """
    filtered_setpoints_df = filtered_df[filtered_df["model"] == model].copy()
    filtered_setpoints_df[TS_COL] = pd.to_datetime(filtered_setpoints_df[TS_COL])

    if min_dts:
        filtered_setpoints_df = filtered_setpoints_df[filtered_setpoints_df[TS_COL] >= pd.Timestamp(min_dts)]
    if max_dts:
        filtered_setpoints_df = filtered_setpoints_df[filtered_setpoints_df[TS_COL] <= max_dts]

    if use_personalized_logic:
        filtered_setpoints_df = filtered_setpoints_df[filtered_setpoints_df[INDEX_COL] == min_isolated].sort_values([ID_COL, TEST_CODE_COL, INDEX_COL, TS_COL], ascending=[True, True, True, True])
        filtered_setpoints_df = filtered_setpoints_df.groupby([ID_COL, TEST_CODE_COL], as_index=False).tail(1).reset_index(drop=True)
    else:
        filtered_setpoints_df = filtered_setpoints_df[(filtered_setpoints_df[INDEX_COL] >= min_isolated)].sort_values([ID_COL, TEST_CODE_COL, INDEX_COL], ascending=[True, True, True])
        filtered_setpoints_df = filtered_setpoints_df.groupby([ID_COL, TEST_CODE_COL], as_index=False).head(1).reset_index(drop=True)

    return filtered_setpoints_df


def precompute_cutoffs(filtered_setpoints_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute the 25th/75th percentile setpoint cutoffs for each test_code and sex."""
    percentiles = [25, 75]
    precomputed_cutoffs_df = (
        filtered_setpoints_df.groupby([TEST_CODE_COL, SEX_COL])[MU]
        .apply(lambda x: {pct: np.percentile(x.dropna(), pct) for pct in percentiles} if not x.empty else {pct: np.nan for pct in percentiles})
        .unstack()
        .reset_index()
        .melt(id_vars=[TEST_CODE_COL, SEX_COL], var_name="percentile", value_name="cutoff")
    )
    return precomputed_cutoffs_df


def expand_dx2setpoint(dx2setpoint: dict) -> dict:
    """Create dict with keys as 'dx_name (setpoint_type)' and values as the original config dict."""
    new_dict = {}
    for dx_name, configs in dx2setpoint.items():
        if not isinstance(configs, list):
            configs = [configs]
        for conf in configs:
            marker = conf["setpoint_type"]
            new_key = f"{dx_name} ({marker})"
            new_dict[new_key] = conf
    return new_dict


def get_base_population(filtered_setpoints_df: pd.DataFrame, precomputed_cutoffs_df: pd.DataFrame):
    """Depending on the cutoff, get the patients that meet these cutoffs."""
    percentiles = [25, 75]

    population_base = filtered_setpoints_df[[ID_COL, SEX_COL]].drop_duplicates(subset=[ID_COL]).copy()

    mu_values = filtered_setpoints_df[[TEST_CODE_COL, ID_COL, MU, SEX_COL]].copy()
    mu_values = mu_values.merge(precomputed_cutoffs_df, on=[TEST_CODE_COL, SEX_COL], how="left")

    for pct in percentiles:
        mu_values[f"meets_cutoff_{pct}"] = False
        mu_values.loc[mu_values["percentile"] == pct, f"meets_cutoff_{pct}"] = (mu_values[SEX_COL] == "M") & (mu_values[MU] < mu_values["cutoff"]) if pct < 50 else (mu_values[MU] >= mu_values["cutoff"])

    return population_base, mu_values


def _get_groups_from_percentiles(df, pct_cutoff, precomputed_cutoffs=None, setpoint_type=None):
    sex = df[SEX_COL]
    setpoint = df[MU]
    m_idx = sex == "M"
    f_idx = sex == "F"

    if precomputed_cutoffs is not None and setpoint_type is not None:
        cutoff_m = precomputed_cutoffs.get((setpoint_type, "M", pct_cutoff), np.nan)
        cutoff_f = precomputed_cutoffs.get((setpoint_type, "F", pct_cutoff), np.nan)
    else:
        male_setpoint = setpoint[m_idx]
        female_setpoint = setpoint[f_idx]
        cutoff_m = np.percentile(male_setpoint, pct_cutoff) if not male_setpoint.empty else np.nan
        cutoff_f = np.percentile(female_setpoint, pct_cutoff) if not female_setpoint.empty else np.nan

    if pct_cutoff < 50:
        group0 = (m_idx & (setpoint >= cutoff_m)) | (f_idx & (setpoint >= cutoff_f))
        group1 = (m_idx & (setpoint < cutoff_m)) | (f_idx & (setpoint < cutoff_f))
    else:
        group0 = (m_idx & (setpoint < cutoff_m)) | (f_idx & (setpoint < cutoff_f))
        group1 = (m_idx & (setpoint >= cutoff_m)) | (f_idx & (setpoint >= cutoff_f))

    if not (isinstance(group0, pd.Series) and isinstance(group1, pd.Series)):
        raise ValueError("group0 and group1 must be pandas Series.")
    if not (group0.index.equals(df.index) and group1.index.equals(df.index)):
        raise ValueError("group0 and group1 must have the same index as df.")
    return [group0, group1]


def get_at_risk_population(population_base, one_dx_incident, test_code, observation_period_start, use_personalized_logic, filtered_setpoints_df, verbose=False):
    """Prepare the patient cohort at risk for a specific diagnosis, anchored on their setpoint."""
    population_base_with_dx_data = population_base.merge(one_dx_incident[[ID_COL, "earliest_contact_date"]], on=ID_COL, how="left")
    population_base_with_dx_data = population_base_with_dx_data.merge(filtered_setpoints_df[[ID_COL, TEST_CODE_COL, MU, TS_COL]], on=[ID_COL], how="left")
    population_base_with_dx_data = population_base_with_dx_data[population_base_with_dx_data[TEST_CODE_COL] == test_code].copy()

    if use_personalized_logic:
        population_base_with_dx_data_at_risk = population_base_with_dx_data[
            (population_base_with_dx_data["earliest_contact_date"].isna()) | (population_base_with_dx_data["earliest_contact_date"] > population_base_with_dx_data[TS_COL])
        ].copy()
    else:
        population_base_with_dx_data_at_risk = population_base_with_dx_data[
            (population_base_with_dx_data["earliest_contact_date"].isna()) | (population_base_with_dx_data["earliest_contact_date"] >= observation_period_start)
        ].copy()

    if verbose:
        print(f"# Patients at risk: {population_base_with_dx_data_at_risk.shape[0]}/{population_base_with_dx_data.shape[0]}")

    return population_base_with_dx_data_at_risk


def get_groups_from_config(df_at_risk_for_event, criteria_setpoint_configs, precomputed_cutoffs_df):
    """Using df_at_risk_for_event (one row per at-risk patient), get grouping from criteria_setpoint_configs."""
    precomputed_cutoffs_dict = {}
    for _, row in precomputed_cutoffs_df.iterrows():
        precomputed_cutoffs_dict[(row[TEST_CODE_COL], row[SEX_COL], row["percentile"])] = row["cutoff"]

    group_mask = _get_groups_from_percentiles(
        df=df_at_risk_for_event,
        pct_cutoff=criteria_setpoint_configs["pct_cutoff"],
        precomputed_cutoffs=precomputed_cutoffs_dict,
        setpoint_type=criteria_setpoint_configs["setpoint_type"],
    )[1]

    return [~group_mask, group_mask]


def compute_event_time(df: pd.DataFrame, dx_col: str = "earliest_contact_date", death_col: str = "death_ts_filled", ref_col: str = TS_COL, observation_window: float = 5):
    """Compute time to first diagnosis and censoring (in years) for Kaplan-Meier analysis.

    Death is treated as censoring. Patients are censored at the earlier of
    death, `observation_window` years after the reference date, or today.
    """
    dx = pd.to_datetime(df[dx_col])
    death = pd.to_datetime(df[death_col])
    origin = pd.to_datetime(df[ref_col])

    time_to_dx = (dx - origin).dt.total_seconds() / (365.25 * 24 * 3600)
    time_to_death = (death - origin).dt.total_seconds() / (365.25 * 24 * 3600)
    time_to_today = (pd.Timestamp.today() - origin).dt.total_seconds() / (365.25 * 24 * 3600)

    time_to_censor = np.minimum(observation_window, time_to_today)
    event_time = np.minimum.reduce([time_to_dx.fillna(np.inf), time_to_death.fillna(np.inf), time_to_censor])
    event_observed = (time_to_dx.notna()) & (time_to_dx <= event_time)

    return event_time, event_observed


def _fit_km_for_groups(groups_for_km_calc, event_time, event_observed, dx_name, current_criteria_pct_label, current_criteria_type_label):
    """Fit Kaplan-Meier curves for each group; long-format output for utils/visuals_fig3.py:fig3km."""
    res = []
    for i, group in enumerate(groups_for_km_calc):
        if not isinstance(group, pd.Series) or group.empty or group.sum() == 0:
            continue

        pct = float(current_criteria_pct_label)
        if pct < 50:
            group_label_km = f">= {current_criteria_pct_label}%" if i == 0 else f"< {current_criteria_pct_label}%"
        else:
            group_label_km = f"< {current_criteria_pct_label}%" if i == 0 else f">= {current_criteria_pct_label}%"
        group_event_time = event_time[group]
        group_event_observed_status = event_observed[group]

        if len(group_event_time) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(group_event_time, group_event_observed_status)
        n_km = len(kmf.survival_function_)
        if n_km > 0:
            res.append(
                pd.DataFrame(
                    {
                        "timeline": kmf.survival_function_.index,
                        "survival": kmf.survival_function_["KM_estimate"].values,
                        "diagnosis": [dx_name] * n_km,
                        "group": [group_label_km] * n_km,
                        "setpoint_type": [current_criteria_type_label] * n_km,
                    }
                )
            )
    res = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    if not res.empty:
        res["count"] = int(event_observed.sum())
    return res
