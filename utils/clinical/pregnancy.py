"""Pregnancy cohort-building logic.

Two marker-outcome pairs (PAIR_SPECS): WBC -> pih (pregnancy-induced
hypertension) and HCT -> received_tf (received a blood transfusion). For each
pair: pre-conception isolated labs are fit via
perri_validation.utils.setpoints.compute_sp_df, the last pre-conception
estimate is carried forward as "the setpoint," and in-pregnancy labs are
joined back to it to compute gestational week/trimester and PerRI/PopRI flags
via add_oo.

"""

import warnings
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from utils.setpoints import compute_sp_df
from utils.clinical.run_clinical import add_oo, build_added_value_tables, build_reclassification_2x2_grids
from constants.runtime import ABOVE_STR, BELOW_STR, DELTA_COL, ID_COL, INDEX_COL, MEASUREMENT_COL, MU, OUT_PERRI_P95_LOWER_COL, OUT_PERRI_P95_UPPER_COL, OUT_POPRI_LOWER_COL, OUT_POPRI_UPPER_COL, SIGMA, TEST_CODE_COL, TS_COL

MIN_PREPREG_ISOLATED = 3
TRIMESTER_MIDPOINTS = {"t1": 6.5, "t2": 20.0, "t3": 34.5}
TRIMESTER_BINS = [0, 13, 28, 42]
TRIMESTER_LABELS = ["t1", "t2", "t3"]
PREDICTOR_LABELS = {MEASUREMENT_COL: "Presenting", DELTA_COL: "Delta"}


@dataclass(frozen=True)
class PairSpec:
    key: str
    test_code: str
    outcome_col: str
    direction: str  # "increase" or "decrease" -- which direction counts as abnormal
    title: str

    @property
    def stem(self) -> str:
        return f"{self.test_code.lower()}_{self.outcome_col.lower()}"

    @property
    def abnormal_label(self) -> str:
        return ABOVE_STR if self.direction == "increase" else BELOW_STR

    @property
    def pop_col(self) -> str:
        return OUT_POPRI_UPPER_COL if self.direction == "increase" else OUT_POPRI_LOWER_COL

    @property
    def per_col(self) -> str:
        return OUT_PERRI_P95_UPPER_COL if self.direction == "increase" else OUT_PERRI_P95_LOWER_COL


PAIR_SPECS: "OrderedDict[str, PairSpec]" = OrderedDict(
    [
        ("wbc_pih", PairSpec(key="wbc_pih", test_code="WBC", outcome_col="pih", direction="increase", title="Pre-eclampsia")),
        ("hct_received_tf", PairSpec(key="hct_received_tf", test_code="HCT", outcome_col="received_tf", direction="decrease", title="Received Transfusion")),
    ]
)


def compute_prepreg_setpoint_table(tests_df: pd.DataFrame, demog_df: pd.DataFrame, pair: PairSpec, *, force: bool = False) -> pd.DataFrame:
    """Fit each patient's pre-conception setpoint for pair.test_code.

    Returns one row per patient with columns: anon_id, setpoint_ts,
    setpoint_measurement, setpoint_index, n_isolated_prepreg, mu, sigma.
    """
    marker_tests = tests_df[tests_df[TEST_CODE_COL] == pair.test_code]
    with_conception = marker_tests.merge(demog_df[[ID_COL, "conception_date"]], on=ID_COL, how="inner")
    prepreg = with_conception[with_conception[TS_COL] < with_conception["conception_date"]].copy()
    if prepreg.empty:
        return pd.DataFrame()

    sp_df = compute_sp_df(prepreg[[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL, "sex"]], test_code=pair.test_code, min_measurements=MIN_PREPREG_ISOLATED, force=force)
    if sp_df.empty:
        return sp_df

    setpoint_table = (
        sp_df.sort_values([ID_COL, TS_COL])
        .groupby(ID_COL, as_index=False)
        .tail(1)
        .rename(columns={TS_COL: "setpoint_ts", MEASUREMENT_COL: "setpoint_measurement", INDEX_COL: "setpoint_index"})
        .reset_index(drop=True)
    )
    setpoint_table["n_isolated_prepreg"] = setpoint_table["setpoint_index"] + 1
    return setpoint_table


def compute_inpreg_analysis_df(tests_df: pd.DataFrame, demog_df: pd.DataFrame, setpoint_table: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    """Join in-pregnancy labs to each patient's pre-conception setpoint; compute
    gestational week/trimester and PerRI/PopRI flags."""
    if setpoint_table.empty:
        return pd.DataFrame()

    marker_tests = tests_df[tests_df[TEST_CODE_COL] == pair.test_code].merge(demog_df, on=ID_COL, how="inner")
    merge_cols = [ID_COL, "setpoint_ts", "setpoint_measurement", "setpoint_index", "n_isolated_prepreg", MU, SIGMA]
    df = marker_tests.merge(setpoint_table[merge_cols], on=ID_COL, how="inner")
    df = df[(df[TS_COL] >= df["conception_date"]) & (df[TS_COL] < df["delivery_date"])].copy()
    if df.empty:
        return df

    df["gestational_age_weeks"] = (df[TS_COL] - df["conception_date"]).dt.days / 7.0
    df = df[df["gestational_age_weeks"].between(0, 42)].copy()
    df["gestational_week"] = np.floor(df["gestational_age_weeks"]).astype(int)
    df["gestational_biweekly"] = ((df["gestational_week"] // 2) * 2).astype(int)
    df["trimester"] = pd.cut(df["gestational_age_weeks"], bins=TRIMESTER_BINS, labels=TRIMESTER_LABELS, right=False)
    df["sex"] = "F"
    df = add_oo(df, setpoint_col=MU, sigma_col=SIGMA, result_col=MEASUREMENT_COL, p=0.95)
    return df.sort_values([ID_COL, TS_COL]).reset_index(drop=True)


def select_trimester_midpoints(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (patient, trimester): the measurement closest to that trimester's midpoint."""
    if analysis_df.empty:
        return analysis_df
    temp = analysis_df.dropna(subset=["trimester"]).copy()
    temp["_tri_mid"] = temp["trimester"].astype(str).map(TRIMESTER_MIDPOINTS)
    temp["_dist"] = (temp["gestational_age_weeks"] - temp["_tri_mid"]).abs()
    temp = temp.sort_values([ID_COL, "trimester", "_dist", TS_COL]).groupby([ID_COL, "trimester"], observed=True, as_index=False).first()
    return temp.drop(columns=["_tri_mid", "_dist"]).reset_index(drop=True)


def task1_summary_df(analysis_df: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    """Biweekly quantile summary (q10/q25/q50/q75/q90) of the marker across gestation."""
    if analysis_df.empty:
        return pd.DataFrame()
    summary = (
        analysis_df.groupby([TEST_CODE_COL, "gestational_biweekly"], observed=True)[MEASUREMENT_COL]
        .agg(n="size", q10=lambda s: s.quantile(0.10), q25=lambda s: s.quantile(0.25), q50=lambda s: s.quantile(0.50), q75=lambda s: s.quantile(0.75), q90=lambda s: s.quantile(0.90))
        .reset_index()
        .sort_values([TEST_CODE_COL, "gestational_biweekly"])
        .reset_index(drop=True)
    )
    summary["pair"] = pair.key
    summary["outcome"] = pair.outcome_col
    return summary


def task1_setpoint_distribution(setpoint_table: pd.DataFrame) -> pd.Series:
    if setpoint_table.empty:
        return pd.Series(dtype=float)
    return setpoint_table[MU].dropna()


def _fit_univariate_logit(df: pd.DataFrame, outcome_col: str, feature: str) -> dict:
    """Standardized univariate logistic regression: exp(coef) as an odds ratio + 95% CI."""
    n = len(df)
    events = int(pd.to_numeric(df[outcome_col], errors="coerce").fillna(0).astype(int).sum())
    row = {"feature": feature, "predictor": PREDICTOR_LABELS[feature], "n": n, "events": events, "or": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "fit_ok": False}
    if n < 3 or events == 0 or events == n:
        return row

    feature_values = pd.to_numeric(df[feature], errors="coerce")
    if feature_values.notna().sum() < 3:
        return row
    std = float(feature_values.std())
    if not np.isfinite(std) or std == 0:
        return row

    x = ((feature_values - feature_values.mean()) / std).to_numpy(dtype=float)
    y = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    X = sm.add_constant(pd.DataFrame({"x": x}), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y, X).fit(disp=0)
        conf = model.conf_int()
        est = float(np.exp(model.params["x"]))
        lo = float(np.exp(conf.loc["x", 0]))
        hi = float(np.exp(conf.loc["x", 1]))
        if not all(np.isfinite([est, lo, hi])):
            return row
        row["or"], row["ci_lower"], row["ci_upper"], row["fit_ok"] = est, lo, hi, True
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        pass
    return row


def task2_results_df(trimester_df: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    """Trimester x {presenting value, delta} univariate odds ratios for pair.outcome_col."""
    if trimester_df.empty:
        return pd.DataFrame()
    df = trimester_df.dropna(subset=[MEASUREMENT_COL, DELTA_COL, "trimester", pair.outcome_col]).copy()
    rows = []
    for trimester in TRIMESTER_LABELS:
        tri_df = df[df["trimester"].astype(str) == trimester]
        for feature in [MEASUREMENT_COL, DELTA_COL]:
            result = _fit_univariate_logit(tri_df, pair.outcome_col, feature)
            result["trimester"] = trimester
            rows.append(result)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pair"] = pair.key
    out["test_code"] = pair.test_code
    out["outcome"] = pair.outcome_col
    return out[["pair", "test_code", "outcome", "trimester", "feature", "predictor", "n", "events", "or", "ci_lower", "ci_upper", "fit_ok"]]


def task3_payload(trimester_df: pd.DataFrame, pair: PairSpec, *, annotate_n: bool = False):
    """Trimester-1 PerRI x PopRI reclassification heatmap (rate/count grids + tidy table)."""
    if trimester_df.empty:
        return None, pd.DataFrame()
    df = trimester_df[trimester_df["trimester"].astype(str) == "t1"].copy()
    if df.empty:
        return None, pd.DataFrame()

    needed_cols = [pair.pop_col, pair.per_col, pair.outcome_col, ID_COL, TS_COL]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required Task 3 columns for {pair.key}: {missing}")

    tables = build_added_value_tables(df, pop_col=pair.pop_col, per_col=pair.per_col, outcome_col=pair.outcome_col, id_col=ID_COL, ts_col=TS_COL, verbose=False)
    reclass = tables["reclassification"].copy()
    grids = build_reclassification_2x2_grids(reclass, abnormal_label=pair.abnormal_label, direction=pair.direction)
    rate_grid, count_grid = grids["rate_df"], grids["count_df"]

    annot_grid = np.empty((2, 2), dtype=object)
    for i, per_label in enumerate(rate_grid.index):
        for j, pop_label in enumerate(rate_grid.columns):
            rate = rate_grid.loc[per_label, pop_label]
            n = int(count_grid.loc[per_label, pop_label])
            annot_grid[i, j] = (f"{rate:.1f}%\nn={n:,}" if annotate_n else f"{rate:.1f}%") if n > 0 and pd.notna(rate) else f"(n={n:,})"

    tidy = reclass.reset_index().copy()
    tidy["pair"] = pair.key
    tidy["test_code"] = pair.test_code
    tidy["outcome"] = pair.outcome_col
    tidy["trimester"] = "t1"

    payload = {"rate_df": rate_grid, "count_df": count_grid, "annot_grid": annot_grid, "reclassification": reclass}
    return payload, tidy
