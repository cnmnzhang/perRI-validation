"""Fig3a/b: mortality hazard ratios by setpoint (mu/sigma/cv), across all 43 markers.

  fig3a ("HR by model"): one mortality Cox regression per marker, using each
    patient's 5th setpoint, filtered for measurements in the popRI.
  fig3b ("HR by baseline index"): the same Cox regression repeated using the
    setpoint estimated from only the patient's 1st/2nd/3rd/4th/5th isolated
    measurement, to see how HR estimates stabilize as more data accumulates.

Required inputs: `tests.csv` (anon_id, ts, test_code, result_value, sex) covering
all 43 pipeline markers and a Demographics table (anon_id, sex, birth_date,
death_ts). Marker display order comes from constants/marker_lab_config.py. 

Run:
    python -m scripts.run_fig3_hazard --input-dir data --output-dir outputs/fig3_hazard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.bootstrap import ensure_importable

ensure_importable()

import matplotlib.pyplot as plt  # noqa: E402

from utils.cache import cache_or_compute  # noqa: E402
from utils.io import load_demographics_csv  # noqa: E402
from utils.logging_utils import tagged_stdout, timed_step  # noqa: E402
from utils.setpoints import fit_markers_lazy  # noqa: E402
from utils.clinical.coxph import run_cox_summary  # noqa: E402
from utils.clinical.get import attach_ref_intervals, compute_within_normal_mask  # noqa: E402
from utils.clinical.run_clinical import get_one_setpoint  # noqa: E402
from utils.log_transform_markers import is_log_transform  # noqa: E402
from constants.marker_config import MARKER_IOI_ORDER, TESTCODES_LIST  # noqa: E402
from constants.runtime import CV_COL, ID_COL, INDEX_COL, MAX_FIT_DATE, MODEL_COL, MU, SEX_COL, SIGMA, TEST_CODE_COL, TS_COL  # noqa: E402
from utils.visuals_fig3 import fig3baseline, fig3hr  # noqa: E402

DEMOGRAPHICS_FILE = "demographics.csv"

OBSERVATION_PERIOD_START = "2014-01-01"  # matches the real pipeline's fig3a/b window
MAX_DTS = MAX_FIT_DATE
MIN_ISOLATED_PANEL_A = 5
BASELINE_INDICES = [1, 2, 3, 4, 5]
VARIABLES = (MU, CV_COL)

# bayesian-setpoint-inference's config/opt_config.py:MIN_MEASUREMENTS -- the bar
# filter_sp_df's "Minimum Measurements Filter" enforces on the full setpoint sequence
# before any per-baseline-index selection. Distinct from (and stricter than)
# constants.runtime.DEFAULT_MIN_MEASUREMENTS (3), which only gates whether perri fits a
# patient at all (utils/setpoints.py) -- a patient with 3-4 isolated measurements passes
# that bar and gets fit, but must still be excluded here to match the live pipeline.
MIN_MEASUREMENTS_FOR_FILTER = 5


def _build_setpoints_with_demog(sp_df: pd.DataFrame, demog_df: pd.DataFrame) -> pd.DataFrame:
    return sp_df.merge(demog_df[demog_df[SEX_COL].isin(["F", "M"])], on=[ID_COL, SEX_COL], how="left")


def _apply_normal_filter(sp_df: pd.DataFrame) -> pd.DataFrame:
    sp_with_ri = attach_ref_intervals(sp_df)
    return sp_with_ri[compute_within_normal_mask(sp_with_ri)].copy()


def _filter_invalid_cv_patients(sp_df: pd.DataFrame) -> pd.DataFrame:
    """Drop every (patient, test_code, model)'s entire setpoint sequence if any of its
    measurements from the 4th isolated point onward (index >= 3) has an "invalid" cv --
    vendored from bayesian-setpoint-inference's utils/setpoints_runner.py:filter_sp_df's
    "CV Filter" step (the last of its three steps -- see _filter_sp_df).

    "Invalid" differs by whether the marker is fit in log-space (utils/log_transform_markers):
    log-space markers back-transform to cv = sqrt(exp(sigma_log^2) - 1), which legitimately
    exceeds 1 for a highly variable patient, so only cv < 0 (a numerically degenerate case) is
    rejected for them; non-log markers keep the standard cv in [0, 1] guard.
    """
    df = sp_df.copy()
    if CV_COL not in df.columns:
        df[CV_COL] = df[SIGMA] / df[MU]
    is_log = df[TEST_CODE_COL].map(is_log_transform)
    invalid_mask = (df[INDEX_COL] >= 3) & ((df[CV_COL] < 0) | (~is_log & (df[CV_COL] > 1)))
    invalid_combos = df.loc[invalid_mask, [ID_COL, TEST_CODE_COL, MODEL_COL]].drop_duplicates()
    if invalid_combos.empty:
        return sp_df
    merged = sp_df.merge(invalid_combos.assign(_invalid=True), on=[ID_COL, TEST_CODE_COL, MODEL_COL], how="left")
    return merged[merged["_invalid"].isna()].drop(columns=["_invalid"])


def _filter_sp_df(sp_df: pd.DataFrame) -> pd.DataFrame:
    """Vendored from bayesian-setpoint-inference's utils/setpoints_runner.py:filter_sp_df,
    applied to fig3's setpoints before any per-patient baseline-index selection
    (get_one_setpoint), matching where and in what order the live pipeline applies it:

    1. Date filter: drop measurements at/after MAX_FIT_DATE.
    2. Minimum measurements filter: drop a (patient, test_code, model)'s entire sequence
       if it has fewer than MIN_MEASUREMENTS_FOR_FILTER rows *after* the date filter above.
       This is the one that actually explains most of fig3b's low-baseline-index drift: a
       patient with only 3-4 isolated measurements passes compute_sp_df's looser fitting bar
       (constants.runtime.DEFAULT_MIN_MEASUREMENTS = 3) and gets fit, contributing rows at
       low index values (1, 2) -- but the live pipeline requires 5 before considering them
       for fig3 at all, at any baseline index. Without this step, those extra patients
       inflate every low-index cohort (baseline_index=1 was ~5.7x ground truth's n) while
       naturally vanishing by index 4-5, since they don't have enough measurements to reach
       there -- exactly the "divergence shrinks as baseline_index grows" pattern seen in
       data/UWM/ comparisons.
    3. CV filter (_filter_invalid_cv_patients).
    """
    dated = sp_df[sp_df[TS_COL] < pd.Timestamp(MAX_FIT_DATE)]
    counts = dated.groupby([ID_COL, TEST_CODE_COL, MODEL_COL])[ID_COL].transform("size")
    met_min = dated[counts >= MIN_MEASUREMENTS_FOR_FILTER]
    return _filter_invalid_cv_patients(met_min)


def build_hr_by_model(sp_df_demog: pd.DataFrame) -> pd.DataFrame:
    """fig3a: one Cox regression per marker, using each patient's personalized (index=5) setpoint."""
    filtered = get_one_setpoint(
        sp_df_demog,
        use_personalized_logic=True,
        model="bayesian",
        min_isolated=MIN_ISOLATED_PANEL_A,
        min_dts=OBSERVATION_PERIOD_START,
        max_dts=MAX_DTS,
    )
    filtered = _apply_normal_filter(filtered)

    cox_summary = {}
    for test_code, group_df in filtered.groupby(TEST_CODE_COL):
        results = run_cox_summary(group_df)
        if any(results.values()):
            cox_summary[(test_code, "bayesian")] = results
        else:
            print(f"fig3_hazard: no Cox results for {test_code}")
    return cox_summary


def build_hr_by_baseline(sp_df_demog: pd.DataFrame) -> pd.DataFrame:
    """fig3b: repeat the Cox regression using the setpoint from each patient's 1st-5th isolated measurement."""
    records = []
    for k in BASELINE_INDICES:
        sp = get_one_setpoint(
            sp_df_demog,
            use_personalized_logic=True,
            model="bayesian",
            min_isolated=k,
            min_dts=OBSERVATION_PERIOD_START,
            max_dts=MAX_DTS,
        )
        if sp.empty:
            continue
        sp = sp.copy()
        sp["baseline_index"] = k
        sp["baseline_label"] = str(k)
        records.append(sp)
    if not records:
        return pd.DataFrame()
    hr_df = pd.concat(records, ignore_index=True)
    hr_df = _apply_normal_filter(hr_df)

    rows = []
    for test_code, tc_df in hr_df.groupby(TEST_CODE_COL):
        with timed_step("run_cox_summary", f"{test_code} across all indices"):
            for (baseline_label, baseline_index), group_df in tc_df.groupby(["baseline_label", "baseline_index"]):
                results = run_cox_summary(group_df)
                for variable in VARIABLES:
                    summary = results.get(variable, {})
                    if not summary:
                        continue
                    rows.append(
                        {
                            TEST_CODE_COL: test_code,
                            "model": "bayesian",
                            "baseline_label": baseline_label,
                            "baseline_index": baseline_index,
                            "variable": variable,
                            "hr": summary["exp(coef)"],
                            "ci_lower": summary["exp(coef) lower 95%"],
                            "ci_upper": summary["exp(coef) upper 95%"],
                            "n": summary["n"],
                        }
                    )
    return pd.DataFrame(rows)


def _hr_by_model_df_to_cox_summary(hr_by_model_df: pd.DataFrame) -> dict:
    """Inverse of the {(test_code, model): {variable: coxph_summary}} -> rows flattening done
    when building hr_by_model_df, so fig3hr (which expects that dict shape) can plot straight
    from a cached fig3a_hr_by_model.csv without refitting anything."""
    cox_summary = {}
    if hr_by_model_df.empty:
        return cox_summary
    for (test_code, model), group in hr_by_model_df.groupby([TEST_CODE_COL, "model"]):
        cox_summary[(test_code, model)] = {row["variable"]: {"exp(coef)": row["exp(coef)"], "exp(coef) lower 95%": row["exp(coef) lower 95%"], "exp(coef) upper 95%": row["exp(coef) upper 95%"], "p": row["p"], "n": row["n"]} for _, row in group.iterrows()}
    return cox_summary


def run(*, input_dir: Path, output_dir: Path, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # sp_df_demog (a Bayesian fit per patient, per marker) is only needed if fig3a/b's own
    # cached plotting data isn't there yet -- computed lazily, at most once, and shared by
    # whichever of the two below actually need it.
    sp_df_holder = {}

    def _get_sp_df_demog() -> pd.DataFrame:
        if "sp_df_demog" not in sp_df_holder:
            demog_df = load_demographics_csv(input_dir / DEMOGRAPHICS_FILE)
            # force is this script's own fig3a/b cache only -- never cascades to the shared
            # setpoint dependency (run_setpoints_by_marker's job, not this script's).
            with timed_step("fit_setpoints", f"Fitting setpoints for {len(TESTCODES_LIST)} markers"):
                sp_df = fit_markers_lazy(input_dir, TESTCODES_LIST, force=False, label="fig3_hazard")
            sp_df_holder["sp_df"] = sp_df
            sp_df_demog = _build_setpoints_with_demog(sp_df, demog_df)
            sp_df_holder["sp_df_demog"] = _filter_sp_df(sp_df_demog)
        return sp_df_holder["sp_df_demog"]

    def _compute_hr_by_model() -> pd.DataFrame:
        with timed_step("hr_by_model", "Building fig3a (HR by marker)"):
            cox_summary = build_hr_by_model(_get_sp_df_demog())
        return pd.DataFrame(
            [
                {TEST_CODE_COL: tc, "model": model, "variable": var, **summary}
                for (tc, model), results in cox_summary.items()
                for var, summary in results.items()
                if summary
            ]
        )

    def _compute_hr_by_baseline() -> pd.DataFrame:
        with timed_step("hr_by_baseline", "Building fig3b (HR by baseline index)"):
            return build_hr_by_baseline(_get_sp_df_demog())

    # A rerun with fig3a/b's CSVs already present skips patient-level setpoint fitting and
    # Cox regression entirely (the expensive part) and plots straight from the saved data --
    # like fig3_dx's fig3_km_data.csv, this checks file existence only (not a content hash of
    # tests.csv/demographics.csv) -- delete these CSVs, or pass --force, after either changes.
    hr_by_model_df = cache_or_compute(output_dir / "fig3a_hr_by_model.csv", _compute_hr_by_model, force=force, file_format="csv")
    hr_baseline_df = cache_or_compute(output_dir / "fig3b_hr_by_baseline.csv", _compute_hr_by_baseline, force=force, file_format="csv")

    fig_a = None
    cox_summary = _hr_by_model_df_to_cox_summary(hr_by_model_df)
    if cox_summary:
        fig_a = fig3hr(cox_summary, MARKER_IOI_ORDER, variables=VARIABLES, save_path=output_dir / "fig3a_hr_by_model.svg", csv_path=output_dir / "fig3a_hr_by_model.csv")
        if fig_a is not None:
            plt.close(fig_a)

    fig_b = None
    if not hr_baseline_df.empty:
        fig_b = fig3baseline(hr_baseline_df, MARKER_IOI_ORDER, variables=VARIABLES, save_path=output_dir / "fig3b_hr_by_baseline.svg", csv_path=output_dir / "fig3b_hr_by_baseline.csv")
        if fig_b is not None:
            plt.close(fig_b)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_markers_fitted": int(sp_df_holder["sp_df"]["test_code"].nunique()) if "sp_df" in sp_df_holder and not sp_df_holder["sp_df"].empty else int(hr_by_model_df[TEST_CODE_COL].nunique()) if not hr_by_model_df.empty else 0,
        "fig3a_n_marker_model_rows": int(len(hr_by_model_df)),
        "fig3b_n_rows": int(len(hr_baseline_df)),
        "outputs": ["fig3a_hr_by_model.csv", "fig3a_hr_by_model.svg", "fig3b_hr_by_baseline.csv", "fig3b_hr_by_baseline.svg", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run fig3a/b's Bayesian-only HR panels.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "fig3_hazard")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    with tagged_stdout("fig3_hazard"):
        run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force)
    # print(json.dumps(manifest, indent=2, sort_keys=True))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
