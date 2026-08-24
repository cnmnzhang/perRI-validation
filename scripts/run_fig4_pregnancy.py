"""Fig4's pregnancy panel: does a personalized pre-pregnancy setpoint (and deviation
from it) predict adverse pregnancy outcomes better than population reference
intervals, and does that change by trimester?

Two marker-outcome pairs (utils/clinical/pregnancy.py:PAIR_SPECS):
WBC -> pih (pregnancy-induced hypertension), HCT -> received_tf (received a
blood transfusion). For each pair, produces a row of 3 panels: Task 1
(biweekly quantile-band trajectory across gestation + pre-pregnancy setpoint
distribution), Task 2 (trimester x {presenting value, delta} odds ratios via
univariate logistic regression), Task 3 (trimester-1 PerRI x PopRI
reclassification heatmap). Also runs a Fisher's exact test comparing
"flagged abnormal by both reference intervals" vs. "normal by both" event
rates at trimester 1.

Required inputs: a pregnancy_labs table (anon_id, ts, test_code,
result_value) and a pregnancy_outcomes_and_demogs table (anon_id, delivery_date,
gestational_age, rbc_tf, pih) -- see README.md. Setpoints are
computed via utils.setpoints.compute_sp_df (perri's isolation
filter already matches the real pipeline's pre-pregnancy isolation logic
exactly), not a pregnancy-specific model.

Run:
    python -m scripts.run_fig4_pregnancy --input-dir data --output-dir data/outputs/fig4_pregnancy
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from utils.bootstrap import ensure_importable

ensure_importable()

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

from utils.io import load_pregnancy_labs_csv, load_pregnancy_outcomes_and_demogs_csv  # noqa: E402
from utils.logging_utils import tagged_stdout, timed_step  # noqa: E402
from utils.clinical.pregnancy import PAIR_SPECS, compute_inpreg_analysis_df, compute_prepreg_setpoint_table, select_trimester_midpoints, task1_setpoint_distribution, task1_summary_df, task2_results_df, task3_payload
from constants.fig_config import A4_WIDTH, FIG4_ROW1_HEIGHT
from utils.visuals_fig4_pregnancy import plot_task1_on_ax, plot_task2_on_ax, plot_task3_on_ax
from utils.visuals_shared import save_fig_as_svg
from constants.runtime import ID_COL  # noqa: E402

PREGNANCY_LABS_FILE = "pregnancy_labs.csv"
PREGNANCY_OUTCOMES_AND_DEMOGS_FILE = "pregnancy_outcomes_and_demogs.csv"

_COL_RATIOS = [1.2, 0.6, 1.0]  # task1, task2, task3
_COMBINED_WIDTH = A4_WIDTH * 0.6


def _fisher_exact_stats(trimester_df: pd.DataFrame, pair) -> dict:
    """Fisher's exact test: trimester-1 event rate for patients flagged abnormal by
    both reference intervals vs. normal by both."""
    t1 = trimester_df[trimester_df["trimester"].astype(str) == "t1"]
    if t1.empty or pair.pop_col not in t1.columns or pair.per_col not in t1.columns:
        return {"error": "insufficient trimester-1 data"}

    pop = t1[pair.pop_col].astype(bool)
    per = t1[pair.per_col].astype(bool)
    outcome = t1[pair.outcome_col].astype(bool)
    inside_mask = pop & per
    outside_mask = (~pop) & (~per)

    def _group_stats(mask):
        sub = outcome[mask]
        n = int(mask.sum())
        e = int(sub.sum()) if n > 0 else 0
        return {"n": n, "n_events": e, "event_rate_pct": (100.0 * e / n) if n > 0 else float("nan")}

    inside, outside = _group_stats(inside_mask), _group_stats(outside_mask)
    odds_ratio, fisher_p = float("nan"), float("nan")
    if inside["n"] > 0 and outside["n"] > 0:
        table = [[inside["n_events"], inside["n"] - inside["n_events"]], [outside["n_events"], outside["n"] - outside["n_events"]]]
        try:
            odds_ratio, fisher_p = (float(v) for v in fisher_exact(table, alternative="two-sided"))
        except ValueError:
            pass

    result = {"inside_both": inside, "outside_both": outside, "odds_ratio": odds_ratio, "fisher_p": fisher_p}
    if math.isfinite(inside["event_rate_pct"]) and math.isfinite(outside["event_rate_pct"]):
        result["interpretation"] = f"{pair.test_code} patients flagged abnormal by both reference intervals (trimester 1) had {inside['event_rate_pct']:.1f}% {pair.outcome_col} rate vs {outside['event_rate_pct']:.1f}% in those normal by both (OR={odds_ratio:.2f}, p={fisher_p:.3g}, N={inside['n']} vs {outside['n']})."
    return result


def build_pair_bundle(tests_df: pd.DataFrame, demog_df: pd.DataFrame, pair, *, force: bool) -> dict:
    setpoint_table = compute_prepreg_setpoint_table(tests_df, demog_df, pair, force=force)
    analysis_df = compute_inpreg_analysis_df(tests_df, demog_df, setpoint_table, pair)
    trimester_df = select_trimester_midpoints(analysis_df)

    task1_df = task1_summary_df(analysis_df, pair)
    task2_df = task2_results_df(trimester_df, pair)
    task3_data, task3_tidy = task3_payload(trimester_df, pair, annotate_n=True)
    stats = _fisher_exact_stats(trimester_df, pair) if not trimester_df.empty else {"error": "no trimester data"}

    return {
        "setpoint_table": setpoint_table,
        "analysis_df": analysis_df,
        "trimester_df": trimester_df,
        "task1_df": task1_df,
        "task2_df": task2_df,
        "task3_payload": task3_data,
        "task3_tidy": task3_tidy,
        "stats": stats,
    }


def plot_mosaic(bundles: dict, path: Path) -> None:
    pairs = list(PAIR_SPECS.values())
    fig, axes = plt.subplots(
        len(pairs),
        3,
        figsize=(_COMBINED_WIDTH, FIG4_ROW1_HEIGHT * len(pairs)),
        squeeze=False,
        gridspec_kw={"width_ratios": _COL_RATIOS, "wspace": 0.7, "hspace": 0.8},
    )
    for row_idx, pair in enumerate(pairs):
        bundle = bundles[pair.key]
        plot_task1_on_ax(axes[row_idx, 0], bundle["task1_df"], pair, task1_setpoint_distribution(bundle["setpoint_table"]), legend=True, labelx=True)
        plot_task2_on_ax(axes[row_idx, 1], bundle["task2_df"], pair, legend=True, labelx=True)
        plot_task3_on_ax(axes[row_idx, 2], bundle["task3_payload"], pair, labelx=True)
        axes[row_idx, 0].text(-0.2, 1.2, pair.title, ha="left", va="center", fontsize=8, fontweight="bold", transform=axes[row_idx, 0].transAxes)

    fig.subplots_adjust(top=0.85, bottom=0.15)
    save_fig_as_svg(fig, "fig4_pregnancy_combined", path)
    plt.close(fig)


def run(*, input_dir: Path, output_dir: Path, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    tests_df = load_pregnancy_labs_csv(input_dir / PREGNANCY_LABS_FILE)
    demog_df = load_pregnancy_outcomes_and_demogs_csv(input_dir / PREGNANCY_OUTCOMES_AND_DEMOGS_FILE)

    bundles = {}
    for pair in PAIR_SPECS.values():
        with timed_step(pair.key, f"Building {pair.title} ({pair.test_code} -> {pair.outcome_col}) cohort"):
            bundles[pair.key] = build_pair_bundle(tests_df, demog_df, pair, force=force)
            # print the size of the analysis_df for each pair
            print(f"[{pair.key}] analysis_df: {len(bundles[pair.key]['analysis_df'])} rows, {len(bundles[pair.key]['analysis_df'][ID_COL].unique())} unique patients")

    for pair in PAIR_SPECS.values():
        bundle = bundles[pair.key]
        if not bundle["task1_df"].empty:
            bundle["task1_df"].to_csv(output_dir / f"{pair.stem}_task1_summary.csv", index=False)
        if not bundle["task2_df"].empty:
            bundle["task2_df"].to_csv(output_dir / f"{pair.stem}_task2_ors.csv", index=False)
        if not bundle["task3_tidy"].empty:
            bundle["task3_tidy"].to_csv(output_dir / f"{pair.stem}_task3_heatmap_t1.csv", index=False)

    mosaic_path = output_dir / "fig4_pregnancy_combined.svg"
    plot_mosaic(bundles, mosaic_path)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pairs": {
            pair.key: {
                "test_code": pair.test_code,
                "outcome_col": pair.outcome_col,
                "n_patients_with_setpoint": int(bundles[pair.key]["setpoint_table"][ID_COL].nunique()) if not bundles[pair.key]["setpoint_table"].empty else 0,
                "n_inpreg_rows": int(len(bundles[pair.key]["analysis_df"])),
                "stats": bundles[pair.key]["stats"],
            }
            for pair in PAIR_SPECS.values()
        },
        "outputs": [mosaic_path.name] + [f"{p.stem}_task{n}_{s}.csv" for p in PAIR_SPECS.values() for n, s in [(1, "summary"), (2, "ors"), (3, "heatmap_t1")]] + ["manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run fig4's pregnancy panel.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "outputs" / "fig4_pregnancy")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    with tagged_stdout("fig4_pregnancy"):
        manifest = run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force)
    # print(json.dumps(manifest, indent=2, sort_keys=True, default=str))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
