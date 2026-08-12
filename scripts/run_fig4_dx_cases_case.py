"""Case inspection for fig4_dx_cases' trajectory panel (column 0).

Finds and plots candidate patients by KM group, sorted by baseline plausibility then |delta|.
Mirrors bayesian-setpoint-inference's scripts/figures/fig4_prog_case.py; the
tiered example-patient selector it feeds (saved case -> group B -> naive
fallback) lives in scripts/run_fig4_dx_cases.py.

Groups (use single letter with --group):
  b  pop_in & per_out  [normal by PopRI, flagged by PerRI -- Subclinical]  <- default
  a  pop_in & per_in   [both normal -- Concordant Negative]
  c  pop_out & per_in  [abnormal by PopRI, normal by PerRI -- Cleared]
  d  pop_out & per_out [both abnormal -- Concordant Positive]

"Sorted by PopRI-normal baseline" means: for each candidate, look at the last
BASELINE_SORT_N isolated lab values before the presenting value. Candidates
whose baseline is entirely inside PopRI rank first; within that, sort by
larger |delta|, then |delta/sigma|.

Usage:
    python -m scripts.run_fig4_dx_cases_case --outcome leukemia
    python -m scripts.run_fig4_dx_cases_case --outcome aki --n 12
    python -m scripts.run_fig4_dx_cases_case --outcome leukemia --group all
    python -m scripts.run_fig4_dx_cases_case --outcome hypothyroidism --group d
    python -m scripts.run_fig4_dx_cases_case --outcome leukemia --accept 2
"""

from __future__ import annotations

import argparse
import math
import webbrowser
from pathlib import Path

from utils.bootstrap import ensure_importable

ensure_importable()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from constants.runtime import DELTA_COL, ID_COL, SIGMA  # noqa: E402
from scripts.run_fig4_dx_cases import (  # noqa: E402
    CASES_JSON_PATH,
    _build_group_masks,
    _direction_columns,
    _load_cases,
    _save_case,
    compute_one_outcome,
    find_candidates,
)
from utils.io import load_demographics_csv, load_dx_incident  # noqa: E402
from utils.logging_utils import tagged_stdout  # noqa: E402
from utils.progression.config import OUTCOME_REGISTRY  # noqa: E402
from utils.visuals_fig4 import plot_single_patient_history_on_ax  # noqa: E402
from utils.visuals_shared import save_fig_as_svg  # noqa: E402

KM_GROUPS = {
    "a": "pop_in & per_in   [Concordant Negative -- both normal]",
    "b": "pop_in & per_out  [Subclinical -- normal by PopRI, flagged by PerRI]",
    "c": "pop_out & per_in  [Cleared -- abnormal by PopRI, normal by PerRI]",
    "d": "pop_out & per_out [Concordant Positive -- both abnormal]",
}
GROUP_CHOICES = ["a", "b", "c", "d", "all"]
OUTCOME_CHOICES = ["all", *OUTCOME_REGISTRY.keys()]

DEMOGRAPHICS_FILE = "demographics.csv"


def _load_outcome_data(outcome_name: str, *, input_dir: Path, output_dir: Path, force: bool):
    """Reuses compute_one_outcome's cohort/setpoint cache instead of re-deriving it."""
    dx_incident_path = Path(__file__).resolve().parent.parent / "outputs" / "dx_incident" / "dx_incident.csv"
    dx_incident = load_dx_incident(dx_incident_path)
    demographics_df = load_demographics_csv(input_dir / DEMOGRAPHICS_FILE)
    spec = compute_one_outcome(outcome_name, input_dir=input_dir, dx_incident=dx_incident, demographics_df=demographics_df, output_dir=output_dir, force=force)
    return spec["cohort"], spec["sp_df"], spec["outcome_cfg"]


def print_group_summary(presenting_df: pd.DataFrame, outcome_cfg) -> None:
    """Print a 2x2 breakdown of group sizes and event rates."""
    per_col, pop_col = _direction_columns(outcome_cfg.flag_below, outcome_cfg.flag_above)
    missing = [c for c in [pop_col, per_col] if c not in presenting_df.columns]
    if missing:
        print(f"[case] Cannot summarize -- missing columns: {missing}")
        return

    masks = _build_group_masks(presenting_df, per_col, pop_col)
    has_event = "any_in_window" in presenting_df.columns
    total = len(presenting_df)
    print(f"\n{'-'*60}")
    print(f"{'Group':<8} {'N':>6}  {'%':>5}  {'Events':>7}  {'Event%':>7}")
    print(f"{'-'*60}")
    for name, mask in masks.items():
        sub = presenting_df[mask]
        n = len(sub)
        pct = 100 * n / total if total else 0
        if has_event:
            events = int(sub["any_in_window"].sum())
            epct = 100 * events / n if n else 0
            print(f"  {name:<6} {n:>6}  {pct:>4.1f}%  {events:>7}  {epct:>6.1f}%")
        else:
            print(f"  {name:<6} {n:>6}  {pct:>4.1f}%")
    print(f"{'-'*60}")
    print(f"  {'TOTAL':<6} {total:>6}\n")


def _print_ranked_cases(candidates: pd.DataFrame, outcome_name: str, n: int = 20, group: str = "b") -> None:
    """Print ranked candidate list and guide the user to --accept N."""
    if candidates.empty:
        print(f"[case] No candidates found for {outcome_name}.")
        return

    saved = _load_cases()
    current = saved.get(outcome_name)

    group_desc = KM_GROUPS.get(group, "")
    print(f"\n{'-'*60}")
    print(f"Top candidates -- {outcome_name} (group {group})  (sorted by PopRI-normal baseline, then |delta|)")
    if group_desc:
        print(f"  {group_desc}")
    if current:
        print(f"  Currently saved: {current}")
    print(f"{'-'*60}")
    print(f"  {'#':>3}  {'base':>6}  {'|d|':>6}  {'|d/s|':>6}  {'delta':>7}  {'dx (days)':>10}  {ID_COL}")

    for i, (_, row) in enumerate(candidates.head(n).iterrows(), 1):
        delta = row.get(DELTA_COL, np.nan)
        sigma = row.get(SIGMA, np.nan)
        z = abs(delta / sigma) if pd.notna(delta) and pd.notna(sigma) and sigma != 0 else np.nan
        dte = row.get("days_to_event", np.nan)
        patient_id = row[ID_COL]
        baseline = "PopRI" if bool(row.get("_baseline_popri_ok", False)) else "other"
        da_str = f"{abs(delta):6.2f}" if pd.notna(delta) else "   n/a"
        z_str = f"{z:6.2f}" if pd.notna(z) else "   n/a"
        d_str = f"{delta:+7.2f}" if pd.notna(delta) else "    n/a"
        dte_str = f"{int(dte):>10}" if pd.notna(dte) else "        n/a"
        marker = " <- saved" if str(patient_id) == str(current) else ""
        print(f"  #{i:>2}  {baseline:>6}  {da_str}  {z_str}  {d_str}  {dte_str}  {patient_id}{marker}")

    print(f"\n  To save a case, re-run with --accept N")
    print(f"  e.g.:  python -m scripts.run_fig4_dx_cases_case --outcome {outcome_name} --accept 1")
    print(f"{'-'*60}\n")


def plot_cases(
    candidates: pd.DataFrame,
    sp_df: pd.DataFrame,
    outcome_cfg,
    *,
    n: int | None = None,
    ncols: int = 3,
    outcome_name: str = "unknown",
) -> Figure | None:
    """Plot up to `n` candidate patient trajectories in a grid."""
    if candidates.empty:
        print(f"[case] No candidates to plot for {outcome_name}.")
        return None

    top = candidates.head(n) if n is not None else candidates
    print(f"[case] {len(candidates)} candidates found; plotting top {len(top)}.")

    test_code = outcome_cfg.markers[0]
    lower, upper = outcome_cfg.flag_below, outcome_cfg.flag_above

    nrows = math.ceil(len(top) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 1.5 * nrows), squeeze=False)

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i // ncols][i % ncols]
        patient_id = row[ID_COL]
        patient_sp = sp_df[sp_df[ID_COL] == patient_id]
        presenting_row = candidates[candidates[ID_COL] == patient_id]

        if patient_sp.empty:
            ax.text(0.5, 0.5, f"No SP data\n{patient_id}", ha="center", va="center", transform=ax.transAxes)
            continue

        plot_single_patient_history_on_ax(
            ax=ax,
            patient_sp_data=patient_sp,
            presenting_row=presenting_row,
            event_row=presenting_row,
            test_code=test_code,
            lower=lower,
            upper=upper,
            add_legend=(i == 0),
        )
        ax.set_title(f"Case {i+1}", fontsize=8)

    for j in range(len(top), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(outcome_name, fontsize=9, y=1.01)
    fig.tight_layout()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and select example patients for fig4_dx_cases' trajectory panel.",
        epilog="""
Examples:
  python -m scripts.run_fig4_dx_cases_case --outcome leukemia
  python -m scripts.run_fig4_dx_cases_case --outcome aki --n 12
  python -m scripts.run_fig4_dx_cases_case --outcome leukemia --group all --no-events-only
  python -m scripts.run_fig4_dx_cases_case --outcome hypothyroidism --group d
  python -m scripts.run_fig4_dx_cases_case --outcome leukemia --accept 2

Groups: a=concordant-neg (pop_in&per_in), b=subclinical (pop_in&per_out) [default],
        c=cleared (pop_out&per_in), d=concordant-pos (pop_out&per_out)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "fig4_dx_cases")
    parser.add_argument("--outcome", choices=OUTCOME_CHOICES, default=list(OUTCOME_REGISTRY.keys())[0])
    parser.add_argument("--group", choices=GROUP_CHOICES, default="b", help="KM group to inspect: a=concordant-neg, b=subclinical (default), c=cleared, d=concordant-pos.")
    parser.add_argument("--n", type=int, default=None, help="Number of cases to plot.")
    parser.add_argument("--ncols", type=int, default=3, help="Grid columns.")
    parser.add_argument("--no-events-only", dest="events_only", action="store_false", default=True, help="Include patients without a diagnosis event.")
    parser.add_argument("--accept", type=int, default=None, metavar="N", help="Save the Nth ranked candidate to the cases config and exit.")
    parser.add_argument("--open", action="store_true", help="Open the plotted grid in the browser.")
    parser.add_argument("--force", action="store_true", help="Recompute caches.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    outcomes = list(OUTCOME_REGISTRY.keys()) if args.outcome == "all" else [args.outcome]

    with tagged_stdout("fig4_dx_cases_case"):
        for outcome_name in outcomes:
            print(f"\n{'='*60}")
            print(f"Loading data for: {outcome_name}")
            try:
                presenting_df, sp_df, outcome_cfg = _load_outcome_data(outcome_name, input_dir=args.input_dir, output_dir=args.output_dir, force=args.force)
            except Exception as exc:
                print(f"[case] Could not load {outcome_name}: {exc}")
                continue

            print_group_summary(presenting_df, outcome_cfg)

            candidates = find_candidates(presenting_df, sp_df, outcome_cfg, group=args.group, events_only=args.events_only)
            _print_ranked_cases(candidates, outcome_name, group=args.group)

            if args.accept is not None:
                if args.accept < 1 or args.accept > len(candidates):
                    print(f"[case] --accept {args.accept} out of range (1-{len(candidates)})")
                    continue
                patient_id = str(candidates.iloc[args.accept - 1][ID_COL])
                _save_case(outcome_name, patient_id)
                continue

            fig = plot_cases(candidates=candidates, sp_df=sp_df, outcome_cfg=outcome_cfg, n=args.n, ncols=args.ncols, outcome_name=outcome_name)
            if fig is not None:
                title = f"fig4_dx_cases_case_{outcome_name}_{args.group}"
                fig_path = args.output_dir / f"{title}.svg"
                save_fig_as_svg(fig, title, fig_path)
                if args.open:
                    webbrowser.open(fig_path.resolve().as_uri())
                plt.close(fig)


if __name__ == "__main__":
    main()
