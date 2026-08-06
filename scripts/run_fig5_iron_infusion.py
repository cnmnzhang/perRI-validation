"""Iron infusion (Fig5) cohort: pre/post-course hemoglobin response to IV iron.

Required inputs: a Tests table (anon_id, ts, test_code, result_value, sex) containing
HB rows, and an iron_mar table (anon_id, ts) already pre-filtered to the intended
route/formulation (IV, iron sucrose) — see README.md.

There's only one marker/outcome here (unlike fig3_hazard/fig3_dx/fig4_dx_cases, which
each cover many markers or outcomes), so a build failure (missing HB, too few isolated
measurements, ...) can't degrade to a partial result -- but it's still caught and
recorded in manifest.json (as an "error" key) rather than left as an uncaught traceback,
matching how run_all.py and the other analyses report failure.

Reads HB via the per-marker split built by
`perri_validation.scripts.run_tests_by_marker` -- run that first (or use `run_all`,
which sequences it automatically); raises a clear FileNotFoundError with the command
to run if it hasn't been built yet -- unless `iv_iron_bundle/` is already fully
cached (see below), in which case the split is never even read. The course/cohort-
building functions themselves (identify_first_course, _select_pre_post_labs,
get_sp_from_courses, build_iv_cohort, attach_metrics,
build_iron_infusion_trajectories) live in
utils/clinical/run_clinical.py.

**HB setpoints prefer an already-fit full population over a fresh cohort-only fit**
(see _get_hb_setpoints): if `run_setpoints_by_marker`/`fig3_hazard`/`fig3_dx` have
already fit HB on the full Tests population, this just filters that cached result
down to the IV-iron cohort instead of fitting the (much smaller) cohort itself --
valid because compute_sp_df fits each patient independently, and HB's grid bounds
come from a fixed pop_ri, not from which other patients are in the batch. If the
full-population fit isn't cached, this never forces it -- it fits the cohort alone,
same as before.

**The whole 5-DataFrame bundle build_iv_iron_bundle() produces is itself cached**,
not just compute_sp_df's per-marker fit: each of hb_marker/first_courses/
iv_events_first/course_sp/iv_cohort is written to its own inspectable CSV under
output_dir/iv_iron_bundle/ (plus a small counts.json). A rerun with all 5 files
already present skips tests.csv/iron_mar.csv entirely and goes straight to
plot_mosaic() -- delete output_dir/iv_iron_bundle/, or pass --force, after
tests.csv/iron_mar.csv change.

Run:
    python -m perri_validation.scripts.run_fig5_iron_infusion --input-dir data --output-dir outputs/fig5_iron_infusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.bootstrap import ensure_importable

ensure_importable()

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils.io import load_iron_mar_csv, load_tests_marker_subset  # noqa: E402
from utils.logging_utils import tagged_stdout  # noqa: E402
from utils.visuals_shared import save_fig_as_svg  # noqa: E402
from utils.setpoints import compute_sp_df, is_fitted, is_fitted_canonical  # noqa: E402
from utils.clinical.inputs import IronInfusionConfig  # noqa: E402
from utils.clinical.run_clinical import (  # noqa: E402
    _select_pre_post_labs,
    attach_metrics,
    build_iron_infusion_trajectories,
    build_iv_cohort,
    get_sp_from_courses,
    identify_first_course,
    n_ids,
    overlap_ids,
)
from constants.fig_config import (  # noqa: E402
    A4_WIDTH,
    FIG5_COL1_WIDTH,
    FIG5_COL2_WIDTH,
    FIG5_COL3_WIDTH,
    FIG5_COL4_WIDTH,
    FIG5_ROW1_HEIGHT,
    FIG5_ROW2_HEIGHT,
    FIG5DOSE_RESPONSE_HSPACE,
    FIG5KM_HSPACE,
    FIG5TRAJECTORY_HSPACE,
    MOSAIC_HSPACE,
    MOSAIC_WSPACE,
)
from constants.runtime import ID_COL, MU, TS_COL  # noqa: E402
from utils.visuals_fig5 import (  # noqa: E402
    fig5dose_response_on_axes,
    fig5heatmap_on_ax,
    fig5interaction_on_ax,
    fig5response_distribution_on_axes,
    fig5swimmer_on_ax,
    fig5trajectory_on_axes,
)

IRON_MAR_FILE = "iron_mar.csv"

# The 5 DataFrames build_iv_iron_bundle() produces, each cached as its own inspectable CSV
# under output_dir/iv_iron_bundle/ 
BUNDLE_DATE_COLUMNS = {
    "hb_marker": [TS_COL],
    "first_courses": ["course_start", "course_end", "next_course_start"],
    "iv_events_first": [TS_COL, "prev_ts"],
    "course_sp": [TS_COL],
    "iv_cohort": ["result_pre_ts", "course_start", "course_end", "result_post_ts", "set_ts"],
}


def _get_hb_setpoints(hb_full: pd.DataFrame, cohort_hb: pd.DataFrame, cohort_ids: list, *, test_code: str) -> pd.DataFrame:
    """Prefers filtering an already-cached full-population fit over fitting the (much
    smaller) IV-iron cohort alone. Checks run_setpoints_by_marker/fig3_hazard's canonical
    cache first, then fig3_dx's fingerprinted one (see utils/setpoints.py's
    _canonical_cache_name vs. _cache_name_for) -- either one means the full population is
    already fit. Falls back to fitting the cohort itself if neither is cached yet.

    Never forced by this script's own --force (that only rebuilds iv_iron_bundle/, this
    script's own cache) -- the shared setpoint dependency is only ever refreshed by
    run_setpoints_by_marker/fig3_hazard/fig3_dx's own --force.
    """
    if is_fitted_canonical(test_code):
        print(f"[fig5_iron_infusion] full-population {test_code} setpoints already cached -- filtering instead of refitting the cohort")
        full_sp = compute_sp_df(hb_full, test_code=test_code, force=False, canonical=True)
        return full_sp[full_sp[ID_COL].isin(cohort_ids)].reset_index(drop=True)
    if is_fitted(hb_full, test_code):
        print(f"[fig5_iron_infusion] full-population {test_code} setpoints already cached -- filtering instead of refitting the cohort")
        full_sp = compute_sp_df(hb_full, test_code=test_code, force=False)
        return full_sp[full_sp[ID_COL].isin(cohort_ids)].reset_index(drop=True)
    return compute_sp_df(cohort_hb, test_code=test_code, force=False)


def build_iv_iron_bundle(*, input_dir: Path, config: IronInfusionConfig) -> dict:
    tests_df = load_tests_marker_subset(input_dir, test_codes=[config.test_code])
    iron_mar = load_iron_mar_csv(input_dir / IRON_MAR_FILE)

    hb_full = tests_df[tests_df["test_code"] == config.test_code].copy()
    if hb_full.empty:
        raise ValueError(f"No {config.test_code} rows in Tests table; cannot build IV iron cohort.")

    # iron_mar is required to already be filtered to the intended route/formulation
    # (IV, iron sucrose) before it's provided -- see README.md's iron_mar note. No
    # route/desc filtering happens here.
    first_courses, iv_events_first = identify_first_course(iron_mar, gap_between_courses=config.gap_between_courses)

    hb = hb_full.loc[hb_full[ID_COL].isin(first_courses[ID_COL].unique())].copy()

    pre, post = _select_pre_post_labs(
        presenting_df=hb,
        courses_df=first_courses,
        pre_days_max=config.pre_days_max,
        post_days_min=config.post_days_min,
        post_days_max=config.post_days_max,
    )

    cohort_ids = first_courses[ID_COL].unique().tolist()
    cohort_hb = hb.loc[hb[ID_COL].isin(cohort_ids)].copy()

    course_sp = _get_hb_setpoints(hb_full, cohort_hb, cohort_ids, test_code=config.test_code)
    if course_sp.empty:
        raise ValueError("No setpoints computable for the IV iron cohort (too few isolated measurements).")

    course_sp_filtered = get_sp_from_courses(
        sp_df=course_sp,
        courses_df=first_courses,
        lookback_min_days=config.setpoint_lookback_min,
        lookback_max_days=config.setpoint_lookback_max,
        min_setpoint_n=config.min_setpoint_measurements,
    )
    if course_sp_filtered.empty:
        raise ValueError("No setpoints in historical window; cannot attach pre-lab setpoints.")

    iv_cohort = build_iv_cohort(pre[[ID_COL, "result_pre", "result_pre_ts"]], course_sp_filtered, post[[ID_COL, "result_post", "result_post_ts"]])
    if iv_cohort.empty:
        raise ValueError("IV cohort empty after setpoint attachment.")

    iv_cohort = iv_cohort.merge(first_courses[[ID_COL, "doses", "n_courses"]], on=ID_COL, how="left")
    iv_cohort["test_code"] = config.test_code
    iv_cohort = attach_metrics(iv_cohort)

    counts = {
        "iv": n_ids(first_courses),
        "pre": n_ids(pre),
        "post": n_ids(post),
        "pre & post": overlap_ids(pre, post),
        "course_sp_filtered": n_ids(course_sp_filtered),
        "final: pre & course_sp_filtered & post": n_ids(iv_cohort),
    }
    print("[fig5_iron_infusion] Cohort attrition:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    return {
        "hb_marker": hb,
        "first_courses": first_courses,
        "iv_events_first": iv_events_first,
        "course_sp": course_sp,
        "iv_cohort": iv_cohort,
        "counts": counts,
    }


def _bundle_paths(output_dir: Path) -> dict:
    bundle_dir = output_dir / "iv_iron_bundle"
    paths = {name: bundle_dir / f"{name}.csv" for name in BUNDLE_DATE_COLUMNS}
    paths["counts"] = bundle_dir / "counts.json"
    return paths


def _load_or_build_bundle(*, input_dir: Path, output_dir: Path, config: IronInfusionConfig, force: bool) -> dict:
    paths = _bundle_paths(output_dir)

    if not force and all(p.exists() for p in paths.values()):
        print("[fig5_iron_infusion] iv_iron_bundle/ cache hit -- skipping tests.csv/iron_mar.csv read and cohort rebuild")
        bundle = {}
        for name, date_cols in BUNDLE_DATE_COLUMNS.items():
            df = pd.read_csv(paths[name])
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            bundle[name] = df
        bundle["counts"] = json.loads(paths["counts"].read_text())
        return bundle

    bundle = build_iv_iron_bundle(input_dir=input_dir, config=config)

    bundle_dir = output_dir / "iv_iron_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_DATE_COLUMNS:
        bundle[name].to_csv(paths[name], index=False)
    paths["counts"].write_text(json.dumps(bundle["counts"], indent=2, sort_keys=True) + "\n")

    return bundle


def plot_mosaic(bundle: dict, path: Path, *, trajectory_df: pd.DataFrame = None, csv_paths: list = None) -> None:
    """Fig5 mosaic: swimmer timeline, dose-response, trajectory, response distribution, interaction, heatmap."""

    iv_cohort = bundle["iv_cohort"]
    hb_marker = bundle["hb_marker"]
    iv_events_first = bundle["iv_events_first"]
    first_courses = bundle["first_courses"]
    course_sp = bundle["course_sp"]

    mosaic = (("a", "a", "a", "b"), ("c", "d", "e", "f"))
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(A4_WIDTH * 1.1, FIG5_ROW1_HEIGHT + FIG5_ROW2_HEIGHT),
        gridspec_kw={
            "width_ratios": [FIG5_COL1_WIDTH, FIG5_COL2_WIDTH, FIG5_COL3_WIDTH, FIG5_COL4_WIDTH],
            "height_ratios": [FIG5_ROW1_HEIGHT, FIG5_ROW2_HEIGHT],
            "wspace": MOSAIC_WSPACE + 0.1,
            "hspace": MOSAIC_HSPACE,
        },
    )

    try:
        fig5swimmer_on_ax(axd["a"], iv_cohort=iv_cohort, lab_df=hb_marker, infusion_df=iv_events_first)
    except Exception as exc:
        axd["a"].text(0.5, 0.5, f"Swimmer error: {exc}", ha="center", va="center", transform=axd["a"].transAxes)

    fig5dose_response_on_axes(axd["b"], iv_cohort=iv_cohort, outcome_df=iv_events_first)

    panel_c = axd["c"]
    gs_c = panel_c.get_subplotspec().subgridspec(2, 1, hspace=FIG5DOSE_RESPONSE_HSPACE)
    panel_c.remove()
    ax_c_top = fig.add_subplot(gs_c[0, 0])
    ax_c_bottom = fig.add_subplot(gs_c[1, 0])
    traj_final = trajectory_df if trajectory_df is not None else build_iron_infusion_trajectories(hb_marker, first_courses, course_sp, iv_cohort)
    fig5trajectory_on_axes(ax_abs=ax_c_top, ax_drop=ax_c_bottom, df=traj_final)

    panel_d = axd["d"]
    gs_d = panel_d.get_subplotspec().subgridspec(2, 1, hspace=FIG5TRAJECTORY_HSPACE)
    panel_d.remove()
    ax_d_top = fig.add_subplot(gs_d[0, 0])
    ax_d_bottom = fig.add_subplot(gs_d[1, 0])
    fig5response_distribution_on_axes(ax_pres_v_res=ax_d_top, ax_drop_v_res=ax_d_bottom, iv_cohort=iv_cohort, add_colorbar=False, equal_aspect=False)

    fig5interaction_on_ax(axd["e"], cohort_df=iv_cohort, outcome_col="response", x_col="result_pre", n_hb_bins=4)

    panel_f = axd["f"]
    gs_f = panel_f.get_subplotspec().subgridspec(2, 1, height_ratios=[1, 1], hspace=FIG5KM_HSPACE)
    panel_f.remove()
    ax_f_f = fig.add_subplot(gs_f[0, 0])
    ax_f_m = fig.add_subplot(gs_f[1, 0])
    fig5heatmap_on_ax(ax=ax_f_f, df=iv_cohort, x_col="result_pre", setpoint_col=MU, outcome_col="response", sex_value="F", label_y=True, label_x=False)
    fig5heatmap_on_ax(ax=ax_f_m, df=iv_cohort, x_col="result_pre", setpoint_col=MU, outcome_col="response", sex_value="M", label_y=True, label_x=True)

    save_fig_as_svg(fig, "iron_infusion_mosaic", path, csv_path=", ".join(str(p) for p in csv_paths) if csv_paths else None)
    plt.close(fig)


def run(*, input_dir: Path, output_dir: Path, config: IronInfusionConfig = None, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or IronInfusionConfig()

    try:
        bundle = _load_or_build_bundle(input_dir=input_dir, output_dir=output_dir, config=config, force=force)
        iv_cohort = bundle["iv_cohort"]

        cohort_path = output_dir / "iv_iron_cohort.csv"
        iv_cohort.to_csv(cohort_path, index=False)

        trajectory_path = output_dir / "fig5_trajectory_data.csv"
        trajectory_df = build_iron_infusion_trajectories(
            bundle["hb_marker"], bundle["first_courses"], bundle["course_sp"], iv_cohort
        )
        trajectory_df.to_csv(trajectory_path, index=False)

        mosaic_path = output_dir / "iron_infusion_mosaic.svg"
        plot_mosaic(bundle, mosaic_path, trajectory_df=trajectory_df, csv_paths=[cohort_path, trajectory_path])

        manifest = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "test_code": config.test_code,
            "n_patients": int(iv_cohort[ID_COL].nunique()),
            "median_drop": float(iv_cohort["drop"].median()) if "drop" in iv_cohort.columns else None,
            "median_response": float(iv_cohort["response"].median()) if "response" in iv_cohort.columns else None,
            "counts": bundle["counts"],
            "outputs": [cohort_path.name, trajectory_path.name, mosaic_path.name, "manifest.json", "iv_iron_bundle/ (cache, delete to force a full rebuild)"],
        }
    except Exception as exc:
        print(f"[fig5_iron_infusion] SKIPPED, analysis failed: {exc}")
        manifest = {"input_dir": str(input_dir), "output_dir": str(output_dir), "error": str(exc)}

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the iron infusion (Fig5) cohort analysis.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "fig5_iron_infusion")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    with tagged_stdout("fig5_iron_infusion"):
        run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force)
    # print(json.dumps(manifest, indent=2, sort_keys=True))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
