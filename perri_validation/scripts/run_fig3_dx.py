"""Fig3 dx use case: fig3's non-cancer Kaplan-Meier panel.

Reads the derived Dx table produced by `perri_validation.scripts.run_dx_incident`
(`dx_incident.csv`) rather than re-deriving it -- run dx_incident first. This
script will raise a clear error naming the expected path if that file isn't
there yet -- unless `fig3_km_data.csv` is already cached in `output_dir` (see
below), in which case dx_incident.csv is never even read.

For each of DX2SETPOINT's 8 non-cancer diagnoses, patients at risk
(not yet diagnosed as of their setpoint estimate) are split by whether their
personal setpoint (mu) falls above/below a sex-specific 25th/75th population
percentile cutoff, and Kaplan-Meier survival curves are fit per group, using
perri_validation/utils/clinical/run_clinical.py's precompute_cutoffs/get_base_population/
get_at_risk_population/get_groups_from_config/compute_event_time/_fit_km_for_groups,
with setpoints computed live via utils.setpoints.compute_sp_df.

Required inputs: `tests.csv` (anon_id, ts, test_code, result_value, sex) covering
the 8 fig3-KM markers (HB, TNEUT, ALB, ALT, MCV, P, GLU, K), and a Demographics
table (anon_id, sex, birth_date, death_ts) -- see perri_validation/README.md. Reads
its markers from the per-marker split built by
`perri_validation.scripts.run_tests_by_marker` -- run that first (or use `run_all`,
which sequences it automatically); raises a clear FileNotFoundError with the command
to run if it hasn't been built yet -- unless `fig3_km_data.csv` is already cached
(see below), in which case the split is never even read.

Run:
    python -m perri_validation.scripts.run_fig3_dx --input-dir perri_validation/data --output-dir perri_validation/outputs/fig3_dx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perri_validation.utils.bootstrap import ensure_importable

ensure_importable()

import matplotlib.pyplot as plt  # noqa: E402

from perri_validation.utils.cache import cache_or_compute  # noqa: E402
from perri_validation.utils.io import load_demographics_csv, load_dx_incident  # noqa: E402
from perri_validation.utils.logging_utils import tagged_stdout, timed_step  # noqa: E402
from perri_validation.utils.setpoints import fit_markers_lazy  # noqa: E402
from perri_validation.utils.clinical.get import attach_ref_intervals, compute_within_normal_mask  # noqa: E402
from perri_validation.utils.clinical.run_clinical import (  # noqa: E402
    _fit_km_for_groups,
    compute_event_time,
    expand_dx2setpoint,
    get_at_risk_population,
    get_base_population,
    get_groups_from_config,
    get_one_setpoint,
    precompute_cutoffs,
)
from perri_validation.constants.icd_config import COMBINED_ORDER, DX2SETPOINT  # noqa: E402
from perri_validation.constants.runtime import ID_COL, MAX_FIT_DATE, TS_COL  # noqa: E402
from perri_validation.utils.visuals_fig3 import fig3km  # noqa: E402

DEMOGRAPHICS_FILE = "demographics.csv"

FIG3_KM_MARKERS = sorted({cfg["setpoint_type"] for configs in DX2SETPOINT.values() for cfg in configs})
OBSERVATION_PERIOD_START = "01-01-2014"  # start of fig3's non-cancer KM observation window
OBSERVATION_WINDOW_YEARS = 6.0
MIN_ISOLATED = 5
USE_PERSONALIZED_LOGIC = True

if set(COMBINED_ORDER) != set(DX2SETPOINT.keys()):
    raise ValueError(f"COMBINED_ORDER and DX2SETPOINT have drifted out of sync: " f"COMBINED_ORDER-only={set(COMBINED_ORDER) - set(DX2SETPOINT)}, DX2SETPOINT-only={set(DX2SETPOINT) - set(COMBINED_ORDER)}")
# The KM panel's diagnoses are processed (and therefore plotted, left-to-right/top-to-bottom
# in fig3km's facet grid) in COMBINED_ORDER, not DX2SETPOINT's raw dict order -- the two
# happen to match today, but only COMBINED_ORDER is the actual intended display order.
DX2SETPOINT_ORDERED = {name: DX2SETPOINT[name] for name in COMBINED_ORDER}


def build_fig3_km_data(*, input_dir: Path, dx_incident: pd.DataFrame) -> pd.DataFrame:
    demog_df = load_demographics_csv(input_dir / DEMOGRAPHICS_FILE)
    demog_df = demog_df.dropna(subset=["birth_date"]).copy()
    demog_df["death_ts_filled"] = demog_df["death_ts"].fillna(pd.Timestamp.today())

    # force (this script's own fig3_km_data.csv cache) never cascades to the shared setpoint
    # dependency -- that's run_setpoints_by_marker's job, not this script's.
    with timed_step("fit_setpoints", f"Fitting setpoints for {len(FIG3_KM_MARKERS)} markers"):
        sp_df = fit_markers_lazy(input_dir, FIG3_KM_MARKERS, force=False, label="fig3_dx")

    sp_df_demog = sp_df.merge(demog_df[[ID_COL, "birth_date", "death_ts_filled"]].drop_duplicates(), on=ID_COL, how="inner")
    n_before = len(sp_df_demog)
    sp_df_demog = sp_df_demog[(sp_df_demog["death_ts_filled"].isnull()) | (sp_df_demog["death_ts_filled"] >= sp_df_demog[TS_COL])]
    print(f"fig3_dx: removed {n_before - len(sp_df_demog):,} rows where the patient died before {TS_COL}")

    filtered_setpoints_df = get_one_setpoint(
        sp_df_demog,
        use_personalized_logic=USE_PERSONALIZED_LOGIC,
        model="bayesian",
        min_isolated=MIN_ISOLATED,
        min_dts=OBSERVATION_PERIOD_START,
        max_dts=MAX_FIT_DATE,
    )
    filtered_setpoints_df = attach_ref_intervals(filtered_setpoints_df)
    filtered_setpoints_df = filtered_setpoints_df[compute_within_normal_mask(filtered_setpoints_df)].copy()

    precomputed_cutoffs_df = precompute_cutoffs(filtered_setpoints_df)
    population_base, _ = get_base_population(filtered_setpoints_df, precomputed_cutoffs_df)

    dx2setpoint_expanded = expand_dx2setpoint(DX2SETPOINT_ORDERED)
    km_frames = []
    for i, (dx_and_setpoint, criteria) in enumerate(dx2setpoint_expanded.items(), 1):
        dx_name = dx_and_setpoint.split(" (")[0]
        test_code = criteria["setpoint_type"]
        print(f"\nfig3_dx: [{i}/{len(dx2setpoint_expanded)}] --- {dx_and_setpoint} ---")

        try:
            df_at_risk = get_at_risk_population(
                population_base=population_base,
                one_dx_incident=dx_incident[dx_incident["diagnosis_name"] == dx_name],
                test_code=test_code,
                observation_period_start=OBSERVATION_PERIOD_START,
                use_personalized_logic=USE_PERSONALIZED_LOGIC,
                filtered_setpoints_df=filtered_setpoints_df,
                verbose=True,
            )
            n_dxd = int(df_at_risk["earliest_contact_date"].notna().sum())
            print(f"fig3_dx: {n_dxd} diagnosed patients at risk")
            if n_dxd < 2:
                print(f"fig3_dx: not enough patients at risk ({n_dxd}) for '{dx_name}', skipping")
                continue

            df_at_risk = df_at_risk.merge(filtered_setpoints_df[[ID_COL, "death_ts_filled"]].drop_duplicates(), on=ID_COL, how="left")

            groups = get_groups_from_config(df_at_risk, criteria, precomputed_cutoffs_df)
            event_time, event_observed = compute_event_time(df_at_risk, observation_window=OBSERVATION_WINDOW_YEARS)
            fitted = _fit_km_for_groups(groups, event_time, event_observed, dx_name, criteria["pct_cutoff"], test_code)
        except Exception as exc:
            print(f"fig3_dx: [{i}/{len(dx2setpoint_expanded)}] {dx_and_setpoint}: SKIPPED, subplot failed: {exc}")
            continue
        if not fitted.empty:
            km_frames.append(fitted)

    return pd.concat(km_frames, ignore_index=True) if km_frames else pd.DataFrame()


def run(*, input_dir: Path, output_dir: Path, dx_incident_path: Path = None, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dx_incident_path is None:
        dx_incident_path = Path(__file__).resolve().parent.parent / "outputs" / "dx_incident" / "dx_incident.csv"

    def _compute_km_data() -> pd.DataFrame:
        dx_incident = load_dx_incident(dx_incident_path)
        return build_fig3_km_data(input_dir=input_dir, dx_incident=dx_incident)

    # A rerun with an already-populated fig3_km_data.csv skips the whole diagnosis-anchoring +
    # KM-fitting pass (get_at_risk_population/get_groups_from_config/compute_event_time/
    # _fit_km_for_groups across all 8 diagnoses, over hundreds of thousands of setpoint rows)
    # -- that loop is the expensive part once compute_sp_df's own per-marker fits are already
    # cached, so caching only sp_df and redoing this every run was still slow on a full rerun.
    # Like fig4_dx_cases's cohort cache, this checks file existence only (not a content hash of
    # tests.csv/dx_incident.csv) -- delete fig3_km_data.csv, or pass --force, after either changes.
    with timed_step("fig3_km", "Building fig3's 8-diagnosis KM panel"):
        km_data = cache_or_compute(output_dir / "fig3_km_data.csv", _compute_km_data, force=force, file_format="csv")

    n_facets = 0
    if not km_data.empty:
        fig = fig3km(km_data, observation_window=OBSERVATION_WINDOW_YEARS, save_path=output_dir / "fig3_km.svg", csv_path=output_dir / "fig3_km_data.csv")
        n_facets = int(km_data[["diagnosis", "setpoint_type"]].drop_duplicates().shape[0])
        plt.close(fig)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dx_incident_path": str(dx_incident_path),
        "fig3_km_markers": FIG3_KM_MARKERS,
        "fig3_km_n_facets": n_facets,
        "outputs": ["fig3_km_data.csv", "fig3_km.svg", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run fig3's dx-anchored Kaplan-Meier panel.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "fig3_dx")
    parser.add_argument("--dx_incident_path", type=Path, default=None, help="Path to dx_incident.csv from `perri_validation.scripts.run_dx_incident`. Default: perri_validation/outputs/dx_incident/dx_incident.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    with tagged_stdout("fig3_dx"):
        run(input_dir=args.input_dir, output_dir=args.output_dir, dx_incident_path=args.dx_incident_path, force=args.force)
    # print(json.dumps(manifest, indent=2, sort_keys=True))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
