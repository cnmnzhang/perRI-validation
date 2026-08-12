"""Replicate Fig. 4h multi-marker burden heatmaps from raw repository inputs.

Run from the repository root::

    python -m scripts.run_fig4h_multimarker

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.bootstrap import ensure_importable

ensure_importable()

from constants.fig_config import (  # noqa: E402
    FIG4_HEATMAP_ANNOT_FONTSIZE,
    FIG4_ROW1_HEIGHT,
)
from constants.marker_config import BATTERY2TESTCODE  # noqa: E402
from constants.runtime import (  # noqa: E402
    DIAGNOSIS_TS_COL,
    ICD10_COL,
    ID_COL,
    INDEX_COL,
    MEASUREMENT_COL,
    MODEL_COL,
    MU,
    OUT_PERRI_P95_COL,
    OUT_POPRI_COL,
    SEX_COL,
    SIGMA,
    TEST_CODE_COL,
    TS_COL,
)
from scripts.run_tests_by_marker import build_tests_by_marker  # noqa: E402
from utils.cache import cache_or_compute  # noqa: E402
from utils.clinical.run_clinical import add_oo  # noqa: E402
from utils.io import (  # noqa: E402
    load_demographics_csv,
    load_tests_marker_subset,
)
from utils.logging_utils import tagged_stdout, timed_step  # noqa: E402
from utils.setpoints import (  # noqa: E402
    compute_sp_df,
)
from utils.visuals_shared import add_panel_label, save_fig_as_svg  # noqa: E402


PANELS = {
    "CBC": tuple(BATTERY2TESTCODE["CBC"]),
    "BMP": tuple(BATTERY2TESTCODE["BMP"]),
}

OUTCOME_RANGES = {
    "Neoplasms": ("C", 0, 96),
    "Hematologic": ("D", 50, 89),
    "Metabolic/Endocrine": ("E", 0, 89),
    "Cardiovascular": ("I", 0, 99),
}

PANEL_OUTCOMES = {
    "CBC": ("Neoplasms", "Hematologic"),
    "BMP": ("Metabolic/Endocrine", "Cardiovascular"),
}

BIN_LABELS = {
    "CBC": ("0", "1-3", "4+"),
    "BMP": ("0", "1-2", "3+"),
}

_ICD10_STEM_RE = re.compile(r"^([A-Z])(\d{2})")


def _fifth_setpoints(input_dir: Path, markers: tuple[str, ...]) -> pd.DataFrame:
    """Load/fit each marker and return one index-5 setpoint row per patient/marker."""
    frames = []
    for position, marker in enumerate(markers, 1):
        sp = compute_sp_df(tests_df=None, test_code=marker,
                           canonical=True, force=False)
        fifth = sp[(sp[MODEL_COL] == "bayesian") & (sp[INDEX_COL] == 5)].copy()
        fifth = fifth.sort_values(TS_COL).drop_duplicates([ID_COL, TEST_CODE_COL], keep="first")
        fifth = fifth[[ID_COL, TEST_CODE_COL, TS_COL, MU, SIGMA, SEX_COL]]
        print(
            f"[Fig. 4h] {marker} ({position}/{len(markers)}): "
            f"{fifth[ID_COL].nunique():,} fifth-estimate patients"
        )
        frames.append(fifth)

    if any(frame.empty for frame in frames):
        missing = [marker for marker, frame in zip(markers, frames) if frame.empty]
        raise ValueError(f"No fifth setpoint estimates were available for: {missing}")
    return pd.concat(frames, ignore_index=True)


def _complete_anchor_table(fifth: pd.DataFrame, markers: tuple[str, ...]) -> pd.DataFrame:
    """Patients complete for all markers, anchored at their latest fifth estimate."""
    complete_ids = (
        fifth.groupby(ID_COL)[TEST_CODE_COL]
        .nunique()
        .loc[lambda count: count == len(markers)]
        .index
    )
    complete = fifth[fifth[ID_COL].isin(complete_ids)].copy()
    anchors = (
        complete.groupby(ID_COL, as_index=False)[TS_COL]
        .max()
        .rename(columns={TS_COL: "anchor_ts"})
    )
    return anchors


def _marker_presenting_rows(
    input_dir: Path,
    marker: str,
    anchors: pd.DataFrame,
    washout_days: int,
) -> pd.DataFrame:
    """Candidate post-washout rows, reduced to one marker value per patient/day."""
    tests = load_tests_marker_subset(input_dir, [marker])
    tests = tests[tests[ID_COL].isin(anchors[ID_COL])].copy()
    tests[TS_COL] = pd.to_datetime(tests[TS_COL], errors="coerce")
    tests = tests.merge(anchors, on=ID_COL, how="inner")
    tests = tests[tests[TS_COL] >= tests["anchor_ts"] + pd.Timedelta(days=washout_days)]
    tests["panel_date"] = tests[TS_COL].dt.normalize()
    tests = tests.sort_values([ID_COL, "panel_date", TS_COL])
    tests = tests.drop_duplicates([ID_COL, "panel_date"], keep="first")
    return tests[[ID_COL, "panel_date", TS_COL, MEASUREMENT_COL, SEX_COL]].rename(
        columns={
            TS_COL: f"{marker}_ts",
            MEASUREMENT_COL: f"{marker}_value",
            SEX_COL: f"{marker}_sex",
        }
    )


def _first_complete_presenting_panel(
    input_dir: Path,
    markers: tuple[str, ...],
    anchors: pd.DataFrame,
    washout_days: int,
) -> pd.DataFrame:
    """Inner-join marker days and retain the earliest complete post-washout panel."""
    merged = None
    for marker in markers:
        rows = _marker_presenting_rows(input_dir, marker, anchors, washout_days)
        merged = rows if merged is None else merged.merge(rows, on=[ID_COL, "panel_date"], how="inner")
        if merged.empty:
            break
    if merged is None or merged.empty:
        return pd.DataFrame(columns=[ID_COL, "panel_date"])
    merged = merged.sort_values([ID_COL, "panel_date"]).drop_duplicates(ID_COL, keep="first")
    timestamp_cols = [f"{marker}_ts" for marker in markers]
    merged["presenting_ts"] = merged[timestamp_cols].max(axis=1)
    return merged


def _classify_panel(
    presenting: pd.DataFrame,
    fifth: pd.DataFrame,
    demographics: pd.DataFrame,
    panel: str,
    markers: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use existing add_oo logic and collapse marker flags into panel burden counts."""
    long_frames = []
    for marker in markers:
        marker_rows = presenting[
            [ID_COL, "panel_date", "presenting_ts", f"{marker}_value", f"{marker}_sex"]
        ].rename(
            columns={f"{marker}_value": MEASUREMENT_COL, f"{marker}_sex": SEX_COL}
        )
        marker_rows[TEST_CODE_COL] = marker
        marker_sp = fifth[fifth[TEST_CODE_COL] == marker][[ID_COL, MU, SIGMA]]
        long_frames.append(marker_rows.merge(marker_sp, on=ID_COL, how="inner"))

    classified = pd.concat(long_frames, ignore_index=True)
    demo_sex = demographics[[ID_COL, SEX_COL]].rename(columns={SEX_COL: "demographic_sex"})
    classified = classified.merge(demo_sex, on=ID_COL, how="inner")
    classified[SEX_COL] = classified["demographic_sex"].where(
        classified["demographic_sex"].isin(["F", "M"]), classified[SEX_COL]
    )
    classified = add_oo(classified, result_col=MEASUREMENT_COL, p=0.95)

    burden = (
        classified.groupby(ID_COL, as_index=False)
        .agg(
            panel_date=("panel_date", "first"),
            presenting_ts=("presenting_ts", "first"),
            markers_observed=(TEST_CODE_COL, "nunique"),
            n_out_popri=(OUT_POPRI_COL, "sum"),
            n_out_perri=(OUT_PERRI_P95_COL, "sum"),
        )
    )
    burden = burden[burden["markers_observed"] == len(markers)].copy()
    burden["panel"] = panel
    burden["popri_bin"] = _burden_bins(burden["n_out_popri"], panel)
    burden["perri_bin"] = _burden_bins(burden["n_out_perri"], panel)
    return burden, classified


def _burden_bins(counts: pd.Series, panel: str) -> pd.Categorical:
    labels = BIN_LABELS[panel]
    if panel == "CBC":
        values = np.select([counts == 0, counts <= 3], [labels[0], labels[1]], default=labels[2])
    else:
        values = np.select([counts == 0, counts <= 2], [labels[0], labels[1]], default=labels[2])
    return pd.Categorical(values, categories=labels, ordered=True)


def _normalize_icd10(code: object) -> str | None:
    """Return an ICD-10 three-character (whole-number) stem, or None."""
    if pd.isna(code):
        return None
    compact = re.sub(r"[^A-Z0-9]", "", str(code).strip().upper())
    match = _ICD10_STEM_RE.match(compact)
    return f"{match.group(1)}{match.group(2)}" if match else None


def _outcome_for_stem(stem: object) -> str | None:
    if not isinstance(stem, str) or len(stem) != 3:
        return None
    letter, number_text = stem[0], stem[1:]
    if not number_text.isdigit():
        return None
    number = int(number_text)
    for outcome, (expected_letter, lower, upper) in OUTCOME_RANGES.items():
        if letter == expected_letter and lower <= number <= upper:
            return outcome
    return None


def _load_relevant_dx(
    path: Path,
    patient_ids: set[str],
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Stream Dx and retain only cohort patients and the four Fig. 4h categories.

    The repository's generic ``load_dx_csv`` is ideal for the smaller diagnosis
    analyses, but Fig. 4h only needs four ICD-10 ranges for a small final cohort.
    Parsing every timestamp in a multi-gigabyte Dx table before cohort construction
    is unnecessarily expensive.  This reader keeps the raw-table schema while doing
    patient/category filtering before date conversion.
    """
    header = pd.read_csv(path, nrows=0).columns
    missing = {ID_COL, ICD10_COL, DIAGNOSIS_TS_COL}.difference(header)
    if missing:
        raise ValueError(f"Dx table is missing required column(s): {sorted(missing)}")

    retained = []
    rows_scanned = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=[ID_COL, ICD10_COL, DIAGNOSIS_TS_COL],
            dtype={ID_COL: str, ICD10_COL: str},
            keep_default_na=False,
            na_values=[],
            chunksize=chunksize,
        ),
        1,
    ):
        rows_scanned += len(chunk)
        chunk = chunk[chunk[ID_COL].isin(patient_ids)].copy()
        if not chunk.empty:
            compact = chunk[ICD10_COL].str.strip().str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
            chunk[ICD10_COL] = compact.str.extract(r"^([A-Z]\d{2})", expand=False)
            chunk = chunk[chunk[ICD10_COL].map(_outcome_for_stem).notna()]
            if not chunk.empty:
                chunk[DIAGNOSIS_TS_COL] = pd.to_datetime(chunk[DIAGNOSIS_TS_COL], errors="coerce")
                retained.append(chunk[[ID_COL, ICD10_COL, DIAGNOSIS_TS_COL]].dropna(subset=[DIAGNOSIS_TS_COL]))
        if chunk_number % 10 == 0:
            kept = sum(len(frame) for frame in retained)
            print(f"[Fig. 4h] Dx streaming: {rows_scanned:,} rows scanned, {kept:,} relevant rows retained")

    out = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=[ID_COL, ICD10_COL, DIAGNOSIS_TS_COL])
    print(f"[Fig. 4h] Dx streaming complete: {rows_scanned:,} rows scanned, {len(out):,} relevant rows retained")
    return out


def _attach_incident_outcomes(
    burden: pd.DataFrame,
    dx: pd.DataFrame,
    followup_days: int,
) -> pd.DataFrame:
    """Attach one flag per diagnosis category using incident ICD-10 stems."""
    cohort = burden.copy()
    for outcome in OUTCOME_RANGES:
        cohort[f"event_{outcome}"] = 0
    if cohort.empty:
        return cohort

    coded = dx[[ID_COL, DIAGNOSIS_TS_COL, ICD10_COL]].copy()
    coded["icd10_stem"] = coded[ICD10_COL].map(_normalize_icd10)
    coded["outcome"] = coded["icd10_stem"].map(_outcome_for_stem)
    coded = coded.dropna(subset=[DIAGNOSIS_TS_COL, "icd10_stem", "outcome"])
    joined = coded.merge(
        cohort[[ID_COL, "panel", "presenting_ts"]], on=ID_COL, how="inner"
    )
    joined["presenting_day"] = pd.to_datetime(joined["presenting_ts"]).dt.normalize()
    joined["diagnosis_day"] = pd.to_datetime(joined[DIAGNOSIS_TS_COL]).dt.normalize()

    # Same-day codes are not counted as future events and are conservatively treated as
    # already present; date-only diagnosis tables cannot establish within-day ordering.
    prior = joined[joined["diagnosis_day"] <= joined["presenting_day"]][
        [ID_COL, "panel", "icd10_stem"]
    ].drop_duplicates()
    future = joined[
        (joined["diagnosis_day"] > joined["presenting_day"])
        & (
            joined["diagnosis_day"]
            <= joined["presenting_day"] + pd.Timedelta(days=followup_days)
        )
    ].copy()
    future = future.merge(
        prior.assign(_prevalent_stem=1),
        on=[ID_COL, "panel", "icd10_stem"],
        how="left",
    )
    future = future[future["_prevalent_stem"].isna()]
    events = future[[ID_COL, "panel", "outcome"]].drop_duplicates()
    if events.empty:
        return cohort
    flags = (
        events.assign(value=1)
        .pivot_table(index=[ID_COL, "panel"], columns="outcome", values="value", fill_value=0)
        .reset_index()
    )
    flags = flags.rename(columns={outcome: f"event_{outcome}" for outcome in OUTCOME_RANGES})
    cohort = cohort.drop(columns=[f"event_{outcome}" for outcome in OUTCOME_RANGES])
    cohort = cohort.merge(flags, on=[ID_COL, "panel"], how="left")
    for outcome in OUTCOME_RANGES:
        col = f"event_{outcome}"
        cohort[col] = cohort[col].fillna(0).astype(int) if col in cohort else 0
    return cohort


def _rate_table(cohort: pd.DataFrame) -> pd.DataFrame:
    records = []
    for panel, outcomes in PANEL_OUTCOMES.items():
        panel_df = cohort[cohort["panel"] == panel]
        for outcome in outcomes:
            for perri_bin in BIN_LABELS[panel]:
                for popri_bin in BIN_LABELS[panel]:
                    cell = panel_df[
                        (panel_df["perri_bin"].astype(str) == perri_bin)
                        & (panel_df["popri_bin"].astype(str) == popri_bin)
                    ]
                    n = len(cell)
                    events = int(cell[f"event_{outcome}"].sum()) if n else 0
                    records.append(
                        {
                            "panel": panel,
                            "outcome": outcome,
                            "perri_bin": perri_bin,
                            "popri_bin": popri_bin,
                            "n": n,
                            "events": events,
                            "event_rate_pct": 100.0 * events / n if n else np.nan,
                        }
                    )
    return pd.DataFrame(records)


def _panel_grids(rates: pd.DataFrame, panel: str, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Percent and n matrices for one panel/outcome cell, indexed like `fig4_counts.py`'s
    hardcoded ``*_pct``/``*_n`` tables."""
    sub = rates[(rates["panel"] == panel) & (rates["outcome"] == outcome)]
    labels = BIN_LABELS[panel]
    pct = sub.pivot(index="perri_bin", columns="popri_bin", values="event_rate_pct").reindex(
        index=labels, columns=labels
    )
    n = sub.pivot(index="perri_bin", columns="popri_bin", values="n").reindex(index=labels, columns=labels)
    return pct, n


def _make_annot(pct: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    annot = pct.round(1).astype(str) + "%\nn=" + n.fillna(0).astype(int).astype(str)
    return annot.mask(pct.isna(), "NA\nn=0")


def _plot_heatmap(ax, pct: pd.DataFrame, n: pd.DataFrame, title: str, vmax: float, label_yticks: bool) -> None:
    sns.heatmap(
        pct,
        ax=ax,
        annot=_make_annot(pct, n),
        annot_kws={"fontsize": FIG4_HEATMAP_ANNOT_FONTSIZE},
        fmt="",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_aspect("equal")
    if label_yticks:
        ax.set_yticklabels(pct.index, rotation=0, fontsize=8)
    else:
        ax.set_yticklabels(["" for _ in pct.index], rotation=0, fontsize=8)
    ax.set_xticklabels(pct.columns, rotation=0, fontsize=8)
    ax.set_title(title, fontsize=8, pad=2)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_heatmaps(rates: pd.DataFrame, path: Path) -> None:
    layout = [
        ("CBC", "Neoplasms"),
        ("CBC", "Hematologic"),
        ("BMP", "Metabolic/Endocrine"),
        ("BMP", "Cardiovascular"),
    ]
    grids = {(panel, outcome): _panel_grids(rates, panel, outcome) for panel, outcome in layout}

    fig = plt.figure(figsize=(3.5, FIG4_ROW1_HEIGHT * 2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.4, wspace=0.0)
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ]

    add_panel_label(axes[0][0], "h", x=-0.6, y=1.3)

    for (row, col), (panel, outcome) in zip([(0, 0), (0, 1), (1, 0), (1, 1)], layout):
        pct, n = grids[(panel, outcome)]
        _plot_heatmap(
            axes[row][col],
            pct,
            n,
            outcome,
            vmax=float(np.nanmax(pct.to_numpy())),
            label_yticks=(col == 0),
        )
        axes[row][col].set_xlabel("", fontsize=8)
        axes[row][col].set_ylabel("", fontsize=8)

    axes[0][0].set_ylabel("CBC", fontsize=8, fontweight="bold")
    axes[1][0].set_ylabel("BMP", fontsize=8, fontweight="bold")
    axes[0][0].yaxis.set_label_coords(-0.45, 0.5)
    axes[1][0].yaxis.set_label_coords(-0.45, 0.5)

    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.2)
    fig.text(0.5, 0.02, "Markers outside popRI", ha="center", fontsize=8)
    fig.text(0.12, 0.5, "Markers outside perRI", ha="left", va="center", rotation="vertical", fontsize=8)
    fig.text(0.5, 0.93, "New diagnosis within 1yr (%)", ha="center", va="center", fontsize=10)

    save_fig_as_svg(fig, "Fig. 4h multi-marker burden heatmaps", path)
    plt.close(fig)


def run(
    *,
    input_dir: Path,
    output_dir: Path,
    washout_days: int = 365,
    followup_days: int = 365,
    force: bool = False,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = output_dir / "fig4h_cohort.csv"
    marker_path = output_dir / "fig4h_marker_classifications.csv"
    rates_path = output_dir / "fig4h_rates.csv"
    panel_manifest_path = output_dir / "fig4h_panel_manifest.json"
    figure_path = output_dir / "fig4h_multimarker.svg"

    def _compute_rates() -> pd.DataFrame:
        with timed_step("fig4h", "Preparing the README raw inputs"):
            build_tests_by_marker(input_dir)
            demographics = load_demographics_csv(input_dir / "demographics.csv")

        burden_frames = []
        marker_frames = []
        panel_manifest = {}
        for panel, markers in PANELS.items():
            with timed_step("fig4h", f"Building {panel} presenting-panel cohort"):
                fifth = _fifth_setpoints(input_dir, markers)
                anchors = _complete_anchor_table(fifth, markers)
                presenting = _first_complete_presenting_panel(
                    input_dir, markers, anchors, washout_days
                )
                if presenting.empty:
                    raise ValueError(
                        f"No {panel} patients had a complete presenting panel after the "
                        f"{washout_days}-day washout."
                    )
                burden, marker_rows = _classify_panel(
                    presenting, fifth, demographics, panel, markers
                )
                burden_frames.append(burden)
                marker_rows["panel"] = panel
                marker_frames.append(marker_rows)
                panel_manifest[panel] = {
                    "markers": list(markers),
                    "patients_with_all_fifth_estimates": int(len(anchors)),
                    "patients_with_complete_presenting_panel": int(len(burden)),
                }

        burden = pd.concat(burden_frames, ignore_index=True)
        marker_rows = pd.concat(marker_frames, ignore_index=True)
        with timed_step("fig4h", "Streaming relevant incident diagnoses"):
            dx = _load_relevant_dx(input_dir / "dx.csv", set(burden[ID_COL].astype(str)))
        cohort = _attach_incident_outcomes(burden, dx, followup_days)
        rates = _rate_table(cohort)

        cohort.to_csv(cohort_path, index=False)
        marker_rows.to_csv(marker_path, index=False)
        panel_manifest_path.write_text(json.dumps(panel_manifest, indent=2, sort_keys=True) + "\n")
        return rates

    # A rerun with an already-populated fig4h_rates.csv skips the whole cohort-building pass --
    # per-marker setpoint loading/fitting, the presenting-panel washout joins, and streaming
    # dx.csv for incident diagnoses -- since that's the expensive part once each marker's own
    # compute_sp_df fit is already cached. Like fig3_dx's/fig4_dx_cases's own caches, this checks
    # file existence only (not a content hash of tests.csv/dx.csv/demographics.csv) -- delete
    # fig4h_rates.csv, or pass --force, after any of those change.
    with timed_step("fig4h", "Building multi-marker burden cohort and rates"):
        rates = cache_or_compute(rates_path, _compute_rates, force=force, file_format="csv")

    _plot_heatmaps(rates, figure_path)

    panel_manifest = json.loads(panel_manifest_path.read_text()) if panel_manifest_path.exists() else {}
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "washout_days": washout_days,
        "followup_days": followup_days,
        "panels": panel_manifest,
        "outputs": [
            figure_path.name,
            rates_path.name,
            cohort_path.name,
            marker_path.name,
            "manifest.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"\tRates:\t{rates_path}\n\tCohort:\t{cohort_path}\n\tMarkers:\t{marker_path}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Replicate Fig. 4h multi-marker CBC/BMP burden heatmaps."
    )
    parser.add_argument("--input-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "fig4h_multimarker")
    parser.add_argument("--washout-days", type=int, default=365)
    parser.add_argument("--followup-days", type=int, default=365)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.washout_days < 0 or args.followup_days <= 0:
        parser.error("--washout-days must be >=0 and --followup-days must be >0")
    with tagged_stdout("fig4h_multimarker"):
        run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            washout_days=args.washout_days,
            followup_days=args.followup_days,
            force=args.force,
        )


if __name__ == "__main__":
    main()
