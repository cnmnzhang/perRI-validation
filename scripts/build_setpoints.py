"""Setpoints-by-marker use case: fits every marker in TESTCODES_LIST up front.

Fits TESTCODES_LIST specifically (all 43 pipeline markers), not every marker present in
tests.csv: T4FR, for example, is read directly off the Tests table by fig4_dx_cases's
lab_threshold outcome check and never passed to compute_sp_df, so it's not fit here.

Required inputs: `tests.csv` (anon_id, ts, test_code, result_value, sex) covering all 43
pipeline markers -- see README.md. Reads its markers from the per-marker
split built by `scripts.build_splits_by_marker` -- run that first (or use
`run_all`, which sequences it automatically); raises a clear FileNotFoundError with the
command to run if it hasn't been built yet.

Run:
    python -m scripts.build_setpoints 
    python -m scripts.build_setpoints --input-dir data --output-dir data/outputs/setpoints_by_marker
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import pandas as pd

from utils.bootstrap import ensure_importable

ensure_importable()

from utils.logging_utils import tagged_stdout, timed_step  # noqa: E402
from utils.setpoints import fit_markers  # noqa: E402
from constants.marker_config import TESTCODES_LIST  # noqa: E402
from constants.runtime import DEFAULT_MIN_MEASUREMENTS, ID_COL  # noqa: E402


def build_setpoints(input_dir: Union[str, Path], test_codes: list, force: bool = False) -> pd.DataFrame:
    """Fit every marker in test_codes via fit_markers, with a timed_step banner."""
    with timed_step("setpoints_by_marker", f"Fitting setpoints for {len(test_codes)} marker(s)"):
        return fit_markers(input_dir, test_codes, force=force, label="setpoints_by_marker")


def run(*, input_dir: Path, output_dir: Path, force: bool = False, markers: list[str] = None, min_measurements=DEFAULT_MIN_MEASUREMENTS) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if markers is not None:
        unknown = sorted(set(markers) - set(TESTCODES_LIST))
        if unknown:
            raise ValueError(f"Not in TESTCODES_LIST: {unknown}")
    test_codes = markers if markers else TESTCODES_LIST

    sp_df = build_setpoints(input_dir, test_codes, force=force)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_markers_requested": len(test_codes),
        "n_markers_fitted": int(sp_df["test_code"].nunique()) if not sp_df.empty else 0,
        "n_patients_fitted": int(sp_df[ID_COL].nunique()) if not sp_df.empty else 0,
        "min_measurements": min_measurements,
        "outputs": [f"data/cache/sp_df_<test_code>_full_m{min_measurements}.csv (one per marker, via utils.setpoints.compute_sp_df's own full_population cache)", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit setpoints for every marker in TESTCODES_LIST.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "outputs" / "setpoints_by_marker")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--marker", action="append", dest="markers", help="Fit just this marker (repeatable). Must be in TESTCODES_LIST. Default: all 43.")
    parser.add_argument("--min-measurements", type=int, default=DEFAULT_MIN_MEASUREMENTS, help=f"Minimum measurements per patient to include in fit. Default: {DEFAULT_MIN_MEASUREMENTS}.")
    args = parser.parse_args(argv)

    with tagged_stdout("setpoints_by_marker"):
        manifest = run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force, markers=args.markers, min_measurements=args.min_measurements)
    # print(json.dumps(manifest, indent=2, sort_keys=True))  

if __name__ == "__main__":
    main()
