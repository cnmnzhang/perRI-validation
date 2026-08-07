"""Setpoints-by-marker use case: fits every marker in TESTCODES_LIST up front.

Not a hard prerequisite for the other analyses the way tests_by_marker/dx_incident are --
utils.setpoints.compute_sp_df already caches per (marker, exact population, min_measurements),
so fig3_hazard/fig3_dx/fig4_dx_cases still fit inline just fine if this hasn't been run. What
this buys you is a dedicated, standalone place to run the expensive part (a Bayesian fit per
patient, per marker, across all 43 markers -- the same ~2.5-3 hour job fig3_hazard does as
part of its own run) independently of any one analysis's other work (Cox regression, plotting,
...), so you can warm the cache once and have every full-population-fitting analysis
afterward -- including fig5_iron_infusion, which prefers filtering an already-cached
full-population HB fit over its own smaller cohort-only fit when one exists -- read straight
from cache.

Fits TESTCODES_LIST specifically (all 43 pipeline markers), not every marker present in
tests.csv: T4FR, for example, is read directly off the Tests table by fig4_dx_cases's
lab_threshold outcome check and never passed to compute_sp_df, so it's not fit here.

Required inputs: `tests.csv` (anon_id, ts, test_code, result_value, sex) covering all 43
pipeline markers -- see README.md. Reads its markers from the per-marker
split built by `scripts.run_tests_by_marker` -- run that first (or use
`run_all`, which sequences it automatically); raises a clear FileNotFoundError with the
command to run if it hasn't been built yet.

Run:
    python -m scripts.run_setpoints_by_marker --input-dir data --output-dir outputs/setpoints_by_marker
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.bootstrap import ensure_importable

ensure_importable()

from utils.logging_utils import tagged_stdout, timed_step  # noqa: E402
from utils.setpoints import fit_markers_lazy  # noqa: E402
from constants.marker_config import TESTCODES_LIST  # noqa: E402
from constants.runtime import DEFAULT_MIN_MEASUREMENTS, ID_COL  # noqa: E402


def run(*, input_dir: Path, output_dir: Path, force: bool = False, markers: list[str] = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if markers is not None:
        unknown = sorted(set(markers) - set(TESTCODES_LIST))
        if unknown:
            raise ValueError(f"Not in TESTCODES_LIST: {unknown}")
    test_codes = markers if markers else TESTCODES_LIST

    with timed_step("setpoints_by_marker", f"Fitting setpoints for {len(test_codes)} marker(s)"):
        sp_df = fit_markers_lazy(input_dir, test_codes, force=force, label="setpoints_by_marker")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_markers_requested": len(test_codes),
        "n_markers_fitted": int(sp_df["test_code"].nunique()) if not sp_df.empty else 0,
        "n_patients_fitted": int(sp_df[ID_COL].nunique()) if not sp_df.empty else 0,
        "outputs": [f"data/cache/sp_df_<test_code>_full_m{DEFAULT_MIN_MEASUREMENTS}.csv (one per marker, via utils.setpoints.compute_sp_df's own canonical cache)", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit setpoints for every marker in TESTCODES_LIST.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "setpoints_by_marker")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--marker", action="append", dest="markers", help="Fit just this marker (repeatable). Must be in TESTCODES_LIST. Default: all 43.")
    args = parser.parse_args(argv)

    with tagged_stdout("setpoints_by_marker"):
        manifest = run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force, markers=args.markers)
    # print(json.dumps(manifest, indent=2, sort_keys=True))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
