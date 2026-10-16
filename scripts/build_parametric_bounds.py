"""Build site-calibrated Bayesian grid bounds from complete, already-isolated marker files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from constants.marker_lab_config import MARKER_CONFIG
from constants.runtime import SEX_COL
from utils.io import load_tests_marker_subset, splits_by_marker_dir
from utils.setpoints import is_sex_stratified
from utils.site_bounds import get_site_bounds

EMPIRICAL_MARKERS = tuple(
    code for code, cfg in MARKER_CONFIG.items()
    if cfg.get("mu_bound_policy", {}).get("method", "clinical_ri") != "clinical_ri"
)


def run(*, input_dir: Path, output_dir: Path, force: bool = False) -> dict:
    """Calibrate empirical-policy markers from each complete site marker file."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    marker_dir = splits_by_marker_dir(input_dir)
    if not marker_dir.is_dir():
        raise FileNotFoundError(f"Expected complete marker files in {marker_dir}")

    built = {}
    missing = []
    for code in EMPIRICAL_MARKERS:
        if not (marker_dir / f"{code}.csv").exists():
            missing.append(code)
            continue
        marker_df = load_tests_marker_subset(input_dir, test_codes=[code])
        sexes = ("F", "M") if is_sex_stratified(code) else ("ALL",)
        for sex in sexes:
            if sex != "ALL" and not (marker_df[SEX_COL] == sex).any():
                continue
            result = get_site_bounds(code, sex, input_dir=input_dir, full_marker_df=marker_df, force=force)
            built[f"{code}|{sex}"] = result
            print(f"{code}/{sex}: {result['n_measurements']:,} isolated measurements; bounds={result['bounds']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"input_dir": str(input_dir), "built": built, "missing_marker_files": missing}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "outputs" / "build_parametric_bounds")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
