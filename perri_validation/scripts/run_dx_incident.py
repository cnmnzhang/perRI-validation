"""Dx incident use case: derived incident-diagnosis table.

Collapses row-level ICD9/ICD10 diagnosis events (one row per diagnosis per
visit) into dx_incident.csv -- one row per (patient, diagnosis) with the
earliest ("incident") date that diagnosis appears. fig3_dx and fig4_dx_cases
both read this output rather than re-deriving it (run dx_incident first:
python -m perri_validation.scripts.run_dx_incident).

Required inputs: a Dx table (anon_id, icd9, icd10, date) -- see
perri_validation/README.md.

Run:
    python -m perri_validation.scripts.run_dx_incident --input-dir perri_validation/data --output-dir perri_validation/outputs/dx_incident
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perri_validation.utils.bootstrap import ensure_importable

ensure_importable()

from perri_validation.utils.io import load_dx_csv  # noqa: E402
from perri_validation.utils.logging_utils import tagged_stdout, timed_step  # noqa: E402
from perri_validation.utils.cache import cache_or_compute  # noqa: E402
from perri_validation.utils.clinical.icd import dx_all_to_first_fast  # noqa: E402
from perri_validation.constants.runtime import ID_COL  # noqa: E402

DX_FILE = "dx.csv"


def build_dx_incident(*, input_dir: Path, output_dir: Path, force: bool) -> pd.DataFrame:
    out_path = output_dir / "dx_incident.csv"

    def _compute():
        dx_all = load_dx_csv(input_dir / DX_FILE)
        return dx_all_to_first_fast(dx_all, id_col=ID_COL)

    with timed_step("dx_incident", "Deriving incident dx_incident.csv from all diagnosis events"):
        dx_incident = cache_or_compute(out_path, _compute, force=force, file_format="csv")
    dx_incident["earliest_contact_date"] = pd.to_datetime(dx_incident["earliest_contact_date"])
    return dx_incident

    
def run(*, input_dir: Path, output_dir: Path, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    dx_incident = build_dx_incident(input_dir=input_dir, output_dir=output_dir, force=force)
    summary = dx_incident["diagnosis_name"].value_counts().rename_axis("diagnosis_name").reset_index(name="n_patients")
    summary.to_csv(output_dir / "dx_incident_summary.csv", index=False)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_diagnosis_rows": int(len(dx_incident)),
        "n_patients": int(dx_incident[ID_COL].nunique()),
        "n_diagnosis_names": int(dx_incident["diagnosis_name"].nunique()),
        "outputs": ["dx_incident.csv", "dx_incident_summary.csv", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the dx_incident use case (derived incident-diagnosis table).")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "dx_incident")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    with tagged_stdout("dx_incident"):
        manifest = run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force)
    # print(json.dumps(manifest, indent=2, sort_keys=True))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
