"""Tests-by-marker use case: splits the master Tests table into one file per marker.

Prerequisite for fig3_dx, fig3_hazard, fig4_dx_cases, and fig5_iron_infusion -- each
reads its markers via utils.io.load_tests_marker_subset, which requires the per-marker
split to already exist at data/cache/splits_by_marker/{test_code}.csv (raises a clear
error naming this command otherwise, the same way fig3_dx requires dx_incident first).
Splitting once, up front, means the (potentially multi-GB) master `tests.csv`/
`tests.csv.gz` is read in full exactly once total, not once per analysis script that
happens to touch it first.

Not the only way to populate that split, though: a site whose data is already
partitioned by marker can skip this script entirely and drop per-marker CSVs (each
matching TESTS_SCHEMA, like the master Tests table) straight into
data/cache/splits_by_marker/ themselves -- load_tests_marker_subset doesn't require
this script's own `_split_complete.json` sentinel, just the files it names.

Required inputs: `tests.csv` or `tests.csv.gz` -- see README.md.

Run:
    python -m scripts.build_splits_by_marker --input-dir data --output-dir data/outputs/splits_by_marker
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import pandas as pd

from utils.bootstrap import ensure_importable

ensure_importable()

from constants.runtime import N_JOBS, TEST_CODE_COL  # noqa: E402
from utils.io import load_tests_csv, resolve_tests_csv_path, splits_by_marker_dir  # noqa: E402
from utils.logging_utils import tagged_stdout, timed_step  # noqa: E402


def _write_marker_csv(test_code: str, group: pd.DataFrame, marker_dir: Path) -> tuple:
    t0 = time.time()
    group.to_csv(marker_dir / f"{test_code}.csv", index=False)
    return test_code, len(group), time.time() - t0


def _write_marker_csvs_parallel(groups: list, marker_dir: Path) -> dict:
    """Write independent per-marker CSVs concurrently using I/O-bound threads."""
    manifest = {}
    n_markers_total = len(groups)
    with ThreadPoolExecutor(max_workers=min(N_JOBS, n_markers_total)) as pool:
        futures = {pool.submit(_write_marker_csv, test_code, group, marker_dir): test_code for test_code, group in groups}
        for i, future in enumerate(as_completed(futures), 1):
            test_code, n_rows, elapsed = future.result()
            manifest[test_code] = n_rows
            print(f"[{i}/{n_markers_total}] {test_code}: {n_rows:,} rows written ({elapsed:.1f}s)")
    return manifest


def build_splits_by_marker(input_dir: Union[str, Path], force: bool = False, markers: list = None) -> dict:
    """Split the master Tests table into one CSV per marker.

    The full split is existence-cached by `_split_complete.json`. Passing `markers`
    refreshes only those marker files and requires a completed full split.
    """
    marker_dir = splits_by_marker_dir(input_dir)
    sentinel_path = marker_dir / "_split_complete.json"

    if markers is not None:
        if not sentinel_path.exists():
            raise FileNotFoundError(f"Expected the full split to already exist at {marker_dir} before refreshing individual markers {markers} -- run the full split first: python -m scripts.build_splits_by_marker --input-dir {input_dir}")
        print(f"splits_by_marker/ refreshing {len(markers)} marker(s): {', '.join(markers)}...")
        tests_df = load_tests_csv(resolve_tests_csv_path(input_dir))
        groups = [(test_code, tests_df[tests_df[TEST_CODE_COL] == test_code]) for test_code in markers]
        manifest = json.loads(sentinel_path.read_text())
        manifest.update(_write_marker_csvs_parallel(groups, marker_dir))
        sentinel_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"splits_by_marker/ refresh complete: {len(markers)} marker(s)")
        return manifest

    if sentinel_path.exists() and not force:
        return json.loads(sentinel_path.read_text())

    print("splits_by_marker/ cache miss -- splitting the master Tests table into one file per marker (one-time cost, shared by every analysis)...")
    tests_df = load_tests_csv(resolve_tests_csv_path(input_dir))
    marker_dir.mkdir(parents=True, exist_ok=True)
    manifest = _write_marker_csvs_parallel(list(tests_df.groupby(TEST_CODE_COL)), marker_dir)
    sentinel_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"splits_by_marker/ split complete: {len(manifest)} markers")
    return manifest


def run(*, input_dir: Path, output_dir: Path, force: bool = False, markers: list[str] = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    step_label = f"Refreshing {len(markers)} marker(s)" if markers else "Splitting the master Tests table into one file per marker"
    with timed_step("splits_by_marker", step_label):
        marker_manifest = build_splits_by_marker(input_dir, force=force, markers=markers)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_markers": len(marker_manifest),
        "n_rows": sum(marker_manifest.values()),
        "markers": marker_manifest,
        "outputs": ["data/cache/splits_by_marker/<test_code>.csv (one per marker)", "data/cache/splits_by_marker/_split_complete.json", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Split the master Tests table into one file per marker.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "outputs" / "splits_by_marker")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--marker", action="append", dest="markers", help="Refresh just this marker's split file, leaving every other marker's file untouched (repeatable). Requires the full split to already exist. Default: full split.")
    args = parser.parse_args(argv)

    with tagged_stdout("splits_by_marker"):
        manifest = run(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force, markers=args.markers)
    # print(json.dumps(manifest, indent=2, sort_keys=True))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
