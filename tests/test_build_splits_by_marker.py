"""Tests for scripts/build_splits_by_marker.py.

tests_by_marker is now an explicit prerequisite step (like build_dx_incident is for
dx_incident.csv), not built implicitly by whichever analysis script happens to run
first -- these tests cover the script's own run()/manifest, and that downstream
loaders correctly see what it produced.
"""

import pandas as pd
import pytest

from scripts.build_splits_by_marker import run
from utils.io import load_tests_marker_subset
from utils.io import tests_by_marker_dir as get_marker_dir


def test_run_builds_split_and_writes_manifest(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    rows = [{"anon_id": f"p{i}", "ts": "2015-01-01", "test_code": tc, "result_value": 50.0, "sex": "F"} for i in range(3) for tc in ("HB", "GLU")]
    pd.DataFrame(rows).to_csv(input_dir / "tests.csv", index=False)

    manifest = run(input_dir=input_dir, output_dir=output_dir)

    assert manifest["n_markers"] == 2
    assert manifest["n_rows"] == 6
    assert manifest["markers"] == {"HB": 3, "GLU": 3}
    assert (output_dir / "manifest.json").exists()

    subset = load_tests_marker_subset(input_dir, test_codes=["HB"])
    assert (subset["test_code"] == "HB").all()


def test_marker_arg_refreshes_only_that_marker(tmp_path):
    """--marker must rewrite only the requested marker's file, leaving every other
    already-split marker's file untouched -- e.g. after a loader fix changes how one
    marker's rows are parsed, there's no need to re-split everything to pick it up."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    rows = [{"anon_id": f"p{i}", "ts": "2015-01-01", "test_code": tc, "result_value": 50.0, "sex": "F"} for i in range(3) for tc in ("HB", "GLU")]
    pd.DataFrame(rows).to_csv(input_dir / "tests.csv", index=False)
    run(input_dir=input_dir, output_dir=output_dir)

    glu_path = get_marker_dir(input_dir) / "GLU.csv"
    glu_mtime_before = glu_path.stat().st_mtime

    # tests.csv changes underneath (extra HB row) -- only HB should pick it up.
    rows.append({"anon_id": "p_new", "ts": "2015-01-01", "test_code": "HB", "result_value": 50.0, "sex": "F"})
    pd.DataFrame(rows).to_csv(input_dir / "tests.csv", index=False)

    manifest = run(input_dir=input_dir, output_dir=output_dir, markers=["HB"])

    assert manifest["markers"]["HB"] == 4
    assert manifest["markers"]["GLU"] == 3  # untouched entry, carried over from the prior manifest
    assert glu_path.stat().st_mtime == glu_mtime_before  # GLU.csv itself was never rewritten

    subset = load_tests_marker_subset(input_dir, test_codes=["HB"])
    assert len(subset) == 4


def test_marker_arg_requires_existing_full_split(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    pd.DataFrame({"anon_id": ["p1"], "ts": ["2015-01-01"], "test_code": ["HB"], "result_value": [50.0], "sex": ["F"]}).to_csv(input_dir / "tests.csv", index=False)

    with pytest.raises(FileNotFoundError):
        run(input_dir=input_dir, output_dir=output_dir, markers=["HB"])
