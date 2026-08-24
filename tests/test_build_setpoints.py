"""Tests for scripts/build_setpoints.py.

Not a hard prerequisite the way tests_by_marker/dx_incident are -- compute_sp_df's own
per-marker caching means fig3_hazard/fig3_dx/fig4_dx_cases still fit inline fine without
this having run. These tests cover the script's own run()/manifest, and that it actually
warms the shared compute_sp_df cache other scripts then read from.
"""

import pandas as pd
import pytest

import utils.setpoints as setpoints_module
from scripts.build_setpoints import run
from scripts.build_splits_by_marker import build_splits_by_marker
from utils.setpoints import is_fitted_full_population


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "sp_cache")


def _write_isolated_tests_csv(input_dir, test_codes):
    dates = pd.date_range("2015-01-01", periods=8, freq="120D")
    rows = [{"anon_id": "p1", "ts": d, "test_code": tc, "result_value": 13.0, "sex": "F"} for d in dates for tc in test_codes]
    pd.DataFrame(rows).to_csv(input_dir / "tests.csv", index=False)


def test_run_fits_requested_markers_and_warms_shared_cache(tmp_path, monkeypatch):
    import scripts.build_setpoints as m

    monkeypatch.setattr(m, "TESTCODES_LIST", ["HB", "GLU"])

    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    _write_isolated_tests_csv(input_dir, ["HB", "GLU"])
    build_splits_by_marker(input_dir)

    manifest = run(input_dir=input_dir, output_dir=output_dir, force=True)

    assert manifest["n_markers_requested"] == 2
    assert manifest["n_markers_fitted"] == 2
    assert (output_dir / "manifest.json").exists()

    # The whole point: compute_sp_df's full_population cache is now warm for a later,
    # independent caller (e.g. fig5_iron_infusion) to reuse without loading tests_df.
    assert is_fitted_full_population("HB") is True
    assert is_fitted_full_population("GLU") is True


def test_marker_arg_restricts_to_just_that_marker(tmp_path, monkeypatch):
    """--marker must fit only the requested marker, leaving every other TESTCODES_LIST
    marker's setpoint cache untouched -- e.g. to refresh just NA after a loader fix,
    without re-fitting all 43 markers."""
    import scripts.build_setpoints as m

    monkeypatch.setattr(m, "TESTCODES_LIST", ["HB", "GLU"])

    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    _write_isolated_tests_csv(input_dir, ["HB", "GLU"])
    build_splits_by_marker(input_dir)

    manifest = run(input_dir=input_dir, output_dir=output_dir, force=True, markers=["HB"])

    assert manifest["n_markers_requested"] == 1
    assert manifest["n_markers_fitted"] == 1
    assert is_fitted_full_population("HB") is True
    assert is_fitted_full_population("GLU") is False  # never requested, never fit


def test_marker_arg_rejects_marker_not_in_testcodes_list(tmp_path):
    with pytest.raises(ValueError):
        run(input_dir=tmp_path / "data", output_dir=tmp_path / "out", markers=["NOT_A_REAL_MARKER"])
