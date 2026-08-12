"""Tests for utils/setpoints.py:compute_sp_df.

Every test monkeypatches CACHE_DIR to tmp_path -- these must never touch the
real, shared data/cache/.
"""

import numpy as np
import pandas as pd
import pytest

import utils.setpoints as setpoints_module
from constants.runtime import DEFAULT_MIN_MEASUREMENTS
from scripts.run_tests_by_marker import build_tests_by_marker
from utils.setpoints import SP_DF_COLUMNS, _compute_popri_patch, _grid_bounds_from_pop_ri, _params_override, compute_sp_df, fit_markers, fit_markers_lazy, is_fitted, is_fitted_canonical


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "cache")


def _isolated_series(anon_id, sex, n=8, start="2015-01-01", value=13.0, gap_days=120):
    dates = pd.date_range(start, periods=n, freq=f"{gap_days}D")
    return pd.DataFrame(
        {
            "anon_id": [anon_id] * n,
            "ts": dates,
            "test_code": ["HB"] * n,
            "result_value": [value] * n,
            "sex": [sex] * n,
        }
    )


def test_returns_expected_columns_and_fits_isolated_patient():
    tests_df = _isolated_series("p1", "F")
    sp_df = compute_sp_df(tests_df, test_code="HB", force=True)

    assert list(sp_df.columns) == SP_DF_COLUMNS
    assert set(sp_df["anon_id"]) == {"p1"}
    assert (sp_df["test_code"] == "HB").all()
    assert (sp_df["model"] == "bayesian").all()
    assert (sp_df["sex"] == "F").all()


def test_empty_when_test_code_not_present():
    tests_df = _isolated_series("p1", "F")
    sp_df = compute_sp_df(tests_df, test_code="GLU", force=True)
    assert sp_df.empty
    assert list(sp_df.columns) == SP_DF_COLUMNS


def test_below_min_measurements_is_excluded():
    tests_df = _isolated_series("p1", "F", n=DEFAULT_MIN_MEASUREMENTS - 1)
    sp_df = compute_sp_df(tests_df, test_code="HB", force=True)
    assert sp_df.empty


def test_densely_spaced_measurements_are_not_isolated():
    """fit_batch's isolation filter requires >=90-day gaps by default -- monthly
    spacing collapses below min_measurements, exactly the synthetic-fixture bug
    class hit earlier this project."""
    tests_df = _isolated_series("p1", "F", n=8, gap_days=30)
    sp_df = compute_sp_df(tests_df, test_code="HB", force=True)
    assert sp_df.empty


def test_different_populations_get_different_cache_files(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", cache_dir)

    tests_df_1 = _isolated_series("p1", "F", value=13.0)
    tests_df_2 = _isolated_series("p2", "F", value=9.0)

    compute_sp_df(tests_df_1, test_code="HB", force=True)
    compute_sp_df(tests_df_2, test_code="HB", force=True)

    cache_files = sorted(p.name for p in cache_dir.glob("sp_df_HB_*.csv"))
    assert len(cache_files) == 2, f"expected two distinct cache files, got {cache_files}"


def test_same_population_reuses_cache_without_refitting(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", cache_dir)

    tests_df = _isolated_series("p1", "F")

    compute_sp_df(tests_df, test_code="HB", force=True)
    n_files_after_first = len(list(cache_dir.glob("sp_df_HB_*.csv")))

    compute_sp_df(tests_df, test_code="HB", force=False)
    n_files_after_second = len(list(cache_dir.glob("sp_df_HB_*.csv")))

    assert n_files_after_first == n_files_after_second == 1


def test_min_measurements_is_part_of_the_cache_key(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", cache_dir)

    tests_df = _isolated_series("p1", "F")

    compute_sp_df(tests_df, test_code="HB", min_measurements=5, force=True)
    compute_sp_df(tests_df, test_code="HB", min_measurements=3, force=True)

    cache_files = sorted(p.name for p in cache_dir.glob("sp_df_HB_*.csv"))
    assert len(cache_files) == 2
    assert any("_m5.csv" in f for f in cache_files)
    assert any("_m3.csv" in f for f in cache_files)


def _multi_patient_isolated_series(test_code, values_by_patient, n=8, gap_days=120):
    frames = [_isolated_series(pid, "F", n=n, value=value, gap_days=gap_days).assign(test_code=test_code) for pid, value in values_by_patient.items()]
    return pd.concat(frames, ignore_index=True)


def test_popri_patch_computed_from_isolated_measurements_for_one_sided_marker():
    """HDL's pop_ri has an infinite upper bound (constants/marker_lab_config.py), so
    _params_override must fall back to an empirical patch computed from this population's
    own isolated measurements rather than perri's bundled defaults."""
    tests_df = _multi_patient_isolated_series("HDL", {f"p{i}": v for i, v in enumerate([30, 40, 50, 60, 70, 80, 90, 100])})

    patch = _compute_popri_patch(tests_df)
    assert patch is not None
    patch_lower, patch_upper = patch
    assert patch_lower < patch_upper

    expected = _grid_bounds_from_pop_ri(patch_lower, patch_upper)
    params = _params_override("HDL", tests_df, sex="ALL")
    assert params["min_mu"] == pytest.approx(expected["min_mu"])
    assert params["max_mu"] == pytest.approx(expected["max_mu"])
    assert params["max_sigma"] == pytest.approx(expected["max_sigma"])


def test_popri_patch_is_none_when_no_isolated_data():
    tests_df = _isolated_series("p1", "F", n=8, gap_days=30)  # too densely spaced to be isolated
    assert _compute_popri_patch(tests_df) is None


def test_hdl_setpoints_fit_end_to_end_via_empirical_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "cache")
    tests_df = _multi_patient_isolated_series("HDL", {f"p{i}": v for i in range(20) for v in [np.random.default_rng(i).normal(55, 10)]})

    sp_df = compute_sp_df(tests_df, test_code="HDL", force=True)
    assert not sp_df.empty
    assert (sp_df["test_code"] == "HDL").all()


def test_is_fitted_false_before_and_true_after_compute_sp_df():
    tests_df = _isolated_series("p1", "F")
    assert is_fitted(tests_df, "HB") is False

    compute_sp_df(tests_df, test_code="HB", force=True)
    assert is_fitted(tests_df, "HB") is True


def test_is_fitted_false_for_marker_not_present():
    tests_df = _isolated_series("p1", "F")  # only has HB rows
    assert is_fitted(tests_df, "GLU") is False


def test_is_fitted_respects_min_measurements_as_part_of_cache_key():
    tests_df = _isolated_series("p1", "F")
    compute_sp_df(tests_df, test_code="HB", min_measurements=5, force=True)
    assert is_fitted(tests_df, "HB", min_measurements=5) is True
    assert is_fitted(tests_df, "HB", min_measurements=3) is False


def test_fit_markers_fits_every_marker_and_concatenates():
    tests_df = pd.concat([_isolated_series("p1", "F"), _isolated_series("p1", "F", value=90.0).assign(test_code="GLU")], ignore_index=True)
    sp_df = fit_markers(tests_df, ["HB", "GLU"], force=True)
    assert set(sp_df["test_code"].unique()) == {"HB", "GLU"}


def test_fit_markers_skips_marker_with_no_data_without_aborting():
    tests_df = _isolated_series("p1", "F")  # only HB
    sp_df = fit_markers(tests_df, ["HB", "NONEXISTENT_MARKER"], force=True)
    assert set(sp_df["test_code"].unique()) == {"HB"}


def test_fit_markers_reuses_cache_on_second_call(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", cache_dir)
    tests_df = _isolated_series("p1", "F")

    fit_markers(tests_df, ["HB"], force=True)
    n_files_after_first = len(list(cache_dir.glob("sp_df_HB_*.csv")))

    fit_markers(tests_df, ["HB"], force=False)
    n_files_after_second = len(list(cache_dir.glob("sp_df_HB_*.csv")))

    assert n_files_after_first == n_files_after_second == 1


def test_compute_sp_df_canonical_writes_fingerprint_free_cache_name():
    tests_df = _isolated_series("p1", "F")
    compute_sp_df(tests_df, test_code="HB", force=True, canonical=True)

    assert is_fitted_canonical("HB") is True
    assert (setpoints_module.CACHE_DIR / f"sp_df_HB_full_m{DEFAULT_MIN_MEASUREMENTS}.csv").exists()


def test_log_transform_marker_gets_distinct_log_suffixed_cache_name():
    """A marker in utils.log_transform_markers.LOG_TRANSFORM_MARKERS (e.g. GLU) caches
    under a `_log`-suffixed filename, distinct from the non-log-transform cache name a
    plain test_code would use -- so toggling a marker in/out of that set (e.g. while
    investigating whether it should be log-transformed, as happened for TSH) produces two
    separate cache files instead of one silently overwriting the other."""
    tests_df = _isolated_series("p1", "F", value=100.0).assign(test_code="GLU")
    compute_sp_df(tests_df, test_code="GLU", force=True, canonical=True)

    assert is_fitted_canonical("GLU") is True
    assert (setpoints_module.CACHE_DIR / f"sp_df_GLU_full_m{DEFAULT_MIN_MEASUREMENTS}_log.csv").exists()
    assert not (setpoints_module.CACHE_DIR / f"sp_df_GLU_full_m{DEFAULT_MIN_MEASUREMENTS}.csv").exists()


def test_compute_sp_df_canonical_hit_never_touches_tests_df():
    tests_df = _isolated_series("p1", "F")
    compute_sp_df(tests_df, test_code="HB", force=True, canonical=True)

    # On a canonical cache hit, tests_df is never read -- None must work fine.
    sp_df = compute_sp_df(None, test_code="HB", canonical=True)
    assert set(sp_df["anon_id"]) == {"p1"}


def test_compute_sp_df_canonical_miss_lazily_loads_from_input_dir(tmp_path):
    """canonical=True with tests_df=None and a cold cache must load test_code's split
    off disk via input_dir (e.g. a caller building an m5 canonical cache that doesn't
    exist yet, without loading/pre-filtering tests_df itself first)."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    tests_df = _isolated_series("p1", "F")
    tests_df.to_csv(input_dir / "tests.csv", index=False)
    build_tests_by_marker(input_dir)

    sp_df = compute_sp_df(None, test_code="HB", force=True, canonical=True, input_dir=input_dir)

    assert set(sp_df["anon_id"]) == {"p1"}
    assert is_fitted_canonical("HB") is True


def test_fit_markers_lazy_skips_loading_an_already_fitted_marker(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    tests_df = _isolated_series("p1", "F")
    tests_df.to_csv(input_dir / "tests.csv", index=False)
    build_tests_by_marker(input_dir)
    compute_sp_df(tests_df, test_code="HB", force=True, canonical=True)

    sp_df = fit_markers_lazy(input_dir, ["HB"], force=False)
    assert set(sp_df["test_code"].unique()) == {"HB"}


def test_fit_markers_lazy_fits_and_caches_canonically_when_cold(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    tests_df = _isolated_series("p1", "F")
    tests_df.to_csv(input_dir / "tests.csv", index=False)
    build_tests_by_marker(input_dir)

    sp_df = fit_markers_lazy(input_dir, ["HB"], force=True)
    assert set(sp_df["test_code"].unique()) == {"HB"}
    assert is_fitted_canonical("HB") is True
