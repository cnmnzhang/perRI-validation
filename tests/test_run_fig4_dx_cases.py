"""Tests for scripts/run_fig4_dx_cases.py's cache layering.

compute_one_outcome must check, in order: (1) the outcome's cohort cache
(fig4_dx_cases_<outcome>_cohort.csv), (2) the marker's full_population setpoint cache (shared
with fig3_hazard/build_setpoints/fig3_dx) -- and only fall back to loading
tests.csv's per-marker split when both of those miss. The heavy clinical logic
(load_or_create_analysis_ready_cohort, KM/forest-model fitting) is monkeypatched out here
since it's unrelated to and already exercised elsewhere; these tests isolate the caching
behavior that changed.
"""

import pandas as pd
import pytest

import scripts.run_fig4_dx_cases as m
import utils.setpoints as setpoints_module
from constants.runtime import ID_COL, MEASUREMENT_COL, PERRI_Z_SCORE_COL, SEX_COL, TEST_CODE_COL, TS_COL
from utils.setpoints import compute_sp_df


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "sp_cache")


def _seed_full_population_sp_cache(test_code):
    tests_df = pd.DataFrame(
        {
            ID_COL: ["p1"] * 8,
            TS_COL: pd.date_range("2015-01-01", periods=8, freq="120D"),
            TEST_CODE_COL: [test_code] * 8,
            MEASUREMENT_COL: [1.0] * 8,
            SEX_COL: ["F"] * 8,
        }
    )
    compute_sp_df(tests_df, test_code=test_code, force=True, full_population=True)


def _seed_cohort_cache(path):
    pd.DataFrame({ID_COL: ["p1"], "any_in_window": [0], "presenting_ts": ["2015-01-01"]}).to_csv(path, index=False)


@pytest.fixture(autouse=True)
def _stub_heavy_clinical_logic(monkeypatch):
    monkeypatch.setattr(m, "load_or_create_analysis_ready_cohort", lambda **kwargs: pd.DataFrame({ID_COL: ["p1"], "any_in_window": [0], "presenting_ts": ["2015-01-01"], PERRI_Z_SCORE_COL: [0.0]}))
    monkeypatch.setattr(m, "attach_ref_intervals", lambda df: df)
    monkeypatch.setattr(m, "compute_popri_continuous", lambda *a, **k: pd.Series([0.0]))
    monkeypatch.setattr(m, "_fit_forest_model_results", lambda df: None)
    monkeypatch.setattr(m, "_select_example_patient", lambda *a, **k: None)

    class _FakeKMInputs:
        masks = {}

    monkeypatch.setattr(m.KMExclusiveInputs, "from_dataframe", staticmethod(lambda *a, **k: _FakeKMInputs()))


def test_warm_caches_never_touch_tests_csv(tmp_path, monkeypatch):
    """With both the cohort CSV and the marker's full_population setpoint cache already present,
    compute_one_outcome must not call load_tests_marker_subset at all."""
    outcome_name = "aki"
    test_code = m.OUTCOME_REGISTRY[outcome_name].markers[0]
    _seed_full_population_sp_cache(test_code)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _seed_cohort_cache(output_dir / f"fig4_dx_cases_{outcome_name}_cohort.csv")

    def _boom(*a, **k):
        raise AssertionError("load_tests_marker_subset should not be called when both caches are warm")

    monkeypatch.setattr(m, "load_tests_marker_subset", _boom)

    result = m.compute_one_outcome(outcome_name, input_dir=tmp_path / "does_not_exist", dx_incident=pd.DataFrame(), demographics_df=pd.DataFrame({ID_COL: [], "birth_date": []}), output_dir=output_dir, force=False)

    assert result["manifest"]["outcome"] == outcome_name


def test_cold_cohort_cache_still_loads_tests_csv(tmp_path, monkeypatch):
    """With the setpoint cache warm but no cohort CSV yet, compute_one_outcome must still
    load tests.csv (to build the cohort) -- proving the cohort cache isn't skipped blindly."""
    outcome_name = "aki"
    test_code = m.OUTCOME_REGISTRY[outcome_name].markers[0]
    _seed_full_population_sp_cache(test_code)

    output_dir = tmp_path / "out"
    output_dir.mkdir()  # no cohort CSV seeded

    calls = []
    monkeypatch.setattr(m, "load_tests_marker_subset", lambda *a, **k: calls.append(1) or pd.DataFrame(columns=[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL, SEX_COL]))

    m.compute_one_outcome(outcome_name, input_dir=tmp_path / "does_not_exist", dx_incident=pd.DataFrame(), demographics_df=pd.DataFrame({ID_COL: [], "birth_date": []}), output_dir=output_dir, force=False)

    assert len(calls) == 1
