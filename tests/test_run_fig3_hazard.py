"""Tests for scripts/run_fig3_hazard.py's plot-data-first caching.

Patient-level setpoint fitting + per-marker Cox regression is the expensive part of a
rerun. Once fig3a_hr_by_model.csv/fig3b_hr_by_baseline.csv are already on disk, a rerun
should plot straight from them instead of refitting anything -- like fig3_dx's
fig3_km_data.csv cache.
"""

import pandas as pd
import pytest

import scripts.run_fig3_hazard as m
import utils.setpoints as setpoints_module
from scripts.run_fig3_hazard import _filter_invalid_cv_patients, _filter_sp_df, run
from utils.setpoints import compute_sp_df


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "sp_cache")


def _seed_hr_by_model(path):
    pd.DataFrame(
        {
            "test_code": ["HB"],
            "model": ["bayesian"],
            "variable": ["mu"],
            "exp(coef)": [1.1],
            "exp(coef) lower 95%": [0.9],
            "exp(coef) upper 95%": [1.3],
            "p": [0.2],
            "n": [100],
        }
    ).to_csv(path, index=False)


def _seed_hr_by_baseline(path):
    pd.DataFrame(
        {
            "test_code": ["HB"],
            "model": ["bayesian"],
            "baseline_label": ["1"],
            "baseline_index": [1],
            "variable": ["mu"],
            "hr": [1.1],
            "ci_lower": [0.9],
            "ci_upper": [1.3],
            "n": [100],
        }
    ).to_csv(path, index=False)


def test_cached_plot_data_skips_setpoint_fitting_entirely(tmp_path):
    """A cache hit on both CSVs must not even try to read demographics.csv -- proves
    fit_markers_lazy (and therefore the whole expensive fitting + Cox pass) never runs."""
    input_dir = tmp_path / "data"  # deliberately never created -- would raise if touched
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _seed_hr_by_model(output_dir / "fig3a_hr_by_model.csv")
    _seed_hr_by_baseline(output_dir / "fig3b_hr_by_baseline.csv")

    manifest = run(input_dir=input_dir, output_dir=output_dir, force=False)

    assert manifest["fig3a_n_marker_model_rows"] == 1
    assert manifest["fig3b_n_rows"] == 1
    assert manifest["n_markers_fitted"] == 1


def test_force_bypasses_the_plot_data_cache(tmp_path):
    """force=True must re-derive, which requires demographics.csv to actually exist --
    a missing input_dir should raise, proving the cached CSVs were NOT reused."""
    input_dir = tmp_path / "data"  # never created
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _seed_hr_by_model(output_dir / "fig3a_hr_by_model.csv")
    _seed_hr_by_baseline(output_dir / "fig3b_hr_by_baseline.csv")

    with pytest.raises(Exception):
        run(input_dir=input_dir, output_dir=output_dir, force=True)


def test_force_rebuilds_plot_data_but_does_not_refresh_the_setpoint_dependency(tmp_path, monkeypatch, capsys):
    """force=True at this script's own layer (fig3a/b) must not cascade into re-fitting
    setpoints that are already cached -- that dependency is only ever refreshed by
    run_setpoints_by_marker's own --force."""
    monkeypatch.setattr(m, "TESTCODES_LIST", ["HB"])

    input_dir = tmp_path / "data"
    input_dir.mkdir()
    pd.DataFrame({"anon_id": ["p1"], "sex": ["F"], "birth_date": ["1980-01-01"], "death_ts": [None]}).to_csv(input_dir / "demographics.csv", index=False)

    tests_df = pd.DataFrame(
        {
            "anon_id": ["p1"] * 8,
            "ts": pd.date_range("2015-01-01", periods=8, freq="120D"),
            "test_code": ["HB"] * 8,
            "result_value": [13.0] * 8,
            "sex": ["F"] * 8,
        }
    )
    compute_sp_df(tests_df, test_code="HB", force=True, canonical=True)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _seed_hr_by_model(output_dir / "fig3a_hr_by_model.csv")
    _seed_hr_by_baseline(output_dir / "fig3b_hr_by_baseline.csv")

    capsys.readouterr()  # discard the seeding step's own "[cache] miss/saved" output above
    run(input_dir=input_dir, output_dir=output_dir, force=True)  # must not raise, and must not refit HB

    out = capsys.readouterr().out
    assert "[cache] hit sp_df_HB_full_m" in out
    assert "[cache] miss sp_df_HB_full_m" not in out


def _sp_row(anon_id, test_code, index, mu, sigma, model="bayesian"):
    return {"anon_id": anon_id, "test_code": test_code, "model": model, "mu": mu, "sigma": sigma, "index": index}


def test_filter_invalid_cv_patients_drops_whole_sequence_on_negative_cv():
    """Regression: bayesian-setpoint-inference's filter_sp_df drops a patient's *entire*
    setpoint sequence for a marker if any measurement from index>=3 onward has cv < 0 --
    not just the offending row. p1 has one bad row at index 3; all 4 of p1's rows must go."""
    sp_df = pd.DataFrame(
        [
            _sp_row("p1", "HB", 0, mu=13.0, sigma=1.0),
            _sp_row("p1", "HB", 1, mu=13.0, sigma=1.0),
            _sp_row("p1", "HB", 2, mu=13.0, sigma=1.0),
            _sp_row("p1", "HB", 3, mu=13.0, sigma=-1.0),  # cv = -0.077 -- invalid
            _sp_row("p2", "HB", 0, mu=13.0, sigma=1.0),
            _sp_row("p2", "HB", 3, mu=13.0, sigma=1.0),  # cv = 0.077 -- valid
        ]
    )
    out = _filter_invalid_cv_patients(sp_df)
    assert set(out["anon_id"]) == {"p2"}
    assert len(out) == 2


def test_filter_invalid_cv_patients_ignores_index_below_3():
    """The cv guard only applies from the 4th isolated measurement (index>=3) onward --
    an early negative cv (still stabilizing) must not drop the patient."""
    sp_df = pd.DataFrame(
        [
            _sp_row("p1", "HB", 0, mu=13.0, sigma=-1.0),  # invalid cv, but index < 3
            _sp_row("p1", "HB", 1, mu=13.0, sigma=1.0),
        ]
    )
    out = _filter_invalid_cv_patients(sp_df)
    assert len(out) == 2


def test_filter_invalid_cv_patients_allows_cv_above_1_for_log_transform_markers():
    """Non-log markers reject cv > 1; log-transform markers (e.g. HSCRP) legitimately
    exceed 1 after the lognormal back-transform and must not be dropped for that alone."""
    sp_df = pd.DataFrame(
        [
            _sp_row("p1", "HSCRP", 3, mu=1.0, sigma=5.0),  # cv = 5.0, but HSCRP is log-transform -- valid
            _sp_row("p2", "HB", 3, mu=13.0, sigma=20.0),  # cv > 1, HB is not log-transform -- invalid
        ]
    )
    out = _filter_invalid_cv_patients(sp_df)
    assert set(out["anon_id"]) == {"p1"}


def _sp_row_ts(anon_id, test_code, index, ts, mu=13.0, sigma=1.0, model="bayesian"):
    return {"anon_id": anon_id, "test_code": test_code, "model": model, "mu": mu, "sigma": sigma, "index": index, "ts": pd.Timestamp(ts)}


def test_filter_sp_df_drops_whole_sequence_below_min_measurements():
    """Regression: a patient with only 3-4 isolated measurements passes compute_sp_df's
    looser fitting bar (DEFAULT_MIN_MEASUREMENTS=3) and gets fit, contributing rows at low
    baseline indices -- but bayesian-setpoint-inference's filter_sp_df requires 5 total
    measurements (MIN_MEASUREMENTS_FOR_FILTER) before considering a patient at all. Without
    this, low-baseline-index cohorts were inflated ~5.7x ground truth's n at baseline_index=1."""
    sp_df = pd.DataFrame(
        [
            _sp_row_ts("p1", "HB", 0, "2015-01-01"),
            _sp_row_ts("p1", "HB", 1, "2015-06-01"),
            _sp_row_ts("p1", "HB", 2, "2016-01-01"),  # only 3 rows -- must be dropped entirely
            _sp_row_ts("p2", "HB", 0, "2015-01-01"),
            _sp_row_ts("p2", "HB", 1, "2015-06-01"),
            _sp_row_ts("p2", "HB", 2, "2016-01-01"),
            _sp_row_ts("p2", "HB", 3, "2016-06-01"),
            _sp_row_ts("p2", "HB", 4, "2017-01-01"),  # 5 rows -- must survive
        ]
    )
    out = _filter_sp_df(sp_df)
    assert set(out["anon_id"]) == {"p2"}
    assert len(out) == 5


def test_filter_sp_df_date_filter_can_push_a_patient_below_the_min_measurements_bar():
    """The date filter runs *before* the min-measurements count, per filter_sp_df's own
    order -- a measurement at/after MAX_FIT_DATE doesn't count toward the bar, and can drop
    an otherwise-5-measurement patient below it."""
    from constants.runtime import MAX_FIT_DATE

    late_ts = pd.Timestamp(MAX_FIT_DATE) + pd.Timedelta(days=1)
    sp_df = pd.DataFrame(
        [
            _sp_row_ts("p1", "HB", 0, "2015-01-01"),
            _sp_row_ts("p1", "HB", 1, "2015-06-01"),
            _sp_row_ts("p1", "HB", 2, "2016-01-01"),
            _sp_row_ts("p1", "HB", 3, "2016-06-01"),
            _sp_row_ts("p1", "HB", 4, late_ts),  # after MAX_FIT_DATE -- doesn't count
        ]
    )
    out = _filter_sp_df(sp_df)
    assert out.empty
