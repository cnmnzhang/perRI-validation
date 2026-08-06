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
from scripts.run_fig3_hazard import run
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
