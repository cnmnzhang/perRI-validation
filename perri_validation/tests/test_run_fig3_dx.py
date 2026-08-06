"""Tests for perri_validation/scripts/run_fig3_dx.py's fig3_km_data.csv caching.

The diagnosis-anchoring + KM-fitting pass (get_at_risk_population/get_groups_from_config/
compute_event_time/_fit_km_for_groups across all 8 diagnoses) is the expensive part of a
rerun once compute_sp_df's own per-marker fits are already cached -- caching
fig3_km_data.csv itself (like fig4_dx_cases's per-outcome cohort cache) means a rerun with
that file already present skips it entirely.
"""

import pandas as pd
import pytest

import perri_validation.utils.setpoints as setpoints_module
from perri_validation.scripts.run_fig3_dx import run


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "sp_cache")


def _seed_km_data(path):
    pd.DataFrame(
        {
            "timeline": [0.0, 1.0],
            "survival": [1.0, 0.9],
            "diagnosis": ["Cirrhosis", "Cirrhosis"],
            "group": ["< 25%", "< 25%"],
            "setpoint_type": ["ALB", "ALB"],
            "count": [10, 10],
        }
    ).to_csv(path, index=False)


def test_cached_km_data_skips_dx_incident_entirely(tmp_path):
    """A cache hit must not even try to read dx_incident.csv -- proves build_fig3_km_data
    (and therefore the whole expensive fitting pass) never runs on a cache hit."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _seed_km_data(output_dir / "fig3_km_data.csv")

    missing_dx_incident_path = tmp_path / "does_not_exist.csv"

    manifest = run(input_dir=input_dir, output_dir=output_dir, dx_incident_path=missing_dx_incident_path, force=False)

    assert manifest["fig3_km_n_facets"] == 1
    assert "error" not in manifest


def test_force_bypasses_the_km_data_cache(tmp_path, monkeypatch):
    """force=True must re-derive, which requires dx_incident.csv to actually exist --
    a missing one should raise, proving the cached CSV was NOT reused."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _seed_km_data(output_dir / "fig3_km_data.csv")

    missing_dx_incident_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        run(input_dir=input_dir, output_dir=output_dir, dx_incident_path=missing_dx_incident_path, force=True)
