"""Tests for perri_validation/scripts/run_fig5_iron_infusion.py's failure handling
and iv_iron_bundle/ caching.

fig5_iron_infusion has only one marker/outcome (HB), unlike fig3_hazard/fig3_dx/
fig4_dx_cases which cover many -- a build failure here can't degrade to a partial
result. It should still fail the same *way* the others do: caught and recorded in
manifest.json as an "error" key, not an uncaught traceback.
"""

import pandas as pd
import pytest

import perri_validation.scripts.run_fig5_iron_infusion as m
import perri_validation.utils.setpoints as setpoints_module
from perri_validation.scripts.run_fig5_iron_infusion import run
from perri_validation.scripts.run_tests_by_marker import build_tests_by_marker
from perri_validation.utils.io import load_tests_marker_subset
from perri_validation.utils.setpoints import compute_sp_df


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "cache")


def _write_iv_iron_inputs(input_dir, n_patients=4):
    """n_patients each with: 14 isolated HB measurements (120d apart, 2015-2019) for a
    setpoint history, a single-dose IV iron course on 2020-06-01, a pre-course lab 17 days
    before it, and a post-course lab 92 days after it -- enough to clear every
    IronInfusionConfig default threshold (pre_days_max=60, post_days_min/max=60/180,
    setpoint_lookback_min/max=365/1095, min_setpoint_measurements=3). Pre-lab values vary
    per patient so fig5interaction_on_ax's n_hb_bins has enough distinct values to bin
    (a single patient makes seaborn's pointplot choke -- unrelated to bundle caching)."""
    setpoint_dates = pd.date_range("2015-01-01", periods=14, freq="120D")
    rows = []
    for i in range(n_patients):
        pid = f"p{i}"
        rows += [{"anon_id": pid, "ts": d, "test_code": "HB", "result_value": 13.0, "sex": "F"} for d in setpoint_dates]
        rows.append({"anon_id": pid, "ts": pd.Timestamp("2020-05-15"), "test_code": "HB", "result_value": 7.0 + i, "sex": "F"})  # pre
        rows.append({"anon_id": pid, "ts": pd.Timestamp("2020-09-01"), "test_code": "HB", "result_value": 11.0, "sex": "F"})  # post
    tests_df = pd.DataFrame(rows)
    tests_df["epic_pat_id"] = "source-only-id"
    tests_df.to_csv(input_dir / "tests.csv", index=False)
    pd.DataFrame({"anon_id": [f"p{i}" for i in range(n_patients)], "ts": [pd.Timestamp("2020-06-01")] * n_patients}).to_csv(input_dir / "iron_mar.csv", index=False)


def test_missing_hb_marker_is_caught_and_recorded_not_raised(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    # tests.csv with no HB rows at all
    pd.DataFrame(
        {
            "anon_id": ["p1"],
            "ts": ["2020-01-01"],
            "test_code": ["GLU"],
            "result_value": [90.0],
            "sex": ["F"],
        }
    ).to_csv(input_dir / "tests.csv", index=False)
    pd.DataFrame({"anon_id": ["p1"], "ts": ["2020-01-01"]}).to_csv(input_dir / "iron_mar.csv", index=False)
    build_tests_by_marker(input_dir)

    manifest = run(input_dir=input_dir, output_dir=output_dir, force=True)  # must not raise

    assert "error" in manifest
    assert "HB" in manifest["error"]

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    assert "error" in manifest_path.read_text()


def test_bundle_cache_round_trips_and_skips_rebuild_on_second_run(tmp_path, monkeypatch):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    _write_iv_iron_inputs(input_dir)
    build_tests_by_marker(input_dir)

    first_manifest = run(input_dir=input_dir, output_dir=output_dir, force=True)
    assert "error" not in first_manifest
    assert first_manifest["n_patients"] == 4
    assert (output_dir / "iv_iron_cohort.csv").exists()
    trajectory_path = output_dir / "fig5_trajectory_data.csv"
    assert trajectory_path.exists()
    trajectory_df = pd.read_csv(trajectory_path)
    assert not trajectory_df.empty
    assert {"anon_id", "result_value", "days_pre", "days_post"}.issubset(trajectory_df.columns)
    assert "epic_pat_id" not in trajectory_df.columns
    assert {"iv_iron_cohort.csv", "fig5_trajectory_data.csv"}.issubset(first_manifest["outputs"])

    bundle_dir = output_dir / "iv_iron_bundle"
    for name in (*m.BUNDLE_DATE_COLUMNS, "counts"):
        suffix = ".json" if name == "counts" else ".csv"
        assert (bundle_dir / f"{name}{suffix}").exists()

    # Deleting the raw inputs proves the second run can't possibly be rebuilding the
    # bundle from tests.csv/iron_mar.csv -- it must be a pure cache hit.
    (input_dir / "tests.csv").unlink()
    (input_dir / "iron_mar.csv").unlink()

    def _must_not_be_called(**kwargs):
        raise AssertionError("build_iv_iron_bundle was called on a cache hit")

    monkeypatch.setattr(m, "build_iv_iron_bundle", _must_not_be_called)

    second_manifest = run(input_dir=input_dir, output_dir=output_dir, force=False)
    assert "error" not in second_manifest
    assert second_manifest["n_patients"] == first_manifest["n_patients"]
    assert (output_dir / "iron_infusion_mosaic.svg").exists()


def _add_non_cohort_hb_patient(input_dir, pid="p_extra"):
    """An HB patient with isolated setpoint history but no IV iron course -- pads the
    full population beyond the IV-iron cohort, so "filter the full-population fit" and
    "fit the cohort alone" are actually distinguishable populations (and therefore
    distinguishable cache files)."""
    setpoint_dates = pd.date_range("2015-01-01", periods=14, freq="120D")
    rows = [{"anon_id": pid, "ts": d, "test_code": "HB", "result_value": 20.0, "sex": "F"} for d in setpoint_dates]
    # parse_dates avoids a mixed string/Timestamp "ts" column when concatenated below --
    # the exact same mixed-date-format bug diagnosed for the real tests.csv.gz.
    existing = pd.read_csv(input_dir / "tests.csv", parse_dates=["ts"])
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(input_dir / "tests.csv", index=False)


def test_get_hb_setpoints_filters_full_population_fit_when_cached(tmp_path):
    """Unit-level proof of _get_hb_setpoints itself: with the full-population fit
    already cached, it must return exactly that fit's rows for the cohort patients --
    not a separately (and possibly numerically different) fit of the cohort alone."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    _write_iv_iron_inputs(input_dir, n_patients=4)
    _add_non_cohort_hb_patient(input_dir)
    build_tests_by_marker(input_dir)

    full_tests_df = load_tests_marker_subset(input_dir, test_codes=["HB"])
    hb_full = full_tests_df[full_tests_df["test_code"] == "HB"].copy()
    full_sp = compute_sp_df(hb_full, test_code="HB", force=True)
    assert full_sp["anon_id"].nunique() == 5  # 4 cohort patients + 1 non-cohort patient

    cohort_ids = [f"p{i}" for i in range(4)]
    cohort_hb = hb_full[hb_full["anon_id"].isin(cohort_ids)].copy()

    result = m._get_hb_setpoints(hb_full, cohort_hb, cohort_ids, test_code="HB")

    expected = full_sp[full_sp["anon_id"].isin(cohort_ids)].sort_values(["anon_id", "index"]).reset_index(drop=True)
    result = result.sort_values(["anon_id", "index"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)
    assert set(result["anon_id"]) == set(cohort_ids)  # the non-cohort patient must not leak in


def test_reuses_full_population_hb_fit_instead_of_refitting_cohort(tmp_path, capsys):
    """End-to-end: with HB already fit on the full population (e.g. by
    run_setpoints_by_marker or fig3_hazard), a full fig5_iron_infusion run must filter
    that fit down to the IV-iron cohort instead of fitting the cohort's smaller
    population itself -- proven by the "filtering instead of refitting" log line and no
    additional sp_df_HB_*.csv cache file appearing."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    _write_iv_iron_inputs(input_dir, n_patients=4)
    _add_non_cohort_hb_patient(input_dir)
    build_tests_by_marker(input_dir)

    # Pre-fit the FULL HB population, via the exact same loading path build_iv_iron_bundle
    # uses internally (load_tests_marker_subset) -- so the population, and therefore the
    # cache's content hash, is guaranteed identical, not just "close enough".
    full_tests_df = load_tests_marker_subset(input_dir, test_codes=["HB"])
    hb_full = full_tests_df[full_tests_df["test_code"] == "HB"].copy()
    compute_sp_df(hb_full, test_code="HB", force=True)

    cache_dir = tmp_path / "cache"
    assert len(list(cache_dir.glob("sp_df_HB_*.csv"))) == 1

    manifest = run(input_dir=input_dir, output_dir=output_dir, force=False)
    assert "error" not in manifest
    assert manifest["n_patients"] == 4  # only the IV-iron cohort, not the extra patient

    assert "already cached -- filtering instead of refitting" in capsys.readouterr().out

    # The whole point: no additional cache file for a cohort-only population.
    assert len(list(cache_dir.glob("sp_df_HB_*.csv"))) == 1
