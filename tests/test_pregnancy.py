"""Tests for the pregnancy analysis: schema loaders + cohort-building logic.

Setpoint-fitting tests monkeypatch CACHE_DIR to tmp_path (never touch the
real, shared data/cache/ -- see test_setpoints.py's docstring).
"""

import pandas as pd
import pytest

import utils.setpoints as setpoints_module
from utils.io import load_pregnancy_labs_csv, load_pregnancy_outcomes_and_demogs_csv
from utils.clinical.pregnancy import PAIR_SPECS, compute_inpreg_analysis_df, compute_prepreg_setpoint_table, select_trimester_midpoints, task1_summary_df, task2_results_df, task3_payload


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(setpoints_module, "CACHE_DIR", tmp_path / "cache")


def test_load_pregnancy_labs_csv_matches_tests_schema_shape(tmp_path):
    path = tmp_path / "pregnancy_labs.csv"
    pd.DataFrame(
        {
            "anon_id": ["m1", "m2"],
            "ts": ["2020-01-01", "2020-02-01"],
            "test_code": ["WBC", "HCT"],
            "result_value": [6.5, 38.0],
        }
    ).to_csv(path, index=False)

    df = load_pregnancy_labs_csv(path)

    assert list(df.columns) == ["anon_id", "ts", "test_code", "result_value", "sex"]
    assert (df["sex"] == "F").all()


def test_load_pregnancy_outcomes_and_demogs_csv_derives_conception_date(tmp_path):
    path = tmp_path / "pregnancy_outcomes_and_demogs.csv"
    pd.DataFrame(
        {
            "anon_id": ["m1"],
            "delivery_date": ["2020-10-01"],
            "gestational_age": [40.0],
            "rbc_tf": ["yes"],
            "pih": ["0"],
        }
    ).to_csv(path, index=False)

    df = load_pregnancy_outcomes_and_demogs_csv(path)

    assert df["anon_id"].iloc[0] == "m1"
    assert df["received_tf"].iloc[0] == 1
    assert df["pih"].iloc[0] == 0
    expected_conception = pd.Timestamp("2020-10-01") - pd.Timedelta(weeks=40.0)
    assert df["conception_date"].iloc[0] == expected_conception


def test_load_pregnancy_outcomes_and_demogs_csv_defaults_missing_gestational_age(tmp_path):
    path = tmp_path / "pregnancy_outcomes_and_demogs.csv"
    pd.DataFrame({"anon_id": ["m1"], "delivery_date": ["2020-10-01"], "gestational_age": [""], "rbc_tf": ["0"], "pih": ["0"]}).to_csv(path, index=False)
    df = load_pregnancy_outcomes_and_demogs_csv(path)
    assert df["gestational_age"].iloc[0] == 40.0


def test_load_pregnancy_outcomes_and_demogs_csv_missing_column_raises(tmp_path):
    path = tmp_path / "pregnancy_outcomes_and_demogs.csv"
    pd.DataFrame({"anon_id": ["m1"], "delivery_date": ["2020-10-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="pregnancy_outcomes_and_demogs"):
        load_pregnancy_outcomes_and_demogs_csv(path)


def _synthetic_pregnancy_cohort(n_mothers=15, seed=0):
    """One mother per index, WBC + HCT, isolated pre-conception + in-pregnancy rows."""
    import numpy as np

    rng = np.random.default_rng(seed)
    labs_rows, demog_rows = [], []
    for i in range(n_mothers):
        mid = f"m{i:03d}"
        conception = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
        delivery = conception + pd.Timedelta(weeks=40.0)
        for tc, (lo, hi) in {"WBC": (4.3, 10.0), "HCT": (36.0, 45.0)}.items():
            mid_val = rng.uniform(lo, hi)
            for k in range(5):
                ts = conception - pd.Timedelta(days=int(120 * (5 - k)))
                labs_rows.append((mid, ts, tc, round(float(mid_val + rng.normal(0, 0.2)), 2)))
            for week in [4, 16, 28, 36]:
                ts = conception + pd.Timedelta(weeks=week)
                labs_rows.append((mid, ts, tc, round(float(mid_val + rng.normal(0, 0.2)), 2)))
        pih = int(rng.random() < 0.3)
        rbc_tf = int(rng.random() < 0.3)
        demog_rows.append((mid, delivery.strftime("%Y-%m-%d"), 40.0, rbc_tf, pih))

    labs_df = pd.DataFrame(labs_rows, columns=["anon_id", "ts", "test_code", "result_value"])
    demog_df = pd.DataFrame(demog_rows, columns=["anon_id", "delivery_date", "gestational_age", "rbc_tf", "pih"])
    return labs_df, demog_df


def test_full_cohort_pipeline_runs_end_to_end(tmp_path):
    labs_path, demog_path = tmp_path / "pregnancy_labs.csv", tmp_path / "pregnancy_outcomes_and_demogs.csv"
    labs_df, demog_df = _synthetic_pregnancy_cohort()
    labs_df.to_csv(labs_path, index=False)
    demog_df.to_csv(demog_path, index=False)

    tests_df = load_pregnancy_labs_csv(labs_path)
    demog = load_pregnancy_outcomes_and_demogs_csv(demog_path)

    for pair in PAIR_SPECS.values():
        setpoint_table = compute_prepreg_setpoint_table(tests_df, demog, pair, force=True)
        assert not setpoint_table.empty, f"{pair.key}: expected setpoints"
        assert set(setpoint_table["anon_id"]).issubset(set(demog["anon_id"]))

        analysis_df = compute_inpreg_analysis_df(tests_df, demog, setpoint_table, pair)
        assert not analysis_df.empty, f"{pair.key}: expected in-pregnancy rows"
        assert analysis_df["gestational_age_weeks"].between(0, 42).all()
        assert set(analysis_df["trimester"].dropna().astype(str)).issubset({"t1", "t2", "t3"})

        trimester_df = select_trimester_midpoints(analysis_df)
        assert not trimester_df.empty
        # at most one row per (patient, trimester)
        assert not trimester_df.duplicated(subset=["anon_id", "trimester"]).any()

        task1_df = task1_summary_df(analysis_df, pair)
        assert not task1_df.empty
        assert (task1_df["q10"] <= task1_df["q50"]).all()
        assert (task1_df["q50"] <= task1_df["q90"]).all()

        task2_df = task2_results_df(trimester_df, pair)
        assert set(task2_df["trimester"]) <= {"t1", "t2", "t3"}

        payload, tidy = task3_payload(trimester_df, pair, annotate_n=True)
        assert payload is not None
        assert payload["rate_df"].shape == (2, 2)
        assert not tidy.empty


def test_setpoint_table_uses_last_preconception_index_as_setpoint(tmp_path):
    """Regression check: "the setpoint" must be the *last* pre-conception isolated
    measurement, not the first or an average."""
    labs_path, demog_path = tmp_path / "pregnancy_labs.csv", tmp_path / "pregnancy_outcomes_and_demogs.csv"
    conception = pd.Timestamp("2015-06-01")
    delivery = conception + pd.Timedelta(weeks=40.0)
    dates = [conception - pd.Timedelta(days=120 * (5 - k)) for k in range(5)]
    values = [4.5, 5.0, 5.5, 6.0, 6.5]  # rising trend -- last isolated value is 6.5

    labs_df = pd.DataFrame({"anon_id": ["m1"] * 5, "ts": dates, "test_code": ["WBC"] * 5, "result_value": values})
    labs_df.to_csv(labs_path, index=False)
    demog_df = pd.DataFrame({"anon_id": ["m1"], "delivery_date": [delivery.strftime("%Y-%m-%d")], "gestational_age": [40.0], "rbc_tf": [0], "pih": [0]})
    demog_df.to_csv(demog_path, index=False)

    tests_df = load_pregnancy_labs_csv(labs_path)
    demog = load_pregnancy_outcomes_and_demogs_csv(demog_path)
    pair = PAIR_SPECS["wbc_pih"]

    setpoint_table = compute_prepreg_setpoint_table(tests_df, demog, pair, force=True)

    assert len(setpoint_table) == 1
    assert setpoint_table["setpoint_measurement"].iloc[0] == pytest.approx(6.5)
    assert setpoint_table["n_isolated_prepreg"].iloc[0] == 5
