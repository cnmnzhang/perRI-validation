"""Tests for constants/schemas.py + utils/io.py.

Covers two real bugs found and fixed this project: pandas' default NA-string
sniffing silently corrupting the "NA" (sodium) marker, and the Dx table's
sparse icd9/icd10 columns round-tripping through CSV correctly.
"""

import pandas as pd
import pytest

from pathlib import Path

from scripts.build_splits_by_marker import build_splits_by_marker
from utils.io import load_demographics_csv, load_dx_csv, load_iron_mar_csv, load_tests_csv, load_tests_marker_subset, resolve_tests_csv_path
from utils.io import splits_by_marker_dir as resolve_splits_by_marker_dir
from constants.schemas import ALL_SCHEMAS, DEMOGRAPHICS_SCHEMA, DX_SCHEMA, IRON_MAR_SCHEMA, TESTS_SCHEMA


def test_all_schemas_have_required_columns():
    for schema in ALL_SCHEMAS:
        assert schema.required_columns, schema.name


def test_tests_schema_requires_sex():
    assert "sex" in TESTS_SCHEMA.required_columns


def test_load_tests_csv_missing_column_raises(tmp_path):
    path = tmp_path / "tests.csv"
    pd.DataFrame({"anon_id": ["p1"], "ts": ["2020-01-01"], "test_code": ["HB"], "result_value": [13.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sex"):
        load_tests_csv(path)


def test_load_tests_csv_preserves_sodium_marker(tmp_path):
    """Regression test: pandas' default read_csv treats the literal string "NA" as
    missing, which silently deleted every sodium ("NA") row via a downstream dropna.
    """
    path = tmp_path / "tests.csv"
    pd.DataFrame(
        {
            "anon_id": ["p1", "p2"],
            "ts": ["2020-01-01", "2020-01-02"],
            "test_code": ["NA", "HB"],
            "result_value": [140.0, 13.0],
            "sex": ["f", "M"],
        }
    ).to_csv(path, index=False)

    df = load_tests_csv(path)

    assert len(df) == 2
    assert set(df["test_code"]) == {"NA", "HB"}
    assert df.loc[df["test_code"] == "NA", "result_value"].iloc[0] == 140.0


def test_load_tests_csv_restores_blank_test_code_to_sodium_marker(tmp_path):
    """Regression: an upstream export step read "NA" (sodium) with default NA-string
    sniffing, turning it into a real NaN that got written back out as a blank cell.
    A blank test_code must be restored to "NA", not dropped or grouped under "".
    """
    path = tmp_path / "tests.csv"
    pd.DataFrame(
        {
            "anon_id": ["p1", "p2"],
            "ts": ["2020-01-01", "2020-01-02"],
            "test_code": ["", "HB"],
            "result_value": [140.0, 13.0],
            "sex": ["F", "M"],
        }
    ).to_csv(path, index=False)

    df = load_tests_csv(path)

    assert set(df["test_code"]) == {"NA", "HB"}
    assert df.loc[df["test_code"] == "NA", "result_value"].iloc[0] == 140.0


def test_load_tests_csv_uppercases_sex(tmp_path):
    path = tmp_path / "tests.csv"
    pd.DataFrame({"anon_id": ["p1"], "ts": ["2020-01-01"], "test_code": ["HB"], "result_value": [13.0], "sex": ["f"]}).to_csv(path, index=False)
    df = load_tests_csv(path)
    assert df["sex"].iloc[0] == "F"


def test_load_tests_csv_drops_source_only_extra_columns(tmp_path):
    path = tmp_path / "tests.csv"
    pd.DataFrame(
        {
            "epic_pat_id": ["ehr-123"],
            "anon_id": ["p1"],
            "ts": ["2020-01-01"],
            "test_code": ["HB"],
            "result_value": [13.0],
            "sex": ["F"],
            "death_ts": [""],
            "birth_date": ["1980-01-01"],
            "source_file": ["export.csv"],
        }
    ).to_csv(path, index=False)

    df = load_tests_csv(path)

    assert list(df.columns) == list(TESTS_SCHEMA.required_columns)
    assert "epic_pat_id" not in df.columns


def test_load_tests_csv_drops_exact_duplicate_rows(tmp_path):
    """Regression: some markers (TSH/T4FR) are ~85% exact-duplicate rows (same patient/
    timestamp/marker/value) from an upstream export artifact. Left in, these silently
    defeat perri's isolation filter -- every row has a same-timestamp "neighbor," so
    almost nothing reads as isolated regardless of how spread out the patient's real
    visits are. A row repeated verbatim must collapse to one; a same-timestamp row with a
    genuinely different value is a distinct measurement and must survive.
    """
    path = tmp_path / "tests.csv"
    pd.DataFrame(
        {
            "anon_id": ["p1", "p1", "p1", "p1"],
            "ts": ["2020-01-01"] * 4,
            "test_code": ["TSH"] * 4,
            "result_value": [2.5, 2.5, 2.5, 3.1],
            "sex": ["F"] * 4,
        }
    ).to_csv(path, index=False)

    df = load_tests_csv(path)

    assert len(df) == 2
    assert sorted(df["result_value"]) == [2.5, 3.1]


def test_load_tests_csv_drops_unparseable_rows(tmp_path):
    path = tmp_path / "tests.csv"
    pd.DataFrame(
        {
            "anon_id": ["p1", "p2"],
            "ts": ["2020-01-01", "not-a-date"],
            "test_code": ["HB", "HB"],
            "result_value": [13.0, "not-a-number"],
            "sex": ["F", "M"],
        }
    ).to_csv(path, index=False)
    df = load_tests_csv(path)
    assert len(df) == 1
    assert df["anon_id"].iloc[0] == "p1"


def test_load_dx_csv_parses_timestamp_and_restores_sparse_nulls(tmp_path):
    path = tmp_path / "dx.csv"
    pd.DataFrame(
        {
            "anon_id": ["p1", "p2"],
            "icd9": ["280", ""],
            "icd10": ["", "N17"],
            "diagnosis_ts": ["2020-01-01", "2020-06-01"],
        }
    ).to_csv(path, index=False)

    df = load_dx_csv(path)

    assert pd.api.types.is_datetime64_any_dtype(df["diagnosis_ts"])
    assert pd.isna(df.loc[df["anon_id"] == "p1", "icd10"].iloc[0])
    assert pd.isna(df.loc[df["anon_id"] == "p2", "icd9"].iloc[0])
    assert df.loc[df["anon_id"] == "p1", "icd9"].iloc[0] == "280"


def test_load_dx_csv_missing_column_raises(tmp_path):
    path = tmp_path / "dx.csv"
    pd.DataFrame({"anon_id": ["p1"], "icd9": ["280"], "icd10": [""]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=DX_SCHEMA.name):
        load_dx_csv(path)


def test_splits_by_marker_dir_defaults_to_repo_data_dir_when_input_dir_is_none():
    repo_root = Path(__file__).resolve().parents[1]
    assert resolve_splits_by_marker_dir(None) == repo_root / "data" / "cache" / "splits_by_marker"


def test_load_demographics_csv_drops_duplicate_anon_id(tmp_path):
    path = tmp_path / "demographics.csv"
    pd.DataFrame(
        {
            "anon_id": ["p1", "p1"],
            "sex": ["f", "f"],
            "birth_date": ["1980-01-01", "1980-01-01"],
            "death_ts": ["", ""],
        }
    ).to_csv(path, index=False)

    df = load_demographics_csv(path)

    assert len(df) == 1
    assert df["sex"].iloc[0] == "F"


def test_load_demographics_csv_missing_column_raises(tmp_path):
    path = tmp_path / "demographics.csv"
    pd.DataFrame({"anon_id": ["p1"], "sex": ["F"], "birth_date": ["1980-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=DEMOGRAPHICS_SCHEMA.name):
        load_demographics_csv(path)


def test_load_iron_mar_csv_renames_raw_timestamp_column(tmp_path):
    """Real MAR extracts use taken_time, not ts -- the loader must rename before
    validating against IRON_MAR_SCHEMA. route/desc are not required (iron_mar.csv
    must already be pre-filtered to the intended route/formulation), so they pass
    through unrenamed if present."""
    path = tmp_path / "iron_mar.csv"
    pd.DataFrame(
        {
            "anon_id": ["p1"],
            "taken_time": ["2020-01-01 08:00:00"],
        }
    ).to_csv(path, index=False)

    df = load_iron_mar_csv(path)

    assert "ts" in df.columns


def test_load_iron_mar_csv_missing_column_raises(tmp_path):
    path = tmp_path / "iron_mar.csv"
    pd.DataFrame({"anon_id": ["p1"], "desc": ["iron sucrose"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=IRON_MAR_SCHEMA.name):
        load_iron_mar_csv(path)


def _write_tests_csv(input_dir, test_codes=("HB", "GLU", "HDL", "WBC"), n_patients=3):
    rows = [{"anon_id": f"p{i}", "ts": "2015-01-01", "test_code": tc, "result_value": 50.0, "sex": "F"} for i in range(n_patients) for tc in test_codes]
    pd.DataFrame(rows).to_csv(input_dir / "tests.csv", index=False)


def test_load_tests_marker_subset_raises_when_split_not_built(tmp_path):
    """Unlike the earlier design, load_tests_marker_subset never builds the split
    itself -- build_splits_by_marker (run via scripts.build_splits_by_marker)
    is now an explicit prerequisite, like build_dx_incident is for dx_incident.csv."""
    _write_tests_csv(tmp_path)
    with pytest.raises(FileNotFoundError, match="build_splits_by_marker"):
        load_tests_marker_subset(tmp_path, test_codes=["HB"])


def test_load_tests_marker_subset_filters_to_requested_codes(tmp_path):
    _write_tests_csv(tmp_path)
    build_splits_by_marker(tmp_path)
    subset = load_tests_marker_subset(tmp_path, test_codes=["HB", "GLU"])
    assert sorted(subset["test_code"].unique()) == ["GLU", "HB"]


def test_load_tests_marker_subset_accepts_hand_populated_marker_files(tmp_path):
    """A site that already has its data split by marker can drop per-marker CSVs
    straight into splits_by_marker/ -- no tests.csv, no build_splits_by_marker run,
    no _split_complete.json sentinel required."""
    marker_dir = tmp_path / "cache" / "splits_by_marker"
    marker_dir.mkdir(parents=True)
    pd.DataFrame({"anon_id": ["p1"], "ts": ["2015-01-01"], "test_code": ["HB"], "result_value": [13.0], "sex": ["F"]}).to_csv(marker_dir / "HB.csv", index=False)

    subset = load_tests_marker_subset(tmp_path, test_codes=["HB", "GLU"])

    assert list(subset["test_code"].unique()) == ["HB"]  # GLU has no file -- silently skipped, not an error


def test_build_splits_by_marker_writes_one_file_per_marker_present_in_source(tmp_path):
    """The split is marker-agnostic: every test_code in tests.csv gets its own file,
    not just whatever some later caller happens to ask for (HDL is never requested
    by load_tests_marker_subset below, but must still get split out, ready for some
    later caller)."""
    _write_tests_csv(tmp_path)  # HB, GLU, HDL, WBC
    manifest = build_splits_by_marker(tmp_path)
    assert set(manifest.keys()) == {"HB", "GLU", "HDL", "WBC"}

    marker_dir = tmp_path / "cache" / "splits_by_marker"
    assert (marker_dir / "_split_complete.json").exists()
    for tc in ("HB", "GLU", "HDL", "WBC"):
        assert (marker_dir / f"{tc}.csv").exists()


def test_load_tests_marker_subset_reuses_split_without_rereading_source(tmp_path):
    """Once split, later calls -- even for a different marker -- must not re-read
    (or even require) tests.csv. This is the whole point of the shared split: the
    master table is read exactly once, regardless of how many different marker
    subsets get requested afterward."""
    _write_tests_csv(tmp_path)
    build_splits_by_marker(tmp_path)

    (tmp_path / "tests.csv").unlink()
    subset = load_tests_marker_subset(tmp_path, test_codes=["GLU"])
    assert (subset["test_code"] == "GLU").all()


def test_build_splits_by_marker_force_rederives(tmp_path):
    _write_tests_csv(tmp_path, test_codes=("HB",), n_patients=2)
    build_splits_by_marker(tmp_path)

    _write_tests_csv(tmp_path, test_codes=("HB",), n_patients=5)  # tests.csv changed
    build_splits_by_marker(tmp_path, force=True)
    subset = load_tests_marker_subset(tmp_path, test_codes=["HB"])
    assert subset["anon_id"].nunique() == 5


def test_build_splits_by_marker_skips_rebuild_without_force(tmp_path):
    _write_tests_csv(tmp_path, test_codes=("HB",), n_patients=2)
    build_splits_by_marker(tmp_path)

    _write_tests_csv(tmp_path, test_codes=("HB",), n_patients=5)  # tests.csv changed
    build_splits_by_marker(tmp_path)  # no force -- must not re-derive
    subset = load_tests_marker_subset(tmp_path, test_codes=["HB"])
    assert subset["anon_id"].nunique() == 2


def test_resolve_tests_csv_path_prefers_plain_csv_over_gz(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "tests.csv", index=False)
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "tests.csv.gz", index=False, compression="gzip")
    assert resolve_tests_csv_path(tmp_path) == tmp_path / "tests.csv"


def test_resolve_tests_csv_path_falls_back_to_gz(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "tests.csv.gz", index=False, compression="gzip")
    assert resolve_tests_csv_path(tmp_path) == tmp_path / "tests.csv.gz"


def test_resolve_tests_csv_path_raises_when_neither_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="tests.csv"):
        resolve_tests_csv_path(tmp_path)




def test_load_tests_marker_subset_reads_gzipped_source(tmp_path):
    _write_tests_csv(tmp_path)
    gz_path = tmp_path / "tests.csv.gz"
    pd.read_csv(tmp_path / "tests.csv").to_csv(gz_path, index=False, compression="gzip")
    (tmp_path / "tests.csv").unlink()

    build_splits_by_marker(tmp_path)
    subset = load_tests_marker_subset(tmp_path, test_codes=["HB"])
    assert (subset["test_code"] == "HB").all()
