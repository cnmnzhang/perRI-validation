"""Generic table loaders + schema validation for 's input tables."""

import time
from pathlib import Path
from typing import Union

import pandas as pd

from constants.runtime import DIAGNOSIS_TS_COL, ICD9_COL, ICD10_COL, ID_COL, MEASUREMENT_COL, SEX_COL, TEST_CODE_COL, TS_COL
from constants.schemas import DEMOGRAPHICS_SCHEMA, DX_SCHEMA, IRON_MAR_SCHEMA, PREGNANCY_LABS_SCHEMA, PREGNANCY_OUTCOMES_AND_DEMOGS_SCHEMA, TESTS_SCHEMA, TableSchema

NUMERIC_OUTCOME_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
NUMERIC_OUTCOME_FALSE_VALUES = {"0", "false", "f", "no", "n", ""}

# Reads at or above this size print a start/complete line with elapsed time -- large enough
# that per-marker split files (usually a few MB) stay quiet, but the multi-GB master Tests
# table (and any unusually large marker) doesn't read in total silence.
_PRINT_THRESHOLD_MB = 20


def _coerce_binary(series: "pd.Series") -> "pd.Series":
    """Coerce a messy 0/1/true/false/yes/no-style column to a clean 0/1 int column."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0) > 0).astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(NUMERIC_OUTCOME_TRUE_VALUES).astype(int)


def _validate_columns(df: pd.DataFrame, schema: TableSchema) -> None:
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{schema.name} table is missing required column(s) {missing}. " f"Required: {list(schema.required_columns)}. Got: {list(df.columns)}")


def _read_csv_no_default_na(path: Union[str, Path], dtype: dict) -> pd.DataFrame:
    """pd.read_csv with pandas' default NA-string sniffing disabled.

    "NA" (sodium)" in this schema collide with pandas' default
    na_values list. Callers that need NaN handling
    for numeric/date columns will coerce explicitly afterward
    (pd.to_numeric/pd.to_datetime with errors="coerce")
    """
    return pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values=[])


def resolve_tests_csv_path(input_dir: Union[str, Path]) -> Path:
    """Resolves the master Tests table: `tests.csv` if present, else `tests.csv.gz`
    (pandas' read_csv auto-decompresses .gz based on the path's extension, so no
    special handling is needed once the right path is found). Raises a clear
    FileNotFoundError naming both paths checked if neither exists.
    """
    input_dir = Path(input_dir)
    plain = input_dir / "tests.csv"
    if plain.exists():
        return plain
    gz = input_dir / "tests.csv.gz"
    if gz.exists():
        return gz
    raise FileNotFoundError(f"No Tests table found -- checked {plain} and {gz}.")


def load_tests_csv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    size_mb = path.stat().st_size / 1e6 if path.exists() else 0
    verbose = size_mb >= _PRINT_THRESHOLD_MB
    if verbose:
        print(f"reading {path} ({size_mb:,.0f} MB)...")
        t0 = time.time()

    df = _read_csv_no_default_na(path, {ID_COL: str, TEST_CODE_COL: str, SEX_COL: str})
    _validate_columns(df, TESTS_SCHEMA)
    # A blank test_code is the sodium marker "NA"
    df.loc[df[TEST_CODE_COL] == "", TEST_CODE_COL] = "NA"
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    df[MEASUREMENT_COL] = pd.to_numeric(df[MEASUREMENT_COL], errors="coerce")
    df[SEX_COL] = df[SEX_COL].str.upper()
    df = df.dropna(subset=[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL, SEX_COL])
    # Normalize to the documented Tests schema. Source exports may contain identifiers
    # and demographics such as epic_pat_id/death_ts/birth_date, but downstream code uses
    # anon_id and loads demographics separately; passing extras through leaks them into
    # derived caches and figure-data artifacts.
    df = df[list(TESTS_SCHEMA.required_columns)]
    df = df.drop_duplicates(subset=[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL])
    df = df.sort_values([ID_COL, TS_COL]).reset_index(drop=True)

    if verbose:
        print(f"read {path}: {len(df):,} rows in {time.time() - t0:.1f}s")
    return df



def splits_by_marker_dir(input_dir: Union[str, Path, None]) -> Path:
    if input_dir is None:
        input_dir = Path(__file__).resolve().parents[1] / "data"
    return Path(input_dir) / "cache" / "splits_by_marker"



def load_tests_marker_subset(input_dir: Union[str, Path] = None, test_codes: list=None) -> pd.DataFrame:
    """Loads `test_codes` from data/cache/splits_by_marker/{test_code}.csv -- either the
    split built by scripts/build_splits_by_marker.py, or per-marker CSVs (each matching
    TESTS_SCHEMA, like the master Tests table) dropped in directly by a site that already
    has its data split by marker and would rather skip building/shipping a combined
    tests.csv at all. Either way, a requested marker with no file present is silently
    skipped (same graceful-handling-of-missing-markers behavior as everywhere else), not
    an error -- only a missing splits_by_marker/ directory raises, since that means nothing
    has been set up here yet.
    """
    marker_dir = splits_by_marker_dir(input_dir)
    if not marker_dir.exists():
        raise FileNotFoundError(
            f"Expected the per-marker Tests split at {marker_dir}, but it doesn't exist. Either run "
            f"python -m scripts.build_splits_by_marker --input-dir {input_dir}, or populate "
            f"{marker_dir} directly with one CSV per marker (matching TESTS_SCHEMA)."
        )
    frames = [load_tests_csv(marker_dir / f"{test_code}.csv") for test_code in test_codes if (marker_dir / f"{test_code}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL, SEX_COL])


def load_dx_csv(path: Union[str, Path], verbose: bool = True) -> pd.DataFrame:
    path = Path(path)
    if verbose:
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"reading {path} ({size_mb:,.0f} MB)...")
        t0 = time.time()

    df = _read_csv_no_default_na(path, {ID_COL: str, ICD9_COL: str, ICD10_COL: str})
    _validate_columns(df, DX_SCHEMA)
    df[DIAGNOSIS_TS_COL] = pd.to_datetime(df[DIAGNOSIS_TS_COL], errors="coerce")
    # icd9/icd10 are sparse by design (a row may populate only one) -- restore
    # NaN for the genuinely-blank cells now that the raw read is NA-string-safe.
    df[ICD9_COL] = df[ICD9_COL].replace("", pd.NA)
    df[ICD10_COL] = df[ICD10_COL].replace("", pd.NA)

    if verbose:
        print(f"read {path}: {len(df):,} rows in {time.time() - t0:.1f}s")
    return df


def load_demographics_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = _read_csv_no_default_na(path, {ID_COL: str, SEX_COL: str})
    _validate_columns(df, DEMOGRAPHICS_SCHEMA)
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["death_ts"] = pd.to_datetime(df["death_ts"], errors="coerce")
    df[SEX_COL] = df[SEX_COL].str.upper()
    return df.drop_duplicates(subset=[ID_COL])


def load_iron_mar_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = _read_csv_no_default_na(path, {ID_COL: str})
    if "taken_time" in df.columns and TS_COL not in df.columns:
        df = df.rename(columns={"taken_time": TS_COL})
    _validate_columns(df, IRON_MAR_SCHEMA)
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    return df.sort_values([ID_COL, TS_COL]).reset_index(drop=True)


def load_dx_incident(path: Union[str, Path]) -> pd.DataFrame:
    """Loads the derived Dx table produced by `scripts.build_dx_incident`.

    Raises error if dx_incident hasn't been run yet -- fig3_dx
    and fig4_dx_cases do not re-derive dx_incident themselves, both to avoid recomputing
    the ICD prefix matching twice and so the scripts can't silently drift apart.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected the derived Dx table at {path}, but it doesn't exist. Run dx_incident first: "
            f"python -m scripts.build_dx_incident --input-dir <input_dir> --output-dir {path.parent}"
        )
    dx_incident = pd.read_csv(path, dtype={ID_COL: str})
    dx_incident["earliest_contact_date"] = pd.to_datetime(dx_incident["earliest_contact_date"], errors="coerce")
    return dx_incident


def load_pregnancy_labs_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Loads a pregnancy_labs export into the exact same shape as load_tests_csv's output
    (anon_id, ts, test_code, result_value, sex) -- pregnancy patients are always female and
    the raw file has no sex column, so sex="F" is added directly. This lets the pregnancy
    analysis call utils.setpoints.compute_sp_df with zero special-casing.
    """
    df = _read_csv_no_default_na(path, {ID_COL: str, TEST_CODE_COL: str})
    _validate_columns(df, PREGNANCY_LABS_SCHEMA)
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    df[MEASUREMENT_COL] = pd.to_numeric(df[MEASUREMENT_COL], errors="coerce")
    df[SEX_COL] = "F"
    df = df.dropna(subset=[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL])
    return df.sort_values([ID_COL, TS_COL]).reset_index(drop=True)


def load_pregnancy_outcomes_and_demogs_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Loads a pregnancy_outcomes_and_demogs export: one row per pregnancy (anon_id may repeat).

    Derives conception_date = delivery_date - gestational_age (weeks); coerces rbc_tf/pih
    to clean 0/1 outcome flags (renamed to received_tf/pih). Unlike the real pipeline's
    schema, mother_dob/race/ethnic_group are not required here -- see
    PREGNANCY_OUTCOMES_AND_DEMOGS_SCHEMA's docstring for why they're dead weight.
    """
    df = _read_csv_no_default_na(path, {ID_COL: str})
    _validate_columns(df, PREGNANCY_OUTCOMES_AND_DEMOGS_SCHEMA)
    df = df.copy()
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    df["gestational_age"] = pd.to_numeric(df["gestational_age"], errors="coerce").fillna(40.0)
    df["conception_date"] = df["delivery_date"] - pd.to_timedelta(df["gestational_age"], unit="W")
    df[SEX_COL] = "F"
    df["received_tf"] = _coerce_binary(df["rbc_tf"])
    df["pih"] = _coerce_binary(df["pih"])
    df = df.dropna(subset=[ID_COL, "conception_date", "delivery_date"])
    return df.sort_values([ID_COL, "delivery_date"]).reset_index(drop=True)
