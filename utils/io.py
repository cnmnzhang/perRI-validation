"""Generic table loaders + schema validation for 's input tables."""

import time
from pathlib import Path
from typing import Union

import pandas as pd

from constants.runtime import ID_COL, MEASUREMENT_COL, SEX_COL, TEST_CODE_COL, TS_COL
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

    Several legitimate values in this schema collide with pandas' default
    na_values list -- most importantly the marker code "NA" (sodium), which
    pandas otherwise silently reads as a missing value (then a naive dropna
    on that column deletes every sodium row). Callers that need NaN handling
    for numeric/date columns already coerce explicitly afterward
    (pd.to_numeric/pd.to_datetime with errors="coerce"), so disabling the
    global sniffing is safe.
    """
    return pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values=[])


_GENERIC_SOURCE_FILE = "all_tests_cbc_bmp_merged.pkl"


def _drop_redundant_generic_source_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Some markers (e.g. the Hepatic panel, TSH) are captured by more than one raw
    export -- once as a battery-specific merged file, and again because the same real
    draw is also bundled into the generic CBC/BMP export. bayesian-setpoint-inference's
    own loader (utils/data_preprocess.py:load_marker) never has this problem: it's an
    if/elif chain that reads exactly one canonical source file per marker (specialty
    battery file if the marker has one, the generic CBC/BMP file otherwise) -- it never
    merges sources for a single marker.

    Without this, a marker present in two sources gets each real encounter recorded
    twice, a few hours apart (one timestamp per source) rather than once -- which
    isn't caught by the exact-duplicate drop below (different timestamps, same draw)
    and silently defeats the 90-day isolation filter: each duplicate is "too close" to
    its own twin, so both get excluded as non-isolated even though the real encounter
    is isolated. Confirmed on real data: one patient's 16 genuinely isolated ALB
    encounters over a decade collapsed to 3 isolated points with the duplicate present,
    and recovered to 16 once the duplicate row was dropped.

    Mirrors load_marker's preference (specialty source over the generic default) by
    dropping the generic-source rows for any test_code that also has rows from a
    non-generic source -- without hardcoding which markers or specialty filenames are
    affected, so this stays correct if the raw exports' marker coverage changes.
    """
    if "source_file" not in df.columns:
        return df
    has_specialty_source = df.groupby(TEST_CODE_COL)["source_file"].transform(lambda s: (s != _GENERIC_SOURCE_FILE).any())
    is_redundant_generic_row = has_specialty_source & (df["source_file"] == _GENERIC_SOURCE_FILE)
    dropped = int(is_redundant_generic_row.sum())
    if dropped:
        print(f"[io] dropping {dropped:,} rows from '{_GENERIC_SOURCE_FILE}' for markers that also have a specialty source (see _drop_redundant_generic_source_rows)")
    return df[~is_redundant_generic_row]


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
        print(f"[io] reading {path} ({size_mb:,.0f} MB)...")
        t0 = time.time()

    df = _read_csv_no_default_na(path, {ID_COL: str, TEST_CODE_COL: str, SEX_COL: str})
    _validate_columns(df, TESTS_SCHEMA)
    df = _drop_redundant_generic_source_rows(df)
    # A blank test_code is the sodium marker "NA" -- some upstream export step read it
    # with pandas' default NA-string sniffing (the same trap _read_csv_no_default_na
    # guards against here), turned it into a real NaN, and wrote that back out as an
    # empty cell. Restored here so every downstream split/fit sees "NA" like any other
    # marker, rather than losing the marker's rows or grouping them under "".
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
    # Exact-duplicate rows (same patient/timestamp/marker/value) are a real upstream export
    # artifact, not repeat measurements -- some markers (TSH/T4FR) are ~85% duplicated this
    # way, which silently defeats perri's isolation filter (every row has a same-timestamp
    # "neighbor," so almost nothing reads as isolated regardless of how spread out the
    # patient's real visits are). Dropped here so every downstream split/fit sees each real
    # measurement once.
    df = df.drop_duplicates(subset=[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL])
    df = df.sort_values([ID_COL, TS_COL]).reset_index(drop=True)

    if verbose:
        print(f"[io] read {path}: {len(df):,} rows in {time.time() - t0:.1f}s")
    return df


def tests_by_marker_dir(input_dir: Union[str, Path]) -> Path:
    return Path(input_dir) / "cache" / "tests_by_marker"


def load_tests_marker_subset(input_dir: Union[str, Path], test_codes: list) -> pd.DataFrame:
    """Loads just `test_codes` from the split built by scripts/run_tests_by_marker.py.

    Unlike the earlier design, this never builds the split itself -- it's a pure
    loader. Raises a clear FileNotFoundError naming the command to run if
    `scripts.run_tests_by_marker` hasn't been run yet, the same way
    load_dx_incident does for dx_incident.csv.
    """
    marker_dir = tests_by_marker_dir(input_dir)
    sentinel_path = marker_dir / "_split_complete.json"
    if not sentinel_path.exists():
        raise FileNotFoundError(
            f"Expected the per-marker Tests split at {marker_dir}, but it doesn't exist. Run it first: "
            f"python -m scripts.run_tests_by_marker --input-dir {input_dir}"
        )
    frames = [load_tests_csv(marker_dir / f"{test_code}.csv") for test_code in test_codes if (marker_dir / f"{test_code}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL, SEX_COL])


def load_dx_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = _read_csv_no_default_na(path, {ID_COL: str, "icd9": str, "icd10": str})
    _validate_columns(df, DX_SCHEMA)
    # dx_all_to_first_fast expects a "diagnosis_ts" column name
    df = df.rename(columns={"date": "diagnosis_ts"})
    df["diagnosis_ts"] = pd.to_datetime(df["diagnosis_ts"], errors="coerce")
    # icd9/icd10 are sparse by design (a row may populate only one) -- restore
    # NaN for the genuinely-blank cells now that the raw read is NA-string-safe.
    df["icd9"] = df["icd9"].replace("", pd.NA)
    df["icd10"] = df["icd10"].replace("", pd.NA)
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
    """Loads the derived Dx table produced by `scripts.run_dx_incident`.

    Raises a clear, actionable error if dx_incident hasn't been run yet -- fig3_dx
    and fig4_dx_cases do not re-derive dx_incident themselves, both to avoid recomputing
    the ICD prefix matching twice and so the scripts can't silently drift apart.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected the derived Dx table at {path}, but it doesn't exist. Run dx_incident first: "
            f"python -m scripts.run_dx_incident --input-dir <input_dir> --output-dir {path.parent}"
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
