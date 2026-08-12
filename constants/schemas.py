"""Required-column schemas for perri_validation's generic input tables.

Canonical column names come from constants/runtime.py so schema declarations and
the loaders that consume them cannot silently drift apart.
"""

from dataclasses import dataclass

from constants.runtime import DIAGNOSIS_TS_COL, ICD9_COL, ICD10_COL, ID_COL, MEASUREMENT_COL, SEX_COL, TEST_CODE_COL, TS_COL


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: tuple
    notes: str = ""


TESTS_SCHEMA = TableSchema(
    name="Tests",
    required_columns=(ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL, SEX_COL),
    notes="Non-isolated lab results, one row per measurement.",
)

DX_SCHEMA = TableSchema(
    name="Dx",
    required_columns=(ID_COL, ICD9_COL, ICD10_COL, DIAGNOSIS_TS_COL),
    notes="One row per diagnosis event. icd9/icd10 may be sparse per row (only one need be populated); codes are matched by longest-prefix, dots/case-insensitive.",
)

DEMOGRAPHICS_SCHEMA = TableSchema(
    name="Demographics",
    required_columns=(ID_COL, SEX_COL, "birth_date", "death_ts"),
    notes="One row per patient. `death_ts` is nullable (still-living patients).",
)

IRON_MAR_SCHEMA = TableSchema(
    name="iron_mar",
    required_columns=(ID_COL, TS_COL),
    notes="Medication administration record rows for iron formulations -- must already be "
    "pre-filtered to the intended route and formulation (IV, iron sucrose) before being "
    "provided.",
)

PREGNANCY_LABS_SCHEMA = TableSchema(
    name="pregnancy_labs",
    required_columns=(ID_COL, TS_COL, TEST_CODE_COL, MEASUREMENT_COL),
    notes="One row per lab measurement (pre-pregnancy and in-pregnancy). Only WBC/HCT rows are "
    "used; other CBC-panel markers present in the file are ignored. Uses the same `anon_id`/`ts` "
    "column names as Tests -- adds `sex='F'` internally (pregnancy patients are always female "
    "and the file has no sex column of its own).",
)

PREGNANCY_OUTCOMES_AND_DEMOGS_SCHEMA = TableSchema(
    name="pregnancy_outcomes_and_demogs",
    required_columns=(ID_COL, "delivery_date", "gestational_age", "rbc_tf", "pih"),
    notes="One row per pregnancy (an anon_id may repeat across pregnancies). `gestational_age` "
    "is delivery-time gestational age in weeks; conception_date is derived as "
    "delivery_date - gestational_age. `rbc_tf`/`pih` are coerced to 0/1 outcome flags.",
)

ALL_SCHEMAS = [
    TESTS_SCHEMA,
    DX_SCHEMA,
    DEMOGRAPHICS_SCHEMA,
    IRON_MAR_SCHEMA,
    PREGNANCY_LABS_SCHEMA,
    PREGNANCY_OUTCOMES_AND_DEMOGS_SCHEMA,
]
