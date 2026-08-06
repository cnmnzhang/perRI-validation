"""Tests for utils/clinical/icd.py's dx_all_to_first_fast.

Regression coverage for a real bug class this project hit: a separate
maintainer script (scripts/diagnostics/build_validation_test_data.py)
stripped periods from the *raw diagnosis codes* before matching but not from
the DX2ICD-derived *prefixes* being matched against, so every diagnosis whose
only codes contain a period (Cirrhosis "571.2", NAFLD "571.8", Neuropathy
"356.9") silently matched zero rows. dx_all_to_first_fast itself never had
this bug (_build_prefix_layers strips dots from both sides symmetrically),
but these tests lock in that guarantee for exactly the diagnoses that broke.
"""

from pathlib import Path

import pandas as pd

from utils.clinical.icd import dx_all_to_first_fast
from constants.icd_config import DX2ICD
from utils.io import load_dx_csv


DX_EXAMPLE_PATH = Path(__file__).parents[1] / "data" / "examples" / "dx_example.csv"


def _dx_row(anon_id, icd9="", icd10="", date="2020-01-01"):
    return {"anon_id": anon_id, "icd9": icd9, "icd10": icd10, "diagnosis_ts": date}


def test_dx_example_covers_realistic_rows_and_maps_end_to_end():
    dx_all = load_dx_csv(DX_EXAMPLE_PATH)

    assert len(dx_all) == 17
    assert dx_all["anon_id"].nunique() == 6
    assert (dx_all["icd9"].notna() & dx_all["icd10"].notna()).sum() == 15
    assert (dx_all["icd9"].isna() ^ dx_all["icd10"].isna()).sum() == 2
    assert dx_all.duplicated(["anon_id", "diagnosis_ts"], keep=False).sum() == 10

    dx_incident = dx_all_to_first_fast(dx_all)
    p001 = dx_incident[dx_incident["anon_id"] == "p001"].set_index("diagnosis_name")
    assert p001.loc["Hypothyroidism", "earliest_contact_date"] == pd.Timestamp("2016-03-14")
    assert p001.loc["Chronic kidney disease", "earliest_contact_date"] == pd.Timestamp("2016-03-14")
    assert p001.loc["Acute kidney injury", "earliest_contact_date"] == pd.Timestamp("2020-01-05")

    # The same-date hypertension row is realistic input but outside DX2ICD's
    # analysis diagnoses, so it must not create an incident diagnosis.
    assert "p006" in set(dx_incident["anon_id"])
    assert "Hypertension" not in set(dx_incident["diagnosis_name"].astype(str))


def test_matches_period_containing_icd9_codes():
    dx_all = pd.DataFrame(
        [
            _dx_row("p1", icd9="571.2"),  # Cirrhosis, exact code
            _dx_row("p2", icd9="571.8"),  # NAFLD, exact code
            _dx_row("p3", icd9="356.9"),  # Neuropathy, exact code
        ]
    )
    dx_incident = dx_all_to_first_fast(dx_all, id_col="anon_id")

    got = dict(zip(dx_incident["anon_id"], dx_incident["diagnosis_name"]))
    assert got["p1"] == "Cirrhosis"
    assert got["p2"] == "Nonalcoholic fatty liver disease"
    assert got["p3"] == "Neuropathy"


def test_matches_period_containing_icd10_codes():
    dx_all = pd.DataFrame(
        [
            _dx_row("p1", icd10="K74.3"),  # Cirrhosis
            _dx_row("p2", icd10="K76.0"),  # NAFLD
            _dx_row("p3", icd10="G62.9"),  # Neuropathy
        ]
    )
    dx_incident = dx_all_to_first_fast(dx_all, id_col="anon_id")

    got = dict(zip(dx_incident["anon_id"], dx_incident["diagnosis_name"]))
    assert got["p1"] == "Cirrhosis"
    assert got["p2"] == "Nonalcoholic fatty liver disease"
    assert got["p3"] == "Neuropathy"


def test_matches_dot_free_raw_codes_against_dotted_config():
    """Real EHR extracts often store codes without dots (e.g. "5712" for "571.2") --
    DX2ICD's config always uses dots, so the matcher must normalize both sides."""
    dx_all = pd.DataFrame([_dx_row("p1", icd9="5712")])
    dx_incident = dx_all_to_first_fast(dx_all, id_col="anon_id")
    assert dx_incident["diagnosis_name"].iloc[0] == "Cirrhosis"


def test_earliest_date_per_patient_diagnosis():
    dx_all = pd.DataFrame(
        [
            _dx_row("p1", icd9="280", date="2021-06-01"),
            _dx_row("p1", icd9="280", date="2020-01-01"),  # earlier -- should win
        ]
    )
    dx_incident = dx_all_to_first_fast(dx_all, id_col="anon_id")
    assert dx_incident["diagnosis_name"].iloc[0] == "Iron Deficiency anemia"
    assert dx_incident["earliest_contact_date"].iloc[0] == pd.Timestamp("2020-01-01")


def test_unmatched_code_produces_no_row():
    dx_all = pd.DataFrame([_dx_row("p1", icd9="999.99")])
    dx_incident = dx_all_to_first_fast(dx_all, id_col="anon_id")
    assert dx_incident.empty


def test_every_dx2icd_code_is_individually_matchable():
    """Broad sweep: every ICD9/ICD10 code in the config matches at least one
    diagnosis (some codes are legitimately shared across diagnosis names, so
    this checks "matched something", not "matched this exact name")."""
    rows = []
    for name, codes in DX2ICD.items():
        for code in codes.get("ICD9", []):
            rows.append(_dx_row(f"icd9_{name}_{code}", icd9=code))
        for code in codes.get("ICD10", []):
            rows.append(_dx_row(f"icd10_{name}_{code}", icd10=code))

    dx_all = pd.DataFrame(rows)
    dx_incident = dx_all_to_first_fast(dx_all, id_col="anon_id")
    matched_ids = set(dx_incident["anon_id"])

    unmatched = [row["anon_id"] for row in rows if row["anon_id"] not in matched_ids]
    assert not unmatched, f"codes matched nothing: {unmatched}"
