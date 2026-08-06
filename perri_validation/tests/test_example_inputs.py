"""Ensure every checked-in example input remains valid for its production loader."""

from pathlib import Path

from perri_validation.utils.io import (
    load_demographics_csv,
    load_dx_csv,
    load_iron_mar_csv,
    load_pregnancy_labs_csv,
    load_pregnancy_outcomes_and_demogs_csv,
    load_tests_csv,
)


EXAMPLES_DIR = Path(__file__).parents[1] / "data" / "examples"


def test_required_input_examples_load_successfully():
    tables = {
        "tests": load_tests_csv(EXAMPLES_DIR / "tests_example.csv"),
        "dx": load_dx_csv(EXAMPLES_DIR / "dx_example.csv"),
        "demographics": load_demographics_csv(EXAMPLES_DIR / "demographics_example.csv"),
        "iron_mar": load_iron_mar_csv(EXAMPLES_DIR / "iron_mar_example.csv"),
        "pregnancy_labs": load_pregnancy_labs_csv(EXAMPLES_DIR / "pregnancy_labs_example.csv"),
        "pregnancy_outcomes_and_demogs": load_pregnancy_outcomes_and_demogs_csv(
            EXAMPLES_DIR / "pregnancy_outcomes_and_demogs_example.csv"
        ),
    }

    assert all(not table.empty for table in tables.values())
    assert set(tables["tests"]["sex"]) == {"F", "M"}
    assert tables["demographics"]["death_ts"].isna().any()
    assert tables["iron_mar"].duplicated(["anon_id"], keep=False).any()
    assert set(tables["pregnancy_labs"]["sex"]) == {"F"}
    assert {"conception_date", "received_tf"}.issubset(tables["pregnancy_outcomes_and_demogs"].columns)
