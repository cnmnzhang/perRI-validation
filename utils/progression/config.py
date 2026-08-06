"""Fig4 progression outcome registry (aki, leukemia, hypothyroidism only)."""

from datetime import date

from utils.clinical.inputs import CohortConfig, OutcomeConfig, OutcomeDefinition

first_n = 3
OUTCOME_REGISTRY: dict[str, OutcomeConfig] = {
    "aki": OutcomeConfig(
        name="aki",
        markers=["CRE"],
        flag_below=False,
        flag_above=True,
        analysis_window_years=2,
        washout_years=1.0,
        cohort=CohortConfig(first_n=first_n),
        outcomes=[OutcomeDefinition(name="Acute kidney injury", type="diagnosis", diagnosis_name="Acute kidney injury")],
    ),
    "leukemia": OutcomeConfig(
        name="leukemia",
        markers=["WBC"],
        flag_below=True,
        flag_above=True,
        analysis_window_years=2,
        washout_years=1.0,
        presenting_min_year=date(2021, 6, 1),
        cohort=CohortConfig(year_cutoff=2020, first_n=None, min_points=3),
        outcomes=[OutcomeDefinition(name="leukemia_diagnosis", type="diagnosis", diagnosis_name="Leukemia")],
    ),
    "hypothyroidism": OutcomeConfig(
        name="hypothyroidism",
        markers=["TSH"],
        flag_below=False,
        flag_above=True,
        analysis_window_years=2,
        washout_years=1.0,
        cohort=CohortConfig(first_n=first_n),
        outcomes=[
            OutcomeDefinition(name="hypothyroidism", type="diagnosis", diagnosis_name="Hypothyroidism"),
            OutcomeDefinition(
                name="t4fr_low",
                type="lab_threshold",
                marker="T4FR",
                thresholds=(None, 0.6),
                exclude_prevalent=True,
            ),
        ],
    ),
}

__all__ = ["OUTCOME_REGISTRY"]
