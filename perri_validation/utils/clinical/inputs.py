"""Structured input dataclasses for clinical analysis."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class OutcomeDefinition:
    """Single outcome endpoint within a progression study.

    Attributes:
        name: Identifier used for column naming (e.g. "t4fr_low", "hypothyroidism").
        type: One of "diagnosis", "lab_threshold", "lab_delta", "order_exists".
        diagnosis_name: Diagnosis label matching dx_incident (required for type="diagnosis").
        marker: Test code for lab outcomes (required for type="lab_threshold"/"lab_delta").
        thresholds: (low, high) tuple; None on either side means unbounded.
            e.g. (None, 0.6) means value < 0.6; (1.2, None) means value > 1.2.
        delta: For lab_delta type — (operator, value) e.g. (">", 0.5).
        window: Custom window string for lab_delta type (e.g. "30-180d").
        exclude_prevalent: If True, patients with this outcome before the presenting
            date are removed from the cohort entirely.
    """

    name: str
    type: str
    diagnosis_name: Optional[str] = None
    marker: Optional[str] = None
    thresholds: Optional[Tuple[Optional[float], Optional[float]]] = None
    delta: Optional[Tuple[str, float]] = None
    window: Optional[str] = None
    exclude_prevalent: bool = False


@dataclass
class CohortConfig:
    """Cohort selection strategy for setpoint-based anchor identification.

    Filters are combinable — any non-None cutoff is applied to narrow the eligible pool,
    then first_n selects the anchor position within that pool.

    Attributes:
        first_n: Position of the anchor measurement (1-indexed) within the eligible pool.
            None means anchor at the last eligible measurement.
        year_cutoff: Restrict eligible measurements to those before January 1 of this year.
        age_cutoff: Restrict eligible measurements to those taken before the patient reached
            this age (in years).
        min_points: Minimum number of eligible measurements required.
    """

    first_n: Optional[int] = 3
    year_cutoff: Optional[int] = None
    age_cutoff: Optional[int] = None
    min_points: int = 0

    def to_dict(self) -> dict:
        """Backward-compatible dict for legacy callers."""
        d = {"first_n": self.first_n, "min_points": self.min_points}
        if self.year_cutoff is not None:
            d["year_cutoff"] = self.year_cutoff
        if self.age_cutoff is not None:
            d["age_cutoff"] = self.age_cutoff
        return d


@dataclass
class OutcomeConfig:
    """Full configuration for a progression study outcome.

    Temporal flow per patient:
        [anchor_ts (Nth measurement)]
            -> washout_years gap
        [presenting_ts (first raw measurement after washout)]
            -> grace_days buffer
        [outcome window: analysis_window_years]

    Attributes:
        name: Registry key (e.g. "hypothyroidism", "aki").
        markers: Test codes used to build the setpoint cohort (first entry is primary).
        flag_below: Flag patients whose value falls below their personal RI (depletion signal).
        flag_above: Flag patients whose value falls above their personal RI (elevation signal).
        analysis_window_years: Duration of the outcome observation window.
        washout_years: Gap between anchor and presenting measurement. Use 0 to
            take the next available measurement; use 1.0 for a 1-year washout.
        presenting_min_year: Hard floor on the presenting measurement date. Applied after
            washout.
        grace_days: Days after presenting_ts before the outcome window opens.
        cohort: CohortConfig defining anchor selection strategy.
        outcomes: List of OutcomeDefinition endpoints to flag.
    """

    name: str
    markers: List[str]
    outcomes: List[OutcomeDefinition]
    cohort: CohortConfig = field(default_factory=CohortConfig)
    flag_below: bool = True
    flag_above: bool = True
    analysis_window_years: float = 5.0
    washout_years: float = 0.0
    presenting_min_year: Optional[date] = None
    grace_days: int = 14
    z_optim_strategy: Optional[str] = None


@dataclass
class IronInfusionConfig:
    """Configuration for IV iron infusion cohort analysis.

    Defines temporal windows and thresholds for selecting pre- and post-treatment
    hemoglobin labs, grouping doses into courses, and filtering setpoint history.

    Attributes:
        pre_days_max: Days before course start to search for a pre-treatment lab (default: 60)
        post_days_min: Minimum days after course end for a post-treatment lab (default: 60)
        post_days_max: Maximum days after course end for a post-treatment lab (default: 180)
        gap_between_courses: Gap in days that defines a new course boundary (default: 60)
        setpoint_lookback_min: Minimum days before course start for setpoint history (default: 365)
        setpoint_lookback_max: Maximum days before course start for setpoint history (default: 1095)
        min_setpoint_measurements: Minimum isolated measurements required for a setpoint (default: 3)
        test_code: Marker analyzed (default: "HB")

    iron_mar.csv is required to already be filtered to the intended route/formulation
    (IV, iron sucrose) before it's provided -- see README.md's iron_mar note. This
    config has no iron_types/route filter because of that; it isn't re-filtered here.
    """

    pre_days_max: int = 60
    post_days_min: int = 60
    post_days_max: int = 180
    gap_between_courses: int = 60
    setpoint_lookback_min: int = 365
    setpoint_lookback_max: int = 1095
    min_setpoint_measurements: int = 3
    test_code: str = "HB"


@dataclass
class ProgressionPanelInputs:
    """Data bundle for rendering a single fig4 progression outcome panel block.

    Attributes:
        presenting_df: Analysis-ready cohort DataFrame with RI flags and outcome columns
        outcome_cfg: Outcome configuration; drives test_code, direction flags, and window
        sp_df: Setpoints DataFrame filtered to test_code, used for patient trajectory panel
    """

    presenting_df: pd.DataFrame
    outcome_cfg: "OutcomeConfig"
    sp_df: pd.DataFrame
