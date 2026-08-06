"""Population reference-interval lookups.

popRI + attach_ref_intervals (used by add_oo and the fig3/progression
cohort-building code), plus compute_within_normal_mask (fig3's KM base
inputs use it as the "normal_filter" step, restricting to setpoints within
the population reference interval before percentile-cutoff grouping).
"""

import pandas as pd

from constants.marker_config import POP_REF_INTERVAL
from constants.runtime import MU, SEX_COL, TEST_CODE_COL


def popRI(sex="ALL", test_code="HB"):
    if sex is None or sex not in ["F", "M"]:
        sex = "ALL"
    return POP_REF_INTERVAL[sex][test_code]


def attach_ref_intervals(
    setpoints_df: pd.DataFrame,
    sex_col: str = SEX_COL,
    test_code_col: str = TEST_CODE_COL,
) -> pd.DataFrame:
    """Attach population reference intervals (ref_low, ref_high) for each (sex, test_code) pair."""
    df = setpoints_df.copy()

    if {"ref_low", "ref_high"}.issubset(df.columns):
        return df

    if sex_col not in df.columns or test_code_col not in df.columns:
        raise ValueError(f"attach_ref_intervals expects columns '{sex_col}' and '{test_code_col}' in setpoints_df")

    unique_pairs = df[[sex_col, test_code_col]].dropna().drop_duplicates()

    records = []
    for _, row in unique_pairs.iterrows():
        sex = row[sex_col]
        tc = row[test_code_col]
        ref_low, ref_high = popRI(sex=sex, test_code=tc)
        records.append(
            {
                sex_col: sex,
                test_code_col: tc,
                "ref_low": ref_low,
                "ref_high": ref_high,
            }
        )

    ref_intervals_df = pd.DataFrame(
        records,
        columns=[sex_col, test_code_col, "ref_low", "ref_high"],
    )
    df = df.merge(ref_intervals_df, on=[sex_col, test_code_col], how="left")
    return df


def compute_within_normal_mask(df: pd.DataFrame, mu_col: str = MU, low_col: str = "ref_low", high_col: str = "ref_high") -> pd.Series:
    """Boolean mask: True where mu_col falls within [low_col, high_col] (inclusive)."""
    required = {mu_col, low_col, high_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"compute_within_normal_mask is missing required columns: {missing}")
    return (df[mu_col] >= df[low_col]) & (df[mu_col] <= df[high_col])
