"""Marker -> battery (panel) grouping.

Used by fig3a/b's HR panels to facet plots by battery (CBC, BMP, ...).
"""

from typing import Optional

import pandas as pd

from constants.marker_config import BATTERY2TESTCODE
from constants.runtime import TEST_CODE_COL


def battery_for_test_code(test_code: str, default: Optional[str] = "Unknown") -> Optional[str]:
    """Map a test code to its battery label."""
    test_code = str(test_code)
    return next((battery for battery, test_codes in BATTERY2TESTCODE.items() if test_code in test_codes), default)


def add_battery_column(
    df: pd.DataFrame,
    *,
    test_code_col: str = TEST_CODE_COL,
    battery_col: str = "battery",
    default: Optional[str] = "Unknown",
) -> pd.DataFrame:
    """Return a copy with a battery column derived from test_code."""
    if test_code_col not in df.columns:
        raise KeyError(f"Missing test code column '{test_code_col}'.")

    out = df.copy()
    out[battery_col] = out[test_code_col].astype(str).map(lambda tc: battery_for_test_code(tc, default=default))
    return out
