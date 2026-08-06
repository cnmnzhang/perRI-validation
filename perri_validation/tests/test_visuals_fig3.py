"""Tests for perri_validation/utils/visuals_fig3.py's marker ordering.

Regression coverage for a real bug: fig3baseline_on_axes used to convert
test_code to a pd.Categorical with categories=IOI_ORDER (a hardcoded 21-marker
list covering only CBC/BMP/WBC-diff), which silently turned every other
marker's test_code to NaN -- wiping out its data points in the rendered SVG
for every other battery (Hepatic/Lipid/Coag/Misc). Markers are now ordered
from constants/marker_lab_config.py via marker_config.MARKER_IOI_ORDER, passed
in explicitly as `ioi_order`, with no Categorical conversion.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from perri_validation.utils.visuals_fig3 import battery_tests_in_ioi_order, fig3baseline_on_axes


def test_battery_tests_in_ioi_order_orders_highest_ioi_first():
    df = pd.DataFrame({"test_code": ["HB", "GLU", "WBC"], "battery": ["CBC", "BMP", "CBC"]})
    # ioi_order is ascending by ioi (HB lower, WBC higher) -- display order is reversed, highest first.
    ordered = battery_tests_in_ioi_order(df, "CBC", ioi_order=["HB", "WBC"])
    assert ordered == ["WBC", "HB"]


def test_battery_tests_in_ioi_order_falls_back_to_alphabetical_for_unlisted_markers():
    """A marker present in the battery but absent from ioi_order (no configured IOI
    row) must still be included -- appended alphabetically, not dropped."""
    df = pd.DataFrame({"test_code": ["ALB", "ALT"], "battery": ["Hepatic", "Hepatic"]})
    ordered = battery_tests_in_ioi_order(df, "Hepatic", ioi_order=["ALT"])
    assert ordered == ["ALT", "ALB"]


def _hr_df():
    return pd.DataFrame(
        [
            {"test_code": "ALB", "battery": "Hepatic", "variable": "mu", "baseline_label": "1", "hr": 1.1, "ci_lower": 0.9, "ci_upper": 1.3},
            {"test_code": "HB", "battery": "CBC", "variable": "mu", "baseline_label": "1", "hr": 1.2, "ci_lower": 1.0, "ci_upper": 1.4},
        ]
    )


def test_fig3baseline_on_axes_retains_marker_missing_from_ioi_order():
    """Regression: with a small ioi_order that doesn't cover ALB (a Hepatic-battery
    marker), the old pd.Categorical(categories=IOI_ORDER) conversion NaN'd out ALB's
    test_code entirely, so it silently rendered with zero data points. It must now
    survive (and its battery panel receive its x-tick) instead of being dropped."""
    hr_df = _hr_df()
    fig, axes = plt.subplots(1, 2, squeeze=False)
    try:
        result = fig3baseline_on_axes(axes, hr_df, ioi_order=["HB"], variables=("mu",))
        assert not result["test_code"].isna().any()
        assert "ALB" in set(result["test_code"])
    finally:
        plt.close(fig)
