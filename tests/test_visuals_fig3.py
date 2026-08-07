"""Tests for utils/visuals_fig3.py's marker ordering.

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

from utils.visuals_fig3 import battery_tests_in_ioi_order, fig3baseline_on_axes, fig3km


def test_battery_tests_in_ioi_order_orders_highest_ioi_first():
    df = pd.DataFrame({"test_code": ["HB", "GLU", "WBC"], "battery": ["CBC", "BMP", "CBC"]})
    # ioi_order is ascending by ioi (HB higher, WBC lower) -- display order is reversed, highest first.
    ordered = battery_tests_in_ioi_order(df, "CBC", ioi_order=["WBC", "HB"])
    assert ordered == ["WBC", "HB"]  # WBC is lower ioi, so appears second in display order


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


def test_fig3km_facets_follow_combined_order_descending_not_alphabetical():
    """Regression: fig3km used to sort facets alphabetically-descending on "diagnosis"
    (sort_values(["diagnosis", ...], ascending=[False, ...])), silently defeating
    COMBINED_ORDER -- run_fig3_dx.py's own comment already documented COMBINED_ORDER as the
    intended display order (descending, by explicit request). Picks 3 diagnoses (from
    COMBINED_ORDER: Cirrhosis, Neuropathy, Heart failure) whose plain alphabetical-descending
    order ("Neuropathy", "Heart failure", "Cirrhosis") differs from their descending-
    COMBINED_ORDER order ("Heart failure", "Neuropathy", "Cirrhosis") -- confirming the fix
    actually follows COMBINED_ORDER, not just accepting a coincidental match."""
    km_data = pd.DataFrame(
        [
            {"diagnosis": dx, "group": "< 25%", "timeline": t, "survival": 1.0, "setpoint_type": "HB", "count": 100}
            for dx in ["Neuropathy", "Heart failure", "Cirrhosis"]
            for t in [0, 1]
        ]
    )
    fig = fig3km(km_data)
    try:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        diagnoses_in_order = [t.split("\n")[0] for t in titles]
        assert diagnoses_in_order == ["Heart failure", "Neuropathy", "Cirrhosis"]
    finally:
        plt.close(fig)


def test_fig3km_retains_diagnosis_missing_from_combined_order():
    """A diagnosis present in the data but absent from COMBINED_ORDER must still render
    last (after the ordered ones), not be dropped -- same convention as
    battery_tests_in_ioi_order's fallback. Since the sort is descending, "last" means the
    fallback diagnosis must be placed *first* in the underlying category list (descending
    walks the category list back-to-front) -- this test would catch a naive re-introduction
    of "categories = ordered + fallback" the way an ascending sort would want it."""
    km_data = pd.DataFrame(
        [
            {"diagnosis": dx, "group": "< 25%", "timeline": t, "survival": 1.0, "setpoint_type": "HB", "count": 100}
            for dx in ["Cirrhosis", "Some Unlisted Diagnosis"]
            for t in [0, 1]
        ]
    )
    fig = fig3km(km_data)
    try:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        diagnoses_in_order = [t.split("\n")[0] for t in titles]
        assert diagnoses_in_order == ["Cirrhosis", "Some Unlisted Diagnosis"]
    finally:
        plt.close(fig)
