"""Pregnancy panel plotting (Task 1/2/3), vendored from scripts/figures/fig4_preg.py.

Task 1: biweekly quantile-band trajectory across gestation, with a pre-pregnancy
setpoint-distribution boxplot at x=-10. Task 2: trimester x {presenting value,
delta} odds-ratio forest plot. Task 3: trimester-1 PerRI x PopRI reclassification
heatmap. All three are matplotlib `_on_ax` functions so they compose into the
same kind of mosaic figure fig4_dx_cases/fig5_iron_infusion use.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from perri_validation.vendor.clinical.pregnancy import PairSpec, PREDICTOR_LABELS, TRIMESTER_LABELS
from perri_validation.vendor.constants.fig_config import FIG4_HEATMAP_ANNOT_FONTSIZE, FIG4_LEGEND_FONTSIZE, FIG4_TITLE_FONTSIZE, MODEL2COLOR, QUANTILE_BAND_INNER_ALPHA, QUANTILE_BAND_OUTER_ALPHA, REF_LINE_STYLE
from perri_validation.vendor.constants.marker_config import TESTCODE_DISPLAY
from perri_validation.vendor.constants.runtime import DELTA_COL, MEASUREMENT_COL


def plot_task1_on_ax(ax: plt.Axes, summary_df: pd.DataFrame, pair: PairSpec, setpoint_values: pd.Series, legend: bool = False, labelx: bool = False) -> None:
    if summary_df.empty:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(pair.title, fontsize=FIG4_TITLE_FONTSIZE)
        return

    x = summary_df["gestational_biweekly"].to_numpy(dtype=float)
    color = MODEL2COLOR["bayesian"]

    ax.fill_between(x, summary_df["q10"], summary_df["q90"], color=color, alpha=QUANTILE_BAND_OUTER_ALPHA, zorder=2, label="10-90th%")
    ax.fill_between(x, summary_df["q25"], summary_df["q75"], color=color, alpha=QUANTILE_BAND_INNER_ALPHA, zorder=3, label="25-75th%")
    ax.plot(x, summary_df["q50"], color="black", linewidth=1.5, label="Median", zorder=4)

    if setpoint_values is not None and not setpoint_values.empty:
        setpoint_x = -10
        ax.boxplot(
            setpoint_values.to_numpy(),
            positions=[setpoint_x],
            widths=2,
            patch_artist=True,
            boxprops=dict(facecolor=(*mcolors.to_rgb(color), QUANTILE_BAND_INNER_ALPHA), color=color),
            medianprops=dict(color="#000000", linewidth=1.5),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker="o", color=color, markersize=0.4, alpha=0.35),
            zorder=4,
        )
        ax.set_xlim(setpoint_x - 5, 40)
        xticks, xticklabels = [setpoint_x, 0, 13, 26, 39], ["Setpoint", "0", "13", "26", "39"]
    else:
        ax.set_xlim(0, 40)
        xticks, xticklabels = [0, 13, 26, 39], ["0", "13", "26", "39"]

    ax.set_xticks(xticks)
    if labelx:
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Week")
    ax.set_ylabel(TESTCODE_DISPLAY.get(pair.test_code, pair.test_code))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    if legend:
        ax.legend(fontsize=FIG4_LEGEND_FONTSIZE, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_task2_on_ax(ax: plt.Axes, results_df: pd.DataFrame, pair: PairSpec, legend: bool = False, labelx: bool = False) -> None:
    if results_df.empty:
        ax.text(0.5, 0.5, "No OR data", ha="center", va="center", transform=ax.transAxes)
        return

    predictor_order = [PREDICTOR_LABELS[MEASUREMENT_COL], PREDICTOR_LABELS[DELTA_COL]]
    offsets = {"t1": -0.15, "t2": 0.0, "t3": 0.15}
    colors = {"t1": "#becbf4", "t2": "#6e85df", "t3": "#3b4ccb"}

    for trimester in TRIMESTER_LABELS:
        sub = results_df[results_df["trimester"] == trimester].set_index("predictor")
        xvals, yvals, yerr_low, yerr_high = [], [], [], []
        for idx, predictor in enumerate(predictor_order):
            if predictor not in sub.index:
                continue
            est, lo, hi = sub.loc[predictor, "or"], sub.loc[predictor, "ci_lower"], sub.loc[predictor, "ci_upper"]
            if pd.isna(est) or pd.isna(lo) or pd.isna(hi) or float(est) <= 0 or float(lo) <= 0 or float(hi) <= 0:
                continue
            xvals.append(idx + offsets[trimester])
            yvals.append(float(est))
            yerr_low.append(float(est) - float(lo))
            yerr_high.append(float(hi) - float(est))
        if xvals:
            ax.errorbar(xvals, yvals, yerr=[yerr_low, yerr_high], fmt="o", markersize=3, capsize=2, capthick=1.0, linewidth=1.0, elinewidth=1.0, markeredgewidth=0.0, color=colors[trimester], label=trimester)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(predictor_order if labelx else [])
    ax.set_yscale("log")
    y_min = min(results_df["ci_lower"].min(), 1)
    y_max = max(results_df["ci_upper"].max(), 1)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([0.5, 1.0, 2.0])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel("Odds Ratio", fontsize=8)
    ax.axhline(y=1.0, **REF_LINE_STYLE)
    if legend:
        ax.legend(fontsize=FIG4_LEGEND_FONTSIZE, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncols=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_task3_on_ax(ax: plt.Axes, payload, pair: PairSpec, labelx: bool = False) -> None:
    if payload is None:
        ax.text(0.5, 0.5, "No T1 heatmap data", ha="center", va="center", transform=ax.transAxes)
        return

    sns.heatmap(
        payload["rate_df"],
        annot=payload["annot_grid"],
        fmt="",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"size": FIG4_HEATMAP_ANNOT_FONTSIZE},
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(payload["rate_df"].to_numpy())) * 1.1),
        ax=ax,
        cbar=False,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    if labelx:
        ax.set_title("Risk at T1", fontsize=FIG4_TITLE_FONTSIZE)
        ax.set_xlabel("PopRI")
    else:
        ax.set_title("", fontsize=FIG4_TITLE_FONTSIZE)
        ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
