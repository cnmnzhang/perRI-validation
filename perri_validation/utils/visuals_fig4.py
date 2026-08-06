"""Figure 4 plotting functions.

The four functions perri_validation/'s fig4_dx_cases mosaic needs:
plot_single_patient_history_on_ax, fig4km_exclusive_on_ax,
fig4heatmap_2x2_on_ax, fig4forest_on_ax.
"""

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

from perri_validation.utils.clinical.inputs import ProgressionPanelInputs
from perri_validation.utils.clinical.run_clinical import build_added_value_tables, build_reclassification_2x2_grids
from perri_validation.utils.clinical import get as _get
from perri_validation.constants.fig_config import (
    FIG3_ERROR_BAR_MARKERSIZE,
    FIG4_AXIS_LABEL_FONTSIZE,
    FIG4_CALLOUT_FONTSIZE,
    FIG4_HEATMAP_ANNOT_FONTSIZE,
    FIG4_LEGEND_FONTSIZE,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_TICK_LABEL,
    LINEWIDTH,
    MODEL2COLOR,
    PER_RI_ALPHA,
    REF_LINE_STYLE,
)
from perri_validation.constants.marker_config import TESTCODE_DISPLAY
from perri_validation.constants.runtime import (
    CV_COL,
    ID_COL,
    MEASUREMENT_COL,
    MU,
    OUT_PERRI_P95_COL,
    OUT_PERRI_P95_LOWER_COL,
    OUT_PERRI_P95_UPPER_COL,
    OUT_POPRI_COL,
    OUT_POPRI_LOWER_COL,
    OUT_POPRI_UPPER_COL,
    PRESENT_TS_COL,
    PRESENT_VAL_COL,
    SEX_COL,
    SIGMA,
    TS_COL,
)
from perri_validation.utils.km_exclusive import KMExclusiveInputs
from perri_validation.utils.visuals_shared import PROG_STYLING, year_since_baseline


def plot_single_patient_history_on_ax(
    ax: plt.Axes,
    patient_sp_data: pd.DataFrame,
    test_code: str,
    lower: bool,
    upper: bool,
    event_row: Optional[pd.DataFrame] = None,
    presenting_row: Optional[pd.DataFrame] = None,
    z_threshold: float = 1.96,
    add_legend: bool = True,
    presenting_ts_col: str = PRESENT_TS_COL,
    event_ts_col: str = "any_progression_event",
):
    patient_data = patient_sp_data.sort_values(by=TS_COL)
    all_dates = pd.to_datetime(patient_data[TS_COL])
    all_values = pd.to_numeric(patient_data[MEASUREMENT_COL])
    all_mus = pd.to_numeric(patient_data[MU])
    all_sigmas = pd.to_numeric(patient_data[SIGMA])

    sex = patient_sp_data[SEX_COL].iloc[0]
    pop_low, pop_high = _get.popRI(sex, test_code)

    years, years_est = year_since_baseline(all_dates)
    years = np.asarray(years)
    years_est = np.asarray(years_est)

    start = all_dates.iloc[0]

    presenting_y = None
    if presenting_row is not None and not presenting_row.empty:
        _pval = presenting_row[PRESENT_VAL_COL].iloc[0] if PRESENT_VAL_COL in presenting_row.columns else None
        if _pval is not None and pd.notna(_pval):
            presenting_y = float(_pval)
    del presenting_y  # kept for parity with source; not directly plotted (matches original)

    event_x = None
    if (event_row is not None) and (not event_row.empty) and pd.notna(event_row[event_ts_col].iloc[0]):
        event_ts = pd.to_datetime(event_row[event_ts_col].iloc[0])
        event_x = (event_ts - start) / pd.Timedelta(days=365.25)

    mus_prior = all_mus.shift(1)
    sigmas_prior = all_sigmas.shift(1)
    z_dynamic = (all_values - mus_prior) / sigmas_prior

    if lower and upper:
        oor_mask = z_dynamic.abs() > z_threshold
    elif lower:
        oor_mask = z_dynamic < -z_threshold
    elif upper:
        oor_mask = z_dynamic > z_threshold
    else:
        oor_mask = pd.Series(False, index=z_dynamic.index)

    if len(oor_mask) > 0:
        oor_mask.iloc[0] = False

    values_oor = all_values[oor_mask]
    years_oor = years[oor_mask.to_numpy()] if values_oor.size > 0 else np.array([])

    ax.axhspan(pop_low, pop_high, color="gray", alpha=0.3, label="PopRI")

    lower_band = np.maximum(all_mus - z_threshold * all_sigmas, 0)
    upper_band = all_mus + z_threshold * all_sigmas

    ax.fill_between(years_est, lower_band, upper_band, color=MODEL2COLOR["bayesian"], alpha=PER_RI_ALPHA, label=f"PerRI (z={z_threshold:.2f})")

    ax.plot(years_est, all_mus, color=MODEL2COLOR["bayesian"], lw=LINEWIDTH * 1.5, label="Setpoint")
    ax.plot(years, all_values, "o-", color="black", markersize=LINEWIDTH + 1, label="Labs")

    if values_oor.size > 0:
        ax.plot(years_oor, values_oor, "o", color="tab:red", markersize=(LINEWIDTH + 1), label="Out")

    if event_x is not None:
        x_float = float(event_x)
        ax.axvline(x=event_x, color="tab:red", linestyle="--", lw=LINEWIDTH)
        ax.annotate(
            "DX",
            xy=(x_float, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, 0),
            textcoords="offset points",
            rotation=0,
            ha="left",
            va="top",
            color="tab:red",
            fontsize=FIG4_CALLOUT_FONTSIZE,
            annotation_clip=False,
        )

    x_min = float(np.nanmin(years))
    x_max = float(np.nanmax(years_est))
    if event_x is not None:
        x_max = max(x_max, float(event_x))
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_xlabel("Year")

    y_min = max(0.0, float(min(all_values.min(), lower_band.min(), pop_low)))
    y_max = float(max(all_values.max(), upper_band.max(), pop_high))

    ax.set_ylim(0.9 * y_min, y_max * 1.1)
    ax.set_yticks(np.linspace(0.9 * y_min, y_max * 1.1, 5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel(TESTCODE_DISPLAY.get(test_code, test_code))

    if add_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=FIG4_LEGEND_FONTSIZE)


def fig4km_exclusive_on_ax(
    ax,
    inputs: KMExclusiveInputs,
    *,
    ylabel: str = "Patients without (%)",
    xlabel: str = "Years after presenting",
    add_legend: bool = True,
    legend_fontsize: float = FIG4_LEGEND_FONTSIZE,
):
    km_all = inputs.km_all
    masks = inputs.masks
    window_days = inputs.window_days
    mins = []

    for style_key, mask in masks.items():
        if not mask.any():
            continue
        sub = km_all.loc[mask]
        if sub.empty:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(sub["duration_days"], event_observed=sub["event_observed"])

        surv = kmf.survival_function_.copy()
        if 0.0 not in surv.index:
            surv.loc[0.0] = 1.0
        surv = surv.sort_index()
        if float(window_days) not in surv.index:
            surv = surv.reindex(surv.index.union([float(window_days)])).ffill()

        surv_vals = surv.iloc[:, 0].to_numpy()
        mins.append(float(surv_vals.min()))

        style_props = PROG_STYLING.get(style_key, {}).copy()
        label = style_props.pop("label", style_key)
        ax.step(surv.index, surv_vals, where="post", label=label, **style_props)

    tick_years = np.arange(0, window_days + 1, 365)
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(t / 365)) for t in tick_years])
    ax.set_xlim(0, window_days + 1)

    if mins:
        ax.set_ylim(min(mins), 1.0)
        ax.set_yticks(np.linspace(min(mins), 1.0, 4))
        ax.set_yticklabels([f"{y:.3f}" for y in ax.get_yticks()])

    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_ANNOTATION)
    if add_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=legend_fontsize)


def fig4heatmap_2x2_on_ax(
    ax,
    inputs: ProgressionPanelInputs,
    *,
    min_count_for_annotation: int = 20,
    annot_fontsize: float = FIG4_HEATMAP_ANNOT_FONTSIZE,
    annotate_n: bool = False,
) -> None:
    """Render a 2x2 reclassification heatmap onto an existing axes.

    Uses the canonical A/B/C/D reclassification cells from run_clinical.
    """
    lower = bool(inputs.outcome_cfg.flag_below)
    upper = bool(inputs.outcome_cfg.flag_above)
    if lower and not upper:
        abnormal_label, pop_col, per_col, direction = "Low", OUT_POPRI_LOWER_COL, OUT_PERRI_P95_LOWER_COL, "decrease"
    elif upper and not lower:
        abnormal_label, pop_col, per_col, direction = "High", OUT_POPRI_UPPER_COL, OUT_PERRI_P95_UPPER_COL, "increase"
    else:
        abnormal_label, pop_col, per_col, direction = "Abnormal", OUT_POPRI_COL, OUT_PERRI_P95_COL, "two_tailed"

    df = inputs.presenting_df
    outcome_col = "any_in_window"

    req_cols = [ID_COL, PRESENT_TS_COL, pop_col, per_col, outcome_col]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    tables = build_added_value_tables(df, pop_col=pop_col, per_col=per_col, outcome_col=outcome_col, id_col=ID_COL, ts_col=PRESENT_TS_COL, verbose=False)
    reclass = tables["reclassification"].copy()
    grids = build_reclassification_2x2_grids(reclass, abnormal_label=abnormal_label, direction=direction)
    rate_df = grids["rate_df"]
    count_df = grids["count_df"]
    annot_grid = np.empty((2, 2), dtype=object)
    for i, per_label in enumerate(rate_df.index):
        for j, pop_label in enumerate(rate_df.columns):
            n = int(count_df.loc[per_label, pop_label])
            rate = float(rate_df.loc[per_label, pop_label]) if n else np.nan
            if annotate_n:
                annot_grid[i, j] = f"{rate:.1f}%\nn={n:,}" if n > 0 and pd.notna(rate) else f"n={n:,}"
            else:
                annot_grid[i, j] = f"{rate:.1f}%" if n >= min_count_for_annotation and n > 0 else f"(n={n:,})"

    sns.heatmap(
        rate_df,
        annot=annot_grid,
        fmt="",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"size": annot_fontsize},
        vmin=0.0,
        vmax=max(1.0, rate_df.max().max()),
        ax=ax,
    )
    if ax.collections and ax.collections[0].colorbar:
        ax.collections[0].colorbar.remove()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)


def fig4forest_on_ax(
    ax: plt.Axes,
    model_results_dict: dict,
    *,
    fig4_axis_label_fontsize: float = FIG4_AXIS_LABEL_FONTSIZE,
):
    all_dfs = []
    for model_result in (model_results_dict or {}).values():
        if not model_result or ("or_table" not in model_result) or (model_result["or_table"] is None):
            continue
        all_dfs.append(model_result["or_table"].copy())

    if not all_dfs:
        ax.text(0.5, 0.5, "No valid model results", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    has_log = {"log_odds_ratio", "log_ci_lower", "log_ci_upper"}.issubset(combined_df.columns)
    has_raw = {"odds_ratio", "ci_lower", "ci_upper"}.issubset(combined_df.columns)

    if not has_raw:
        if not has_log:
            ax.text(0.5, 0.5, "Missing OR columns", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return pd.DataFrame()
        combined_df["odds_ratio"] = np.exp(pd.to_numeric(combined_df["log_odds_ratio"], errors="coerce"))
        combined_df["ci_lower"] = np.exp(pd.to_numeric(combined_df["log_ci_lower"], errors="coerce"))
        combined_df["ci_upper"] = np.exp(pd.to_numeric(combined_df["log_ci_upper"], errors="coerce"))

    for c in ["odds_ratio", "ci_lower", "ci_upper"]:
        combined_df[c] = pd.to_numeric(combined_df[c], errors="coerce")

    combined_df = combined_df.loc[
        (combined_df["odds_ratio"] > 0)
        & (combined_df["ci_lower"] > 0)
        & (combined_df["ci_upper"] > 0)
        & np.isfinite(combined_df["odds_ratio"])
        & np.isfinite(combined_df["ci_lower"])
        & np.isfinite(combined_df["ci_upper"])
    ].copy()

    if combined_df.empty:
        ax.text(0.5, 0.5, "No finite OR rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return combined_df

    if "feature" not in combined_df.columns:
        ax.text(0.5, 0.5, "Missing 'feature' column", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return pd.DataFrame()

    feature_order = ["presenting", MU, "delta", CV_COL]
    _exclude = {SEX_COL, "age_at_presenting"}
    present_feats = [f for f in feature_order if f in combined_df["feature"].unique().tolist()]
    extra_feats = sorted([f for f in combined_df["feature"].dropna().unique().tolist() if f not in present_feats and f not in _exclude])
    feature_order = present_feats + extra_feats

    feature2label = {
        "presenting": "Presenting",
        MU: "Setpoint",
        CV_COL: "CV",
        "delta": "Delta",
        SEX_COL: "Sex",
        "age_at_presenting": "Age",
    }

    base_x = np.arange(len(feature_order), dtype=float)
    color = MODEL2COLOR["bayesian"]

    for i, feature in enumerate(feature_order):
        r = combined_df[combined_df["feature"] == feature]
        if r.empty:
            continue
        row = r.iloc[0]
        or_val = float(row["odds_ratio"])
        ci_low = float(row["ci_lower"])
        ci_high = float(row["ci_upper"])
        ax.errorbar(
            x=base_x[i],
            y=or_val,
            yerr=[[or_val - ci_low], [ci_high - or_val]],
            fmt="o",
            color=color,
            markeredgewidth=0.0,
            markersize=FIG3_ERROR_BAR_MARKERSIZE,
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
        )

    ax.axhline(y=1.0, **REF_LINE_STYLE)
    ax.set_xticks(base_x)
    ax.set_xticklabels([feature2label.get(f, f) for f in feature_order], rotation=30, ha="right", fontsize=FONT_SIZE_TICK_LABEL)

    y_lo = min(float(np.nanmin(combined_df["ci_lower"])), 1.0)
    y_hi = max(float(np.nanmax(combined_df["ci_upper"])), 1.0)

    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks([0.5, 1.0, 2.0])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel("Odds Ratio", fontsize=fig4_axis_label_fontsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return combined_df
