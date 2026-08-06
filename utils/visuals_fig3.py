"""Fig3 panel plotting. 
"""

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from utils.clinical.battery import add_battery_column, battery_for_test_code
from constants.fig_config import A4_WIDTH, FONT_SIZE_PANEL_TITLE, FONT_SIZE_TICK_LABEL, MODEL2COLOR, REF_LINE_STYLE
from constants.marker_config import BATTERY2TESTCODE, TESTCODE_DISPLAY
from constants.runtime import CV_COL, MU, TEST_CODE_COL
from utils.visuals_shared import FIG3_CAPSIZE, FIG3_ERROR_BAR_MARKERSIZE, OFFSET, _snap_yticks_to_ylim, format_battery_label, save_fig_as_svg


def fig3km(
    input_df: pd.DataFrame,
    observation_window: Optional[int] = None,
    col_wrap: int = 4,
    axes=None,
    save_path=None,
    figsize=None,
    csv_path=None,
) -> Optional[plt.Figure]:
    """Plot KM survival curves faceted by diagnosis + setpoint_type.

    Parameters
    ----------
    input_df : pd.DataFrame
        Columns: ["diagnosis", "group", "timeline", "survival", "setpoint_type"].
        Optional "count" column for N in facet titles.
    observation_window : int, optional
        Truncate timeline to this many years.
    col_wrap : int
        Max columns in the subplot grid.
    axes : array-like of Axes, optional
        Pre-created axes for mosaic composition. When provided the function
        plots into them and returns None. When None, a new figure is created
        and returned.
    save_path : Path, optional
        Save the standalone figure as SVG to this path (only when axes is None).
    """
    plot_df = input_df.copy()

    if observation_window is not None:
        plot_df = plot_df[plot_df["timeline"] <= observation_window].copy()
    print(f"[visuals_fig3] Observation window: {observation_window} years; timeline max={plot_df['timeline'].max()} years")
    # Build facet titles
    plot_df["setpoint_type_display"] = plot_df["setpoint_type"].map(lambda t: TESTCODE_DISPLAY.get(t, t))
    if "count" in plot_df.columns:
        plot_df["facet_title"] = plot_df["diagnosis"].astype(str) + "\n" + plot_df["setpoint_type_display"].astype(str)
    else:
        n_per_facet = plot_df[plot_df["timeline"] == 0].groupby(["diagnosis", "setpoint_type"]).size().rename("N").reset_index()
        plot_df = plot_df.merge(n_per_facet, on=["diagnosis", "setpoint_type"], how="left")
        plot_df["N"] = plot_df["N"].fillna(0).astype(int)
        plot_df["facet_title"] = plot_df["diagnosis"].astype(str) + " (" + plot_df["setpoint_type_display"].astype(str) + ")\nN=" + plot_df["N"].astype(str)

    plot_df.sort_values(["diagnosis", "group", "timeline"], ascending=[False, True, True], inplace=True)

    palette = {
        "< 25%": "#C43C39",  # muted red
        ">= 25%": "#3B5CCB",  # lighter Bayesian blue
        "< 75%": "#3B5CCB",  # Bayesian blue
        ">= 75%": "#D98A24",  # muted orange
    }

    facet_groups = list(plot_df.groupby("facet_title", sort=False))
    n_facets = len(facet_groups)
    ncols = min(col_wrap, n_facets)

    if axes is None:
        nrows = int(np.ceil(n_facets / ncols))
        fig, axes_arr = plt.subplots(
            nrows,
            ncols,
            figsize=figsize or (A4_WIDTH, nrows * 1.5),
            squeeze=False,
            sharex=False,  # each panel keeps its own x-limit (event counts differ per diagnosis)
            gridspec_kw={"hspace": 0.6, "wspace": 0.3},
        )
        flat_axes = list(axes_arr.flat)
        standalone = True
    else:
        flat_axes = list(np.asarray(axes, dtype=object).reshape(-1))
        fig = None
        standalone = False

    for idx, ((facet_title, subdf), ax) in enumerate(zip(facet_groups, flat_axes)):
        for grp, sdf in subdf.groupby("group", sort=False):
            sns.lineplot(data=sdf, x="timeline", y="survival", ax=ax, label=str(grp), linewidth=1.2, color=palette.get(grp, None))

        ax.set_title(facet_title, size=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        ax.legend_ = None

        is_rightmost = ((idx + 1) % ncols == 0) or (idx == n_facets - 1)
        if is_rightmost:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="lower left", frameon=False)

    if standalone:
        for ax in flat_axes[n_facets:]:
            ax.set_visible(False)
    else:
        for ax in flat_axes[n_facets:]:
            ax.set_axis_off()

    plt.draw()

    for (facet_title, subdf), ax in zip(facet_groups, flat_axes[:n_facets]):
        min_y = float(subdf["survival"].min()) if len(subdf) else 0.95
        ax.set_ylim(min_y, 1.0)
        ax.set_yticks(np.linspace(min_y, 1.0, 4))
        ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()], fontsize=FONT_SIZE_TICK_LABEL)
        x_max = observation_window if observation_window is not None else (float(subdf["timeline"].max()) if len(subdf) else 0.0)
        ax.set_xlim(0, x_max)

    if standalone:
        # Suppress x-tick labels on all but the bottom non-empty panel in each column,
        # simulating sharex='col' without forcing a shared x-limit across different windows.
        for col_idx in range(ncols):
            col_rows = [r for r in range(nrows) if r * ncols + col_idx < n_facets]
            for r in col_rows[:-1]:
                flat_axes[r * ncols + col_idx].tick_params(labelbottom=False)

    if standalone:
        fig.subplots_adjust(bottom=0.2)
        fig.supxlabel("Years since setpoint estimate", fontsize=8)
        fig.supylabel("Patients without (%)", fontsize=8)

        if save_path is not None:
            save_fig_as_svg(fig, "fig3_km", save_path, csv_path=csv_path)

        return fig


# ---------------------------------------------------------------------------
# fig3a: HR by model, fig3b: HR by baseline index
# ---------------------------------------------------------------------------


def cox_dict_to_hr_df(test_code_analysis_dict: dict, variables=(MU, CV_COL), model_list=None) -> pd.DataFrame:
    """Convert a {(test_code, model): {variable: coxph_summary}} dict to a plotting DataFrame."""
    records = []
    for (test_code, model), analysis_dict in test_code_analysis_dict.items():
        if model_list and model not in model_list:
            continue
        battery = battery_for_test_code(test_code, default=None)
        if battery is None:
            continue
        base_summary_dict = analysis_dict.get("coxph_summary", analysis_dict)
        for variable in variables:
            summary = base_summary_dict.get(variable, {})
            if not summary:
                continue
            records.append(
                {
                    TEST_CODE_COL: test_code,
                    "battery": battery,
                    "model": model,
                    "variable": variable,
                    "hr": summary["exp(coef)"],
                    "err_lower": summary["exp(coef)"] - summary["exp(coef) lower 95%"],
                    "err_upper": summary["exp(coef) upper 95%"] - summary["exp(coef)"],
                }
            )
    if not records:
        print(f"No valid CoxPH summaries found for variables={variables}")
        return pd.DataFrame()
    return pd.DataFrame(records)


def battery_tests_in_ioi_order(df: pd.DataFrame, battery: str, ioi_order: list, test_col: str = TEST_CODE_COL) -> list:
    """Return tests in a battery, lowest-`ioi` first (see marker_config.MARKER_IOI_ORDER).
    Markers present in the data but missing from `ioi_order` (no configured IOI)
    fall back to alphabetical order, appended after the ordered ones."""
    if "battery" not in df.columns:
        df = add_battery_column(df, default=None)
    present = set(df.loc[df["battery"] == battery, test_col].dropna().astype(str).tolist())
    ordered = [tc for tc in ioi_order if tc in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def fig3hr_on_axes(axes, test_code_analysis_dict: pd.DataFrame, ioi_order: list, variables=(MU, CV_COL), add_legend: bool = True) -> pd.DataFrame:
    """Draw Figure 3A hazard-ratio panel (rows=variables, cols=batteries, hue=model) on pre-created axes."""
    df = test_code_analysis_dict.copy()
    if "battery" not in df.columns:
        df = add_battery_column(df, default=None)
    df = df[df["variable"].isin(variables)].copy()
    if df.empty:
        return pd.DataFrame()

    variables = [v for v in variables if v in df["variable"].unique()]
    batteries = [b for b in BATTERY2TESTCODE.keys() if b in set(df["battery"].dropna().unique())]
    n_rows_data, n_cols_data = len(variables), len(batteries)

    axes_arr = np.asarray(axes, dtype=object)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape(1, -1)
    n_rows_avail, n_cols_avail = axes_arr.shape
    for i in range(n_rows_avail):
        for j in range(n_cols_avail):
            if i >= n_rows_data or j >= n_cols_data:
                axes_arr[i, j].set_axis_off()

    variable2label = {"mu": "Setpoint", "cv": "CV"}
    preferred_model_order = ["bayesian", "gmm"]
    observed_models = df["model"].dropna().astype(str).unique().tolist()
    model_order = [m for m in preferred_model_order if m in observed_models]
    model_order += [m for m in observed_models if m not in model_order]
    n_models = len(model_order)
    offsets = np.linspace(-OFFSET * n_models, OFFSET * n_models, n_models) if n_models > 1 else np.array([0.0])
    model_offsets = dict(zip(model_order, offsets))
    legend_handles_map = {}

    for i, variable in enumerate(variables):
        if i >= n_rows_avail:
            break
        var_df = df[df["variable"] == variable].copy()

        for j, battery in enumerate(batteries):
            if j >= n_cols_avail:
                break
            ax = axes_arr[i, j]
            cell_df = var_df[var_df["battery"] == battery].copy()
            battery_tests = battery_tests_in_ioi_order(df, battery, ioi_order)
            x_map = {tc: idx for idx, tc in enumerate(battery_tests)}

            for tc in battery_tests:
                tc_df = cell_df[cell_df[TEST_CODE_COL].astype(str) == tc]
                x_center = x_map[tc]
                for model in model_order:
                    rows = tc_df[tc_df["model"].astype(str) == model]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    x_pos = x_center + model_offsets[model]
                    ax.errorbar(
                        x=x_pos,
                        y=row["hr"],
                        yerr=[[row["err_lower"]], [row["err_upper"]]],
                        fmt="o",
                        color=MODEL2COLOR.get(model, "gray"),
                        ecolor=MODEL2COLOR.get(model, "gray"),
                        markeredgecolor=MODEL2COLOR.get(model, "gray"),
                        markeredgewidth=0.0,
                        markersize=FIG3_ERROR_BAR_MARKERSIZE,
                        elinewidth=1.0,
                        capsize=2.0,
                        capthick=1.0,
                        alpha=0.8,
                        clip_on=True,
                    )
                    if model not in legend_handles_map:
                        legend_handles_map[model] = ax.plot([], [], marker="o", linestyle="", color=MODEL2COLOR.get(model, "gray"), markersize=FIG3_ERROR_BAR_MARKERSIZE)[0]

            ax.axhline(1.0, **REF_LINE_STYLE)
            ax.set_xticks(range(len(battery_tests)))
            ax.tick_params(axis="x", which="both", bottom=True)
            ax.spines["bottom"].set_visible(True)

            if i == min(n_rows_data, n_rows_avail) - 1:
                ax.set_xticklabels([TESTCODE_DISPLAY.get(t, t) for t in battery_tests], rotation=45, ha="center", fontsize=5)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(variable2label.get(variable, str(variable).upper()), rotation=90)
            else:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)

            if i == 0:
                ax.set_title(format_battery_label(battery).upper(), fontsize=FONT_SIZE_PANEL_TITLE)
            ax.set_yticks([0.5, 1.0, 1.5, 2.0])
            ax.set_ylim(0.5, 2.0)
            ax.grid(False)
        _snap_yticks_to_ylim(axes_arr[i, 0])

    if add_legend and legend_handles_map and n_cols_avail > 0:
        handles = [legend_handles_map[m] for m in model_order if m in legend_handles_map]
        labels = [m for m in model_order if m in legend_handles_map]
        axes_arr[0, 0].legend(handles, labels, loc="upper left", ncol=1, frameon=False, borderaxespad=0.0, handletextpad=0.1)
    return df


def fig3hr(test_code_analysis_dict: dict, ioi_order: list, variables=(MU, CV_COL), model_list=None, save_path=None, figsize=None, csv_path=None) -> Optional[plt.Figure]:
    """Figure 3A: hazard ratios for mu/cv, rows=variables, cols=batteries, hue=model."""
    df = cox_dict_to_hr_df(test_code_analysis_dict, variables=variables, model_list=model_list)
    if df.empty:
        return None
    variables = [v for v in variables if v in df["variable"].unique()]
    batteries = [b for b in BATTERY2TESTCODE.keys() if b in set(df["battery"].dropna().unique())]
    width_ratios = [max(df[df["battery"] == b][TEST_CODE_COL].nunique(), 1) for b in batteries]
    n_rows, n_cols = len(variables), len(batteries)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize or (A4_WIDTH, n_rows),
        sharex=False,
        sharey="row",
        gridspec_kw={"width_ratios": width_ratios, "hspace": 0.25, "wspace": 0.2},
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.atleast_2d(axes)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    fig3hr_on_axes(axes, df, ioi_order, variables=variables, add_legend=True)
    if n_rows > 0 and n_cols > 0:
        fig.align_ylabels([axes[i, 0] for i in range(n_rows)])
        fig.align_xlabels([axes[n_rows - 1, j] for j in range(n_cols)])
    if save_path is not None:
        save_fig_as_svg(fig, "fig3a_hr_by_model", save_path, csv_path=csv_path)
    return fig


def fig3baseline_on_axes(axes, hr_df: pd.DataFrame, ioi_order: list, variables=(MU, CV_COL), add_legend: bool = True) -> pd.DataFrame:
    """Draw Figure 3B baseline-HR panel (rows=variables, cols=batteries, hue=baseline index) on pre-created axes."""
    df = hr_df.copy()
    if "battery" not in df.columns:
        df = add_battery_column(df, default=None)
    df = df[df["variable"].isin(variables)].copy()
    if df.empty:
        return df

    baseline_order = sorted(df["baseline_label"].dropna().unique().tolist())
    df["baseline_label"] = pd.Categorical(df["baseline_label"], categories=baseline_order, ordered=True)

    variables = [v for v in variables if v in df["variable"].unique()]
    all_batteries_cfg = list(BATTERY2TESTCODE.keys())
    present_batteries = df["battery"].dropna().unique().tolist()
    batteries = sorted(present_batteries, key=lambda b: all_batteries_cfg.index(b) if b in all_batteries_cfg else 1e9)

    axes_arr = np.asarray(axes, dtype=object)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape(1, -1)
    n_rows_avail, n_cols_avail = axes_arr.shape
    n_rows_data, n_cols_data = len(variables), len(batteries)
    for i in range(n_rows_avail):
        for j in range(n_cols_avail):
            if i >= n_rows_data or j >= n_cols_data:
                axes_arr[i, j].set_axis_off()

    n_baselines = len(baseline_order)
    upper_limit = n_baselines * OFFSET
    offsets = np.linspace(-upper_limit, upper_limit, n_baselines) if n_baselines > 1 else np.array([0.0])
    base_color = MODEL2COLOR["bayesian"]
    colors = sns.light_palette(base_color, n_colors=n_baselines + 1)[1:]
    legend_handles_map = {}

    for i, var in enumerate(variables):
        if i >= n_rows_avail:
            break
        for j, battery in enumerate(batteries):
            if j >= n_cols_avail:
                break
            ax = axes_arr[i, j]
            cell_df = df[(df["battery"] == battery) & (df["variable"] == var)].copy()
            battery_tests = battery_tests_in_ioi_order(df, battery, ioi_order)
            x_map = {tc: idx for idx, tc in enumerate(battery_tests)}

            for tc in battery_tests:
                tc_df = cell_df[cell_df[TEST_CODE_COL].astype(str) == tc]
                x_center = x_map[tc]
                for k, base_label in enumerate(baseline_order):
                    rows = tc_df[tc_df["baseline_label"] == base_label]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    x_pos = x_center + offsets[k]
                    hr = row["hr"]
                    ax.errorbar(
                        x=x_pos,
                        y=hr,
                        yerr=[[hr - row["ci_lower"]], [row["ci_upper"] - hr]],
                        fmt="o",
                        color=colors[k],
                        ecolor=colors[k],
                        markeredgecolor=colors[k],
                        markeredgewidth=0.0,
                        markersize=FIG3_ERROR_BAR_MARKERSIZE,
                        elinewidth=1.0,
                        capsize=2.0,
                        capthick=1.0,
                        alpha=1,
                        clip_on=True,
                    )
                    if base_label not in legend_handles_map:
                        legend_handles_map[base_label] = ax.plot([], [], marker="o", linestyle="", color=colors[k], markersize=FIG3_ERROR_BAR_MARKERSIZE)[0]

            ax.axhline(1.0, **REF_LINE_STYLE)
            ax.set_xticks(range(len(battery_tests)))
            if i == min(n_rows_data, n_rows_avail) - 1:
                ax.set_xticklabels([TESTCODE_DISPLAY.get(t, t) for t in battery_tests], rotation=45, ha="center", fontsize=5)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel("Setpoint" if var.lower().startswith(MU) else var.upper(), rotation=90)
            else:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)

        yticks = [0.5, 1.0, 1.5, 2.0]
        for j in range(min(n_cols_data, n_cols_avail)):
            axes_arr[i, j].set_ylim(yticks[0], yticks[-1])
        _snap_yticks_to_ylim(axes_arr[i, 0])

    if add_legend and baseline_order:
        filtered_order = [b for b in baseline_order if b in legend_handles_map]
        handles = [legend_handles_map[b] for b in filtered_order]
        axes_arr[0, 0].legend(
            handles,
            filtered_order,
            title="Number of Tests Used:",
            title_fontsize=FONT_SIZE_TICK_LABEL,
            loc="upper left",
            bbox_to_anchor=(0, 1.2),
            ncol=len(filtered_order),
            frameon=False,
            borderaxespad=0.0,
            handletextpad=0.1,
        )
    return df


def fig3baseline(hr_df: pd.DataFrame, ioi_order: list, variables=(MU, CV_COL), save_path=None, figsize=None, csv_path=None) -> Optional[plt.Figure]:
    """Figure 3B: hazard ratios by battery (columns) and variable (rows), hue=baseline index."""
    df = hr_df.copy()
    if "battery" not in df.columns:
        df = add_battery_column(df, default=None)
    df = df[df["variable"].isin(variables)].copy()
    if df.empty:
        return None

    variables = [v for v in variables if v in df["variable"].unique()]
    all_batteries_cfg = list(BATTERY2TESTCODE.keys())
    present_batteries = df["battery"].dropna().unique().tolist()
    batteries = sorted(present_batteries, key=lambda b: all_batteries_cfg.index(b) if b in all_batteries_cfg else 1e9)
    width_ratios = [max(df[df["battery"] == b][TEST_CODE_COL].nunique(), 1) for b in batteries]
    n_rows, n_cols = len(variables), len(batteries)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize or (A4_WIDTH, n_rows),
        sharey=False,
        sharex=False,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.2, "hspace": 0.25},
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.atleast_2d(axes)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig3baseline_on_axes(axes, hr_df, ioi_order, variables=variables, add_legend=True)
    if n_rows > 0 and n_cols > 0:
        fig.align_ylabels([axes[i, 0] for i in range(n_rows)])
        fig.align_xlabels([axes[n_rows - 1, j] for j in range(n_cols)])
    if save_path is not None:
        save_fig_as_svg(fig, "fig3b_hr_by_baseline", save_path, csv_path=csv_path)
    return fig
