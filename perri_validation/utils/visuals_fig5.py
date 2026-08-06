"""Figure 5 plotting functions.

The six `_on_ax`/`_on_axes` panel-rendering functions used by fig5's real
mosaic (dose-response, response-distribution, trajectory, swimmer,
interaction, heatmap).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from perri_validation.constants.fig_config import (
    ANNOT_Y,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK_LABEL,
    LINEWIDTH,
    QUANTILE_BAND_INNER_ALPHA,
    QUANTILE_BAND_OUTER_ALPHA,
    REF_LINE_STYLE,
)
from perri_validation.constants.runtime import (
    BELOW_STR,
    ID_COL,
    INSIDE_STR,
    MEASUREMENT_COL,
    MU,
    OUT_PERRI_P95_LOWER_COL,
    OUT_PERRI_P95_UPPER_COL,
    OUT_POPRI_LOWER_COL,
    OUT_POPRI_UPPER_COL,
    RANDOM_SEED,
    SEX_COL,
    SIGMA,
    TS_COL,
)

HB_LARGE_IMPROVEMENT_THRESHOLD_G_DL = 1


def fig5dose_response_on_axes(ax, iv_cohort, outcome_df):
    """Fig 5B dose-response boxplot with per-bin n= annotations."""
    dose_counts = outcome_df[outcome_df[ID_COL].isin(iv_cohort[ID_COL])].groupby(ID_COL).size()
    labels = ["1", "2", "3", "4", "5+"]
    count_by_bin = pd.cut(dose_counts, bins=[0, 1, 2, 3, 4, np.inf], labels=labels).value_counts().reindex(labels, fill_value=0)

    df = iv_cohort.copy()
    df["dose_bin"] = df["doses"].apply(lambda x: str(int(x)) if x < 5 else "5+")
    df["dose_bin"] = pd.Categorical(df["dose_bin"], categories=labels, ordered=True)

    sns.boxplot(x="dose_bin", y="response", data=df, order=labels, hue="dose_bin", legend=False, palette="Blues", fliersize=0, showfliers=True, ax=ax)

    ax.axhline(0, **REF_LINE_STYLE)
    ax.set_yticks(np.arange(-3, np.ceil(df["response"].max()) + 1, 3))
    y_lim_bounds = ax.get_yticks()[[0, -1]]
    ax.set_ylim(y_lim_bounds)
    ax.set_xlabel("Treatment Dose Count")
    ax.set_ylabel("Response")

    for i, lbl in enumerate(labels):
        ax.text(i, ANNOT_Y, f"n={count_by_bin[lbl]}", rotation=90, ha="center", va="top", fontsize=FONT_SIZE_ANNOTATION, transform=ax.get_xaxis_transform(), clip_on=False)

    return ax


def fig5response_distribution_on_axes(
    ax_pres_v_res,
    ax_drop_v_res,
    iv_cohort,
    mu_col: str = MU,
    sigma_col: str = SIGMA,
    drop_col: str = "drop",
    response_col: str = "response",
    dose_col: str = "doses",
    presenting_col: str = "result_pre",
    *,
    add_colorbar: bool = True,
    colorbar_ax=None,
    equal_aspect: bool = True,
):
    """Fig 5B: Two-panel diagnostic for IV cohort.

    TOP: Presenting vs Response. BOTTOM: Drop vs Response.
    Color = presenting z-score. Size = number of doses.
    """
    df_clean = iv_cohort.copy()
    SIZE_FACTOR = 5

    hard_coded = [-6, -3, 0, 3, 6, 9]
    min_lim = hard_coded[0]
    max_lim = hard_coded[-1]
    y_shared_ticks = hard_coded
    y_shared_min = hard_coded[0]
    y_shared_max = hard_coded[-1]

    if "z_score" not in df_clean.columns:
        df_clean["z_score"] = (df_clean[presenting_col] - df_clean[mu_col]) / df_clean[sigma_col]

    sns.kdeplot(x=df_clean[presenting_col], y=df_clean[response_col], ax=ax_pres_v_res, levels=5, color=".5", linewidths=0.7, alpha=0.5, zorder=1)

    sc_d = ax_pres_v_res.scatter(
        df_clean[presenting_col],
        df_clean[response_col],
        c=df_clean["z_score"],
        s=df_clean[dose_col] * SIZE_FACTOR,
        cmap="RdBu",
        vmin=-3,
        vmax=3,
        alpha=0.4,
        edgecolor="k",
        linewidths=0.5,
        clip_on=True,
        zorder=2,
    )

    x_p = df_clean[presenting_col]
    y_p = df_clean[response_col]
    if x_p.nunique() > 1:
        slope_p, intercept_p, r_c, _, _ = stats.linregress(x_p, y_p)
        x_line_p = np.array([x_p.min(), x_p.max()])
        y_line_p = intercept_p + slope_p * x_line_p
        ax_pres_v_res.annotate(f"Slope={slope_p:.2f}\nR²={r_c**2:.2f}", ha="right", va="top", xy=(0.95, ANNOT_Y), xycoords="axes fraction", fontsize=FONT_SIZE_ANNOTATION)
        ax_pres_v_res.plot(x_line_p, y_line_p, color="black", linewidth=LINEWIDTH, clip_on=False)

    ax_pres_v_res.axhline(0, **REF_LINE_STYLE)

    x_min = np.floor(df_clean[presenting_col].min())
    x_max = np.ceil(df_clean[presenting_col].max())
    num_ticks = int(np.ceil((x_max - x_min) / 3)) + 1
    ax_pres_v_res.set_xticks(np.linspace(x_min, x_max + 1, num_ticks))
    ax_pres_v_res.set_xlim(x_min, x_max)
    ax_pres_v_res.set_yticks(y_shared_ticks)
    ax_pres_v_res.set_ylim(y_shared_min, y_shared_max)
    ax_pres_v_res.set_xlabel("Pre-Treatment HB", fontsize=FONT_SIZE_TICK_LABEL)
    ax_pres_v_res.set_ylabel("Response", fontsize=FONT_SIZE_TICK_LABEL)
    if equal_aspect:
        ax_pres_v_res.set_aspect("equal", adjustable="box")

    sns.kdeplot(x=df_clean[drop_col], y=df_clean[response_col], ax=ax_drop_v_res, levels=5, color=".5", linewidths=0.7, alpha=0.5, zorder=1)

    ax_drop_v_res.scatter(
        df_clean[drop_col],
        df_clean[response_col],
        c=df_clean["z_score"],
        s=df_clean[dose_col] * SIZE_FACTOR,
        cmap="RdBu",
        vmin=-3,
        vmax=3,
        alpha=0.4,
        edgecolor="k",
        linewidths=0.5,
        clip_on=True,
        zorder=2,
    )

    x_c = df_clean[drop_col]
    y_c = df_clean[response_col]
    if x_c.nunique() > 1:
        slope_c, intercept_c, r_c, _, _ = stats.linregress(x_c, y_c)
        x_line_c = np.array([min_lim, max_lim])
        y_line_c = intercept_c + slope_c * x_line_c
        ax_drop_v_res.annotate(f"Slope={slope_c:.2f}\nR²={r_c**2:.2f}", xy=(0.05, ANNOT_Y), ha="left", va="top", xycoords="axes fraction", fontsize=FONT_SIZE_ANNOTATION)
        ax_drop_v_res.plot(x_line_c, y_line_c, color="black", linewidth=LINEWIDTH, alpha=1, clip_on=False)

    ax_drop_v_res.plot([min_lim, max_lim], [min_lim, max_lim], color="gray", linestyle="--", linewidth=LINEWIDTH, alpha=1, zorder=0)

    ax_drop_v_res.set_xticks(hard_coded)
    ax_drop_v_res.set_xlim(min_lim, max_lim)
    ax_drop_v_res.set_yticks(hard_coded)
    ax_drop_v_res.set_ylim(y_shared_min, y_shared_max)
    ax_drop_v_res.set_xlabel("Pre-Treatment Drop", fontsize=FONT_SIZE_TICK_LABEL)
    ax_drop_v_res.set_ylabel("Response", fontsize=FONT_SIZE_TICK_LABEL)

    red_color = plt.cm.RdBu(0.2)
    handles_c = [plt.scatter([], [], color=red_color, alpha=0.5, s=d * SIZE_FACTOR, label=f"{d} doses") for d in [1, 5]]
    ax_drop_v_res.legend(handles=handles_c, bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol=2, fontsize=FONT_SIZE_LEGEND, frameon=False)

    if add_colorbar:
        fig = ax_pres_v_res.get_figure()
        if colorbar_ax is None:
            colorbar_ax = fig.add_axes((0.85, 0.3, 0.03, 0.4))
        cbar = fig.colorbar(sc_d, cax=colorbar_ax)
        cbar.set_label("Presenting z-score", rotation=270, labelpad=15)

    return ax_pres_v_res, ax_drop_v_res


def fig5trajectory_on_axes(
    ax_abs,
    ax_drop,
    df: pd.DataFrame,
    value_col: str = MEASUREMENT_COL,
    pre_col: str = "result_pre",
    setpoint_col: str = MU,
    window_range: tuple = (-365, 180),
    bin_width_days: int = 14,
    id_col: str = ID_COL,
):
    df_plot = df.copy()
    df_plot["aligned_days"] = np.select([df_plot["days_pre"] <= 0, df_plot["days_post"] >= 0], [df_plot["days_pre"], df_plot["days_post"]], default=np.nan)

    df_plot = df_plot.dropna(subset=["aligned_days"])
    df_plot = df_plot[df_plot["aligned_days"].between(*window_range)]
    df_plot["day_bin"] = (df_plot["aligned_days"] / bin_width_days).round() * bin_width_days

    def plot_panel(ax, col_name, color, label_y, y_lim_pad=1, legend=False):
        sns.lineplot(data=df_plot, x="aligned_days", y=col_name, units=id_col, estimator=None, color="black", alpha=0.03, linewidth=1, ax=ax, zorder=1)

        summary = df_plot.groupby("day_bin")[col_name].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
        summary.columns = ["q10", "q25", "q50", "q75", "q90"]
        x_vals = summary.index

        ax.fill_between(x_vals, summary["q10"], summary["q90"], color=color, alpha=QUANTILE_BAND_OUTER_ALPHA, zorder=2, label="10-90th%")
        ax.fill_between(x_vals, summary["q25"], summary["q75"], color=color, alpha=QUANTILE_BAND_INNER_ALPHA, zorder=3, label="25-75th%")
        ax.plot(x_vals, summary["q50"], color="black", linewidth=LINEWIDTH, zorder=4, label="Median")

        ax.axvline(0, **REF_LINE_STYLE)
        if col_name != value_col:
            ax.axhline(0, **REF_LINE_STYLE)

        ax.set_ylabel(label_y)

        y_min = summary["q25"].min() - y_lim_pad
        y_max = summary["q75"].max() + y_lim_pad
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))

        if legend:
            ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.4), fontsize=FONT_SIZE_LEGEND, frameon=False)

        return y_max

    df_plot["drop"] = df_plot[value_col] - df_plot[setpoint_col]
    df_plot["dirac"] = df_plot[value_col] - df_plot[pre_col]
    configs = [
        {"ax": ax_abs, "col": value_col, "color": "#1f77b4", "lbl": "HB"},
        {"ax": ax_drop, "col": "drop", "color": "#d62728", "lbl": "Drop"},
    ]

    for cfg in configs:
        y_max = plot_panel(cfg["ax"], cfg["col"], cfg["color"], cfg["lbl"], legend=cfg["col"] == "drop")
        if cfg["ax"] == ax_abs:
            cfg["ax"].text(-5, y_max, "← Before", ha="right", style="italic", fontsize=FONT_SIZE_ANNOTATION)
            cfg["ax"].text(5, y_max, "After →", ha="left", style="italic", fontsize=FONT_SIZE_ANNOTATION)

    ax_abs.set_xlabel("")
    ax_drop.set_xlabel("Days Relative to Treatment")
    ax_drop.set_xlim(window_range)

    return ax_abs, ax_drop


def fig5swimmer_on_ax(ax, iv_cohort, lab_df, infusion_df, id_col=ID_COL, ts_col=TS_COL, n_sample=15, sort_by="setpoint_age", seed=RANDOM_SEED):
    """Swimmer-style validation from iv_cohort.

    Uses course_start as the anchor. Connects Setpoint -> Pre-Lab -> Infusion
    doses -> Post-Lab. Plots all infusion doses for sampled patients.
    """
    required_cols = {"course_start", "set_ts", "result_pre_ts", "result_post_ts", "doses"}
    missing = required_cols - set(iv_cohort.columns)
    if missing:
        raise ValueError(f"iv_cohort missing required columns: {missing}")

    iv = iv_cohort.copy()
    iv["course_start"] = pd.to_datetime(iv["course_start"])
    iv["set_ts"] = pd.to_datetime(iv["set_ts"])
    iv["result_pre_ts"] = pd.to_datetime(iv["result_pre_ts"])
    iv["result_post_ts"] = pd.to_datetime(iv["result_post_ts"])

    iv = iv[(iv["course_start"] - iv["set_ts"]).dt.days <= 3 * 365]

    anchors = iv[[id_col, "course_start"]].sort_values("course_start").groupby(id_col, as_index=False).first().rename(columns={"course_start": "anchor_ts"})

    valid_pats = anchors[id_col].unique()
    if len(valid_pats) == 0:
        ax.text(0.5, 0.5, "No patients found in iv_cohort.", ha="center", va="center", transform=ax.transAxes)
        return ax, "fig5f_swimmer_plot"

    sample_ids = np.random.default_rng(seed).choice(valid_pats, size=min(n_sample, len(valid_pats)), replace=False)
    anchors = anchors[anchors[id_col].isin(sample_ids)].copy()

    sub_iv = iv[iv[id_col].isin(sample_ids)].merge(anchors, on=id_col, how="inner")
    sub_iv["set_day"] = (sub_iv["set_ts"] - sub_iv["anchor_ts"]).dt.days
    sub_iv["pre_day"] = (sub_iv["result_pre_ts"] - sub_iv["anchor_ts"]).dt.days
    sub_iv["post_day"] = (sub_iv["result_post_ts"] - sub_iv["anchor_ts"]).dt.days

    patient_post_day = sub_iv.drop_duplicates(subset=[id_col]).set_index(id_col)["post_day"]

    sorter = pd.DataFrame(index=sample_ids)
    ascending = True
    if sort_by == "doses":
        doses = sub_iv.groupby(id_col)["doses"].first()
        sorter["metric"] = sorter.index.map(doses)
        ascending = True
    elif sort_by == "post_timing":
        post_days = sub_iv.groupby(id_col)["post_day"].min()
        sorter["metric"] = sorter.index.map(post_days)
        ascending = True
    elif sort_by == "setpoint_age":
        set_days = sub_iv.groupby(id_col)["set_day"].min()
        sorter["metric"] = sorter.index.map(set_days)
        ascending = False
    else:
        sorter["metric"] = 1
        ascending = True

    sorter = sorter.sort_values("metric", ascending=ascending)
    sorted_ids = sorter.index.tolist()
    y_map = {pid: i for i, pid in enumerate(sorted_ids)}

    inf = infusion_df[infusion_df[id_col].isin(sample_ids)].copy()
    inf[ts_col] = pd.to_datetime(inf[ts_col])
    inf = inf.merge(anchors, on=id_col, how="inner")
    inf["rel_day"] = (inf[ts_col] - inf["anchor_ts"]).dt.days
    inf["y"] = inf[id_col].map(y_map)
    inf["post_day"] = inf[id_col].map(patient_post_day)
    inf = inf[inf["rel_day"] <= inf["post_day"]]

    min_day = min(sub_iv["set_day"].min(), inf["rel_day"].min())
    max_day = max(sub_iv["post_day"].max(), inf["rel_day"].max())
    pad_left = abs(min_day) * 0.1
    pad_right = max_day * 0.1 if max_day > 0 else 10
    window_days = (min_day - pad_left, max_day + pad_right)

    sub_labs = lab_df[lab_df[id_col].isin(sample_ids)].copy()
    sub_labs[ts_col] = pd.to_datetime(sub_labs[ts_col])
    sub_labs = sub_labs.merge(anchors, on=id_col, how="inner")
    sub_labs["rel_day"] = (sub_labs[ts_col] - sub_labs["anchor_ts"]).dt.days
    sub_labs["y"] = sub_labs[id_col].map(y_map)
    sub_labs = sub_labs[sub_labs["rel_day"].between(window_days[0], window_days[1])]
    sub_labs["post_day"] = sub_labs[id_col].map(patient_post_day)
    sub_labs = sub_labs[sub_labs["rel_day"] <= sub_labs["post_day"]]

    marker_size = 20
    ax.scatter(sub_labs["rel_day"], sub_labs["y"], c="lightgray", s=15, alpha=0.6, label="Other Labs", zorder=1)

    sub_set = sub_iv.drop_duplicates(subset=[id_col]).copy()
    sub_set["y"] = sub_set[id_col].map(y_map)

    ax.hlines(y=sub_set["y"], xmin=sub_set["set_day"], xmax=sub_set["post_day"], color="gray", alpha=0.3, linewidth=1, linestyle=":", zorder=1)
    ax.scatter(sub_set["set_day"], sub_set["y"], c="#1f77b4", s=marker_size, marker="o", edgecolors="k", linewidth=0.5, label="Setpoint Source", zorder=3)
    ax.scatter(sub_set["pre_day"], sub_set["y"], c="gold", s=marker_size, marker="<", edgecolors="k", linewidth=0.5, label="Pre-Treatment HB", zorder=4)

    inf_plot = inf[inf["rel_day"].between(window_days[0], window_days[1])]
    ax.scatter(inf_plot["rel_day"], inf_plot["y"], c="#d62728", s=marker_size / 2, marker="o", edgecolors="k", linewidth=0.5, alpha=0.7, label="Infusion Dose", zorder=2)
    ax.scatter(sub_set["post_day"], sub_set["y"], c="lightblue", s=marker_size, marker=">", edgecolors="k", linewidth=0.5, label="Post-Treatment Lab", zorder=4)

    ax.set_yticks(range(len(sample_ids)))
    ax.set_yticklabels(["" for _ in sorted_ids])
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    title = f"fig5f_swimmer_plot {len(sample_ids)} Random IV Iron Patients (Sorted by {sort_by.replace('_', ' ').title()})"
    ax.set_xlabel("Days Relative to Infusion Treatment Start")
    ax.set_ylabel("Timelines")
    ax.legend(loc="lower left", frameon=True, fontsize=FONT_SIZE_LEGEND)
    ax.set_xlim(window_days)

    return ax, title


def fig5interaction_on_ax(ax, cohort_df, *, outcome_col: str = "response", x_col: str = "result_pre", n_hb_bins: int = 4):
    """Interaction panel: response vs pre-treatment drop, colored by presenting HB bin."""
    df = cohort_df.copy()
    df = df.dropna(subset=[x_col, "drop", outcome_col]).copy()

    cut_points = [0, 1, 2]
    drop_bins = [-np.inf] + cut_points + [np.inf]
    raw_labels = [f"< {cut_points[0]}"] + [f"{cut_points[i]}-{cut_points[i + 1]}" for i in range(len(cut_points) - 1)] + [f">= {cut_points[-1]}"]
    df["drop_group"] = pd.cut(df["drop"], bins=drop_bins, labels=raw_labels)

    _, x_cuts = pd.qcut(df[x_col], q=n_hb_bins, retbins=True, duplicates="drop")
    hb_labels = [f"{x_cuts[i]:.1f}-{x_cuts[i + 1]:.1f}" for i in range(len(x_cuts) - 1)]
    df["hb_group"] = pd.cut(df[x_col], bins=x_cuts, labels=hb_labels, include_lowest=True)

    cmap = plt.get_cmap("Reds")
    if n_hb_bins <= 1:
        custom_palette = [cmap(0.8)]
    else:
        custom_palette = [cmap(0.3 + (n_hb_bins - 1 - i) * (0.6 / (n_hb_bins - 1))) for i in range(n_hb_bins)]

    sns.pointplot(
        data=df,
        x="drop_group",
        y=outcome_col,
        hue="hb_group",
        palette=custom_palette,
        dodge=0.4,
        capsize=0.1,
        markersize=4,
        errorbar=("ci", 95),
        linestyles="-",
        ax=ax,
    )

    ax.axhline(0, **REF_LINE_STYLE)
    ax.set_xlabel("Pre-Treatment Drop", fontsize=FONT_SIZE_TICK_LABEL)
    ax.set_ylabel("Response", fontsize=FONT_SIZE_TICK_LABEL)
    ax.legend(title="Pre-Treatment HB", loc="lower right", title_fontsize=FONT_SIZE_LEGEND, fontsize=FONT_SIZE_LEGEND, ncols=2, frameon=False)


def fig5heatmap_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    x_col: str = "result_pre",
    setpoint_col: str = MU,
    outcome_col: str = "response",
    sex_col: str = SEX_COL,
    sex_value: str = None,
    use_popri: bool = True,
    drop_use_perri: bool = True,
    drop_static_threshold: float = 1.0,
    outcome_threshold: float = HB_LARGE_IMPROVEMENT_THRESHOLD_G_DL,
    annotate: bool = True,
    cmap: str = "coolwarm",
    label_y: bool = True,
    label_x: bool = True,
):
    """Heatmap panel: P(large improvement) by PopRI x PerRI classification of the pre-treatment value.

    drop = setpoint - presenting (positive means presenting is below setpoint).
    """
    data = df.copy()

    if (sex_value is not None) and (sex_col in data.columns):
        data = data[data[sex_col] == sex_value].copy()

    data["_x"] = pd.to_numeric(data[x_col], errors="coerce")
    data["_set"] = pd.to_numeric(data[setpoint_col], errors="coerce")
    data["_y"] = pd.to_numeric(data[outcome_col], errors="coerce")

    value_col = "_y_bin"
    data[value_col] = (data["_y"] >= float(outcome_threshold)).astype(int)
    vmin, vmax = 0.0, 1.0

    required = ["_x", "_set", "drop", value_col] + ([OUT_PERRI_P95_LOWER_COL] if drop_use_perri else [])
    data = data.dropna(subset=required).copy()

    if data.empty:
        ax.text(0.5, 0.5, "No complete cases", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None, pd.DataFrame()

    def _make_presenting_bins(sub, use_popri=False):
        if use_popri:
            x_codes = sub[OUT_POPRI_LOWER_COL].astype(int)
            x_codes = 1 - x_codes
            x_labels = [BELOW_STR, INSIDE_STR]
        else:
            x_codes, x_edges = pd.qcut(sub["_x"].astype(float), q=2, labels=False, retbins=True, duplicates="drop")
            x_edges = np.asarray(x_edges, dtype=float)
            x_labels = ["{0:.1f}-{1:.1f}".format(x_edges[i], x_edges[i + 1]) for i in range(len(x_edges) - 1)]
        return x_codes, x_labels

    def _make_drop_bins(sub, drop_static_threshold=1.0, use_perri=False):
        if use_perri:
            drop_codes = sub[OUT_PERRI_P95_LOWER_COL].astype(int)
            drop_labels = [INSIDE_STR, BELOW_STR]
        else:
            thr_drop = float(drop_static_threshold)
            d = sub["drop"].astype(float)
            drop_edges = np.asarray([d.min() - 1e-6, thr_drop, d.max() + 1e-6], dtype=float)
            drop_codes = pd.cut(d, bins=drop_edges, labels=False, include_lowest=True)
            drop_labels = [f"≤ {thr_drop:.0f}", f"> {thr_drop:.0f}"]
        return drop_codes, drop_labels

    drop_codes, drop_labels = _make_drop_bins(data, drop_static_threshold, use_perri=drop_use_perri)
    x_codes, x_labels = _make_presenting_bins(data, use_popri=use_popri)
    data["x_bin"] = x_codes
    data["drop_bin"] = drop_codes

    data = data.dropna(subset=["x_bin", "drop_bin"]).copy()

    # filter out data that is out of the popRI
    pre_pop_high_counts = len(data)
    data = data[data[OUT_POPRI_UPPER_COL] == 0].copy()
    post_pop_high_counts = len(data)
    print(f"Filtering out {pre_pop_high_counts - post_pop_high_counts} samples above popRI for heatmap. Remaining samples: {post_pop_high_counts}")

    # filter out data that is out of the perRI
    pre_perri_high_counts = len(data)
    data = data[data[OUT_PERRI_P95_UPPER_COL] == 0].copy()
    post_perri_high_counts = len(data)
    print(f"Filtering out {pre_perri_high_counts - post_perri_high_counts} samples above perRI for heatmap annotation. Remaining samples: {post_perri_high_counts}")

    if data.empty:
        ax.text(0.5, 0.5, "No binned data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None, pd.DataFrame()

    cell_counts = data.groupby(["drop_bin", "x_bin"], observed=True).size().unstack(fill_value=0)

    agg_df = data.groupby(["drop_bin", "x_bin"], observed=True).agg(value=(value_col, "mean"), n=(value_col, "size")).reset_index()

    heat = agg_df.pivot(index="drop_bin", columns="x_bin", values="value").sort_index().sort_index(axis=1)
    counts = agg_df.pivot(index="drop_bin", columns="x_bin", values="n").reindex(index=heat.index, columns=heat.columns)

    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel("PopRI")

    x_cols = heat.columns.to_list()
    y_rows = heat.index.to_list()

    x_tick_labels = [x_labels[int(i)] if int(i) < len(x_labels) else str(i) for i in x_cols]
    y_tick_labels = [drop_labels[int(i)] if int(i) < len(drop_labels) else str(i) for i in y_rows]

    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, fontsize=FONT_SIZE_TICK_LABEL)

    if not label_x:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

    ax.set_yticks(np.arange(len(y_tick_labels)))
    if label_y:
        ax.set_yticks(np.arange(len(y_tick_labels)))
        ax.set_yticklabels(y_tick_labels, rotation="vertical", fontsize=FONT_SIZE_TICK_LABEL, va="center")

        if sex_value in ["F", "M"]:
            ax.set_ylabel(f"PerRI ({sex_value})")
        else:
            ax.set_ylabel("PerRI")
    else:
        ax.set_yticklabels(["", ""])

    if annotate:
        mat = heat.to_numpy()
        nmat = counts.to_numpy()
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                n_cell = int(nmat[r, c]) if np.isfinite(nmat[r, c]) else 0
                if n_cell == 0:
                    continue
                txt = f"{mat[r, c] * 100:.1f}%\nn={n_cell}" if outcome_threshold is not None else f"{mat[r, c]:.2f}\nn={n_cell}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=FONT_SIZE_ANNOTATION)

    return im, cell_counts
