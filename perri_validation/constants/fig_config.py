"""Figure styling constants.

Pure style constants + the global matplotlib rcParams block only. Scripts call
fig.savefig(path, format="svg") directly with their own output paths.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

REF_LINE_STYLE = {"color": "gray", "linestyle": "--", "linewidth": 0.8, "zorder": 0}

MOSAIC_WSPACE = 0.5
MOSAIC_HSPACE = 0.7
SUBPLOTS_WSPACE_LARGE = 0.5

A4_WIDTH = 8
A4_HEIGHT = 11.7
LINEWIDTH = 1.25

FIG4_ROW1_HEIGHT = 1.6
FIG4_ROW2_HEIGHT = 0.75

FIG5_ROW1_HEIGHT = 2
FIG5_ROW2_HEIGHT = 3
FIG5_COL1_WIDTH = 2
FIG5_COL2_WIDTH = 1
FIG5_COL3_WIDTH = 2
FIG5_COL4_WIDTH = 1
FIG5DOSE_RESPONSE_HSPACE = 0.7
FIG5TRAJECTORY_HSPACE = 0.7
FIG5KM_HSPACE = 0.7

BLOCK_PLOTS = False
ANNOT_Y = 1.05
QUANTILE_BAND_OUTER_ALPHA = 0.4
QUANTILE_BAND_INNER_ALPHA = 0.6
PER_RI_ALPHA = 0.6
POP_RI_ALPHA = 0.3

FONT_SIZE_BASE = 8
FONT_SIZE_AXIS_LABEL = 10
FONT_SIZE_AXIS_LABEL_SMALL = 8
FONT_SIZE_TICK_LABEL = 6
FONT_SIZE_PANEL_TITLE = 8
FONT_SIZE_PANEL_TITLE_LARGE = 10
FONT_SIZE_LEGEND = 6
FONT_SIZE_ANNOTATION = 6
FONT_SIZE_CALLOUT = 7
FONT_SIZE_HEATMAP_ANNOTATION = 6

FIG4_AXIS_LABEL_FONTSIZE = FONT_SIZE_AXIS_LABEL_SMALL
FIG4_TITLE_FONTSIZE = FONT_SIZE_PANEL_TITLE
FIG4_TITLE_LARGE_FONTSIZE = FONT_SIZE_PANEL_TITLE_LARGE
FIG4_LEGEND_FONTSIZE = FONT_SIZE_LEGEND
FIG4_NOTE_FONTSIZE = FONT_SIZE_ANNOTATION
FIG4_CALLOUT_FONTSIZE = FONT_SIZE_CALLOUT
FIG4_HEATMAP_ANNOT_FONTSIZE = FONT_SIZE_HEATMAP_ANNOTATION

FIG3_ERROR_BAR_MARKERSIZE = 3


def mm_to_pt(mm: float) -> float:
    return mm * 72.0 / 25.4


DEFAULT_LINEWIDTH_PT = mm_to_pt(0.3)
DEFAULT_AXIS_LINEWIDTH_PT = DEFAULT_LINEWIDTH_PT
TICK_MAJOR_LENGTH_PT = mm_to_pt(1)
TICK_MINOR_LENGTH_PT = TICK_MAJOR_LENGTH_PT / 2

mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Droid Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["svg.fonttype"] = "path"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "Droid Sans"],
        "font.size": FONT_SIZE_BASE,
        "axes.titlesize": 12,
        "axes.labelsize": FONT_SIZE_AXIS_LABEL,
        "axes.titleweight": "normal",
        "axes.labelpad": 2,
        "xtick.labelsize": FONT_SIZE_TICK_LABEL,
        "ytick.labelsize": FONT_SIZE_TICK_LABEL,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.title_fontsize": 8,
        "xtick.major.size": TICK_MAJOR_LENGTH_PT,
        "ytick.major.size": TICK_MAJOR_LENGTH_PT,
        "xtick.minor.size": TICK_MINOR_LENGTH_PT,
        "ytick.minor.size": TICK_MINOR_LENGTH_PT,
        "xtick.major.width": DEFAULT_LINEWIDTH_PT,
        "ytick.major.width": DEFAULT_LINEWIDTH_PT,
        "xtick.minor.width": DEFAULT_LINEWIDTH_PT,
        "ytick.minor.width": DEFAULT_LINEWIDTH_PT,
        "lines.linewidth": DEFAULT_LINEWIDTH_PT,
        "lines.markersize": 4,
        "axes.linewidth": DEFAULT_AXIS_LINEWIDTH_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.2,
        "legend.handletextpad": 0.3,
        "legend.columnspacing": 0.4,
        "savefig.transparent": True,
        "savefig.dpi": 300,
    }
)


def _to_rgba(color: str, alpha: float) -> str:
    red, green, blue = [int(round(channel * 255)) for channel in mpl.colors.to_rgb(color)]
    return f"rgba({red}, {green}, {blue}, {alpha})"


MODEL2COLOR = {
    "Kalman": "#F6BD60",
    "gmm": "#1B9E77",
    "bayesian": "#3B5CCB",
    "particle_filter": "#F4978E",
    "hmm": "#CDB4DB",
    "population": "#9E9E9E",
    "moving_average": "#84DCC6",
    "greedy": "#F2B5D4",
}

COLOR_MAP = {model: {"line": color, "fill": _to_rgba(color, 0.2)} for model, color in MODEL2COLOR.items()}

BATTERY2COLOR = dict(
    [
        ("CBC", "#d73027"),
        ("BMP", sns.color_palette("Set2", n_colors=7)[0]),
        ("WBC diff", sns.color_palette("Set2", n_colors=7)[1]),
        ("LFT", sns.color_palette("Set2", n_colors=7)[2]),
        ("Hepatic", sns.color_palette("Set2", n_colors=7)[3]),
        ("Lipid", sns.color_palette("Set2", n_colors=7)[3]),
        ("Coag", sns.color_palette("Set2", n_colors=7)[4]),
        ("Misc", sns.color_palette("Set2", n_colors=7)[5]),
    ]
)

SEX_PALETTE = {
    "All": "#9E9E9E",
    "ALL": "#9E9E9E",
    "all": "#9E9E9E",
    "M": "#0072B2",
    "F": "#CC79A7",
}
