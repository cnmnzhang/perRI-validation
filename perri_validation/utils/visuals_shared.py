"""Styling and layout helpers 
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from perri_validation.constants.fig_config import LINEWIDTH, MODEL2COLOR
from perri_validation.constants.marker_config import BATTERY2LABEL
from perri_validation.constants.runtime import INSIDE_STR, OUTSIDE_STR

OFFSET = 0.08
Y_TICK_LABELSIZE = 8
FIG3_ERROR_BAR_MARKERSIZE = 3
FIG3_CAPSIZE = 0

def format_battery_label(battery: str) -> str:
    return BATTERY2LABEL.get(battery, str(battery))


def _snap_yticks_to_ylim(ax) -> None:
    """Extend ticks so the top tick equals the axis upper limit."""
    lo, hi = ax.get_ylim()
    ticks = sorted(t for t in ax.get_yticks() if t >= lo - 1e-10)
    if len(ticks) < 2:
        return
    step = ticks[-1] - ticks[-2]
    while ticks[-1] < hi - 1e-10:
        ticks.append(ticks[-1] + step)
    ax.set_yticks([t for t in ticks if t >= lo - 1e-10])
    ax.set_ylim(lo, ticks[-1])

PROG_STYLING = {
    "A_popIn_perIn": {"color": MODEL2COLOR["bayesian"], "ls": "-", "lw": LINEWIDTH, "label": "In Both"},
    "B_popIn_perOut": {"color": "tab:orange", "ls": "-", "lw": LINEWIDTH, "label": f"{INSIDE_STR} PopRI, {OUTSIDE_STR} PerRI"},
    "C_popOut_perIn": {"color": "darkgray", "ls": "-", "lw": LINEWIDTH, "label": f"{OUTSIDE_STR} PopRI, {INSIDE_STR} PerRI"},
    "C_popOut_perOut": {"color": "tab:red", "ls": "-", "lw": LINEWIDTH, "label": "Out Both"},
}


def year_since_baseline(ts, extend_days=90):
    ts = pd.to_datetime(ts)
    ts = pd.Series(ts) if not isinstance(ts, pd.Series) else ts
    start = ts.min()

    new_point = ts.iloc[-1] + pd.Timedelta(days=extend_days)
    ts_est = pd.to_datetime(np.concatenate([ts.iloc[1:].to_numpy(), [new_point]]))

    years = (ts - start) / pd.Timedelta(days=365.25)
    years_est = (ts_est - start) / pd.Timedelta(days=365.25)
    return years.to_numpy(), years_est.to_numpy()


def add_panel_label(ax, label: str, x: float = -0.65, y: float = 1.05, fontsize: int = 12, fontweight: str = "bold") -> None:
    """Add a panel label (e.g. "a", "b", "e") in axes coordinates, used by the
    pregnancy analysis's combined mosaic."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight=fontweight, ha="left", va="top")


def save_fig_as_svg(fig, title: str, path: Union[str, Path], csv_path: Optional[Union[str, Path]] = None) -> Path:
    """Save `fig` as SVG to `path` and print a two-line "Figure: <path>" (+ "Data: <csv_path>"
    if given) block, so console output links directly to what a run actually produced --
    every script's real replacement for the noisy full manifest.json dump at the end of
    main(). `csv_path` is informational only (any companion CSV is already written
    separately by the caller before this is called, not written here).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    print(f"{title}\n\tFigure:\t{path}")
    if csv_path is not None:
        print(f"\tData:\t{csv_path}")
    return path
