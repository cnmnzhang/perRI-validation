"""Kaplan-Meier 2x2 analysis input object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_KM_STYLES = {
    "A_popIn_perIn": {
        "color": "tab:blue",
        "linestyle": "-",
        "lw": 1.5,
        "label": "PopRI In, PerRI In",
    },
    "B_popIn_perOut": {
        "color": "tab:orange",
        "linestyle": "-",
        "lw": 1.5,
        "label": "PopRI In, PerRI Out",
    },
    "C_popOut_perIn": {
        "color": "tab:green",
        "linestyle": "--",
        "lw": 1.5,
        "label": "PopRI Out, PerRI In",
    },
    "C_popOut_perOut": {
        "color": "tab:red",
        "linestyle": "-",
        "lw": 1.5,
        "label": "PopRI Out, PerRI Out",
    },
}


@dataclass
class KMExclusiveInputs:
    """Kaplan-Meier 2x2 analysis inputs (PerRI x PopRI)."""

    km_all: pd.DataFrame
    masks: Dict[str, pd.Series]
    window_days: int
    legend_title: Optional[str] = None

    @staticmethod
    def _resolve_columns(
        *,
        out_popri_col: Optional[str],
        lower: bool,
        upper: bool,
    ) -> tuple[str, str]:
        """Resolve PopRI/PerRI flag columns without importing repo config."""
        if out_popri_col is None:
            if lower and upper:
                out_popri_col = "out_popri"
            elif lower:
                out_popri_col = "out_popri_lower"
            elif upper:
                out_popri_col = "out_popri_upper"
            else:
                raise ValueError("At least one of lower or upper must be True.")

        if lower and upper:
            per_col = "out_perri_p95"
        elif lower:
            per_col = "out_perri_p95_lower"
        elif upper:
            per_col = "out_perri_p95_upper"
        else:
            raise ValueError("At least one of lower or upper must be True.")

        return out_popri_col, per_col

    @staticmethod
    def _compute_km_frame(
        df: pd.DataFrame,
        *,
        window_days: int,
        presenting_col: str,
    ) -> pd.DataFrame:
        """Compute duration and event-observed columns for KM plotting."""
        required = [presenting_col, "first_in_window_event", "any_in_window"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for KM frame: {missing}")

        out = df.copy()
        out[presenting_col] = pd.to_datetime(out[presenting_col], errors="coerce")
        out["event_date"] = pd.to_datetime(out["first_in_window_event"], errors="coerce")
        event_in_window = out["any_in_window"].fillna(False).astype(bool)

        win_hi = out[presenting_col] + pd.Timedelta(days=int(window_days))
        event_time = (out["event_date"] - out[presenting_col]).dt.days
        censor_time = (win_hi - out[presenting_col]).dt.days

        out["duration_days"] = np.where(event_in_window, event_time, censor_time).astype(float)
        out["duration_days"] = np.clip(out["duration_days"], a_min=0, a_max=int(window_days)).astype(float)
        out["event_observed"] = event_in_window.astype(int)
        out = out[out["duration_days"] > 0].copy()

        cols = ["duration_days", "event_observed"]
        if presenting_col in out.columns:
            cols.append(presenting_col)
        return out[cols]

    @staticmethod
    def _print_cohort_summary(df: pd.DataFrame) -> str:
        """Print the compact cohort summary used in Fig. 4 KM legends."""
        total = len(df)
        print(f"\n▶ Cohort size: {total}")
        date_cols = [c for c in df.columns if c.endswith("_date")]
        for date_col in date_cols:
            root = date_col[: -len("_date")]
            in_col = f"{root}_in_window"
            if date_col in df.columns and in_col in df.columns:
                n_any = df[date_col].notna().sum()
                n_in = int(df[in_col].sum())
                line = f"  {root}: any_date {n_any}/{total} ({n_any / total:.1%})"
                line += f" | in_window (raw) {n_in}/{total} ({n_in / total:.1%})"
                print(line)

        n_raw_in_window = int(df["any_in_window"].fillna(False).sum()) if "any_in_window" in df.columns else 0
        print("  --------------------------------------------------")
        print(f"  any_in_window (Raw): {n_raw_in_window}/{total} ({n_raw_in_window / total:.1%})")
        return f"{n_raw_in_window}/{total} ({n_raw_in_window / total:.1%})"

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        window_days: int,
        out_popri_col: Optional[str] = None,
        presenting_col: str = "presenting_ts",
        lower: bool = True,
        upper: bool = True,
    ) -> "KMExclusiveInputs":
        """Build KM-exclusive inputs from a cohort dataframe."""
        out_popri_col, per_col = cls._resolve_columns(
            out_popri_col=out_popri_col,
            lower=lower,
            upper=upper,
        )
        required = [out_popri_col, per_col, presenting_col, "first_in_window_event", "any_in_window"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for KMExclusiveInputs: {missing}")

        km_all = cls._compute_km_frame(df, window_days=window_days, presenting_col=presenting_col)

        pop_in = df[out_popri_col] == 0
        pop_out = df[out_popri_col] == 1
        per_in = df[per_col] == 0
        per_out = df[per_col] == 1

        masks = {
            "A_popIn_perIn": pop_in & per_in,
            "B_popIn_perOut": pop_in & per_out,
            "C_popOut_perIn": pop_out & per_in,
            "C_popOut_perOut": pop_out & per_out,
        }
        legend_title = cls._print_cohort_summary(df)
        return cls(km_all=km_all, masks=masks, window_days=window_days, legend_title=legend_title)

    def plot_on_ax(
        self,
        ax,
        *,
        ylabel: str = "Patients without (%)",
        xlabel: str = "Years after presenting",
        add_legend: bool = True,
        legend_fontsize: float = 6,
        styles: Optional[dict] = None,
    ):
        """Plot the four KM-exclusive curves on an existing Matplotlib axes."""
        from lifelines import KaplanMeierFitter

        style_map = styles or DEFAULT_KM_STYLES
        mins = []

        for style_key, mask in self.masks.items():
            if not mask.any():
                continue

            sub = self.km_all.loc[mask]
            if sub.empty:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(sub["duration_days"], event_observed=sub["event_observed"])

            surv = kmf.survival_function_.copy()
            if 0.0 not in surv.index:
                surv.loc[0.0] = 1.0
            surv = surv.sort_index()
            if float(self.window_days) not in surv.index:
                surv = surv.reindex(surv.index.union([float(self.window_days)])).ffill()

            surv_vals = surv.iloc[:, 0].to_numpy()
            mins.append(float(surv_vals.min()))

            style_props = style_map.get(style_key, {}).copy()
            label = style_props.pop("label", style_key)
            ax.step(surv.index, surv_vals, where="post", label=label, **style_props)

        tick_years = np.arange(0, self.window_days + 1, 365)
        ax.set_xticks(tick_years)
        ax.set_xticklabels([str(int(t / 365)) for t in tick_years])
        ax.set_xlim(0, self.window_days + 1)

        if mins:
            ax.set_ylim(min(mins), 1.0)
            ax.set_yticks(np.linspace(min(mins), 1.0, 4))
            ax.set_yticklabels([f"{y:.3f}" for y in ax.get_yticks()])

        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_ylabel(ylabel, fontsize=6)
        if add_legend:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=legend_fontsize)
        return ax

    def plot(
        self,
        *,
        figsize: tuple[float, float] = (4, 3),
        ylabel: str = "Patients without (%)",
        xlabel: str = "Years after presenting",
        add_legend: bool = True,
        legend_fontsize: float = 6,
        styles: Optional[dict] = None,
    ):
        """Create a Matplotlib figure with the KM-exclusive curves."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        self.plot_on_ax(
            ax,
            ylabel=ylabel,
            xlabel=xlabel,
            add_legend=add_legend,
            legend_fontsize=legend_fontsize,
            styles=styles,
        )
        return fig, ax
