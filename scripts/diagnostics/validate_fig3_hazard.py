"""Render fig3a in the same "one point per marker, hue=model" style as the original
bayesian-vs-gmm comparison, but with the two "models" being this repo's own freshly computed
fig3a_hr_by_model.csv ("validation") and data/UWM/fig3_hazard/'s reference snapshot ("UWM") --
the same underlying idea as utils/ground_truth_check.py's text warnings, as a plot instead.

Shipped (unlike utils/ground_truth_check.py, scripts/diagnostics/compare_to_ground_truth.py,
and tests/test_ground_truth_check.py, which stay maintainer-only) so external validators can run
their own side-by-side check against fig3's ground truth. data/UWM/fig3_hazard/ is shipped for
the same reason; data/UWM/'s other subdirectories (fig4, fig5, setpoints, ...) are not.

Run (after scripts.run_fig3_hazard has produced data/outputs/fig3_hazard/fig3a_hr_by_model.csv):
    python -m scripts.diagnostics.validate_fig3_hazard
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.bootstrap import ensure_importable

ensure_importable()

import matplotlib.pyplot as plt  # noqa: E402

from constants.marker_config import MARKER_IOI_ORDER  # noqa: E402
from constants.runtime import CV_COL, MODEL_COL, MU, TEST_CODE_COL  # noqa: E402
from scripts.run_fig3_hazard import _hr_by_model_df_to_cox_summary  # noqa: E402
from utils.visuals_fig3 import fig3hr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_PATH = ROOT / "data" / "outputs" / "fig3_hazard" / "fig3a_hr_by_model.csv"
UWM_PATH = ROOT / "data" / "UWM" / "fig3_hazard" / "fig3a_hr_by_model.csv"
SAVE_PATH = ROOT / "data" / "outputs" / "fig3_hazard" / "validate_fig3_hazard_uwm_vs_validation.svg"
VARIABLES = (MU, CV_COL)


def _load_labeled(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[MODEL_COL] = label
    return df


def main() -> None:
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"{VALIDATION_PATH} not found -- run `python -m scripts.run_fig3_hazard` first.")
    if not UWM_PATH.exists():
        raise FileNotFoundError(f"{UWM_PATH} not found -- see data/UWM/fig3_hazard/README.md.")

    combined = pd.concat(
        [
            _load_labeled(VALIDATION_PATH, "validation"),
            _load_labeled(UWM_PATH, "UWM"),
        ],
        ignore_index=True,
    )
    cox_summary = _hr_by_model_df_to_cox_summary(combined)

    fig = fig3hr(cox_summary, MARKER_IOI_ORDER, variables=VARIABLES, model_list=["UWM", "validation"], save_path=SAVE_PATH)
    if fig is not None:
        plt.close(fig)
        print(f"Saved: {SAVE_PATH}")
    else:
        print("No overlapping markers between validation and UWM -- nothing to plot.")


if __name__ == "__main__":
    main()
