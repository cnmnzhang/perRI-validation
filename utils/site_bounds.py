"""Cache empirical grid bounds from a site's complete, already-isolated marker files."""

import hashlib
import pandas as pd
from perri.bounds import resolve_grid_bounds

from constants.marker_lab_config import MARKER_CONFIG
from constants.marker_config import POP_REF_INTERVAL
from constants.runtime import MEASUREMENT_COL, SEX_COL
from utils.cache import cache_or_compute
from utils.io import splits_by_marker_dir


def calibration_tag(test_code: str, input_dir=None) -> str:
    """Change when a marker split or its bound policy/clinical RI changes."""
    source = splits_by_marker_dir(input_dir) / f"{test_code}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Complete marker file required for site calibration: {source}")
    cfg = MARKER_CONFIG[test_code]
    stat = source.stat()
    payload = (test_code, str(source.resolve()), stat.st_size, stat.st_mtime_ns,
               cfg.get("mu_bound_policy"), cfg.get("pop_ri"))
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def get_site_bounds(test_code: str, sex: str, *, input_dir=None, full_marker_df=None, force: bool = False) -> dict:
    """Build from complete site rows or read the existing artifact for fitting."""
    policy = MARKER_CONFIG[test_code]["mu_bound_policy"]
    source = splits_by_marker_dir(input_dir) / f"{test_code}.csv"
    tag = calibration_tag(test_code, input_dir)
    artifact = source.parent.parent / "site_bounds" / f"{test_code}_{sex}_{tag}.json"

    def compute():
        if full_marker_df is None:
            raise FileNotFoundError(f"Missing {artifact}; run python -m scripts.build_parametric_bounds first")
        rows = full_marker_df if sex == "ALL" else full_marker_df[full_marker_df[SEX_COL] == sex]
        values = pd.to_numeric(rows[MEASUREMENT_COL], errors="coerce").dropna()
        bounds = resolve_grid_bounds(POP_REF_INTERVAL[sex][test_code], policy, values.to_numpy())
        return {
            "test_code": test_code, "sex": sex, "method": policy["method"],
            "percentiles": list(policy["percentiles"]), "n_measurements": int(len(values)),
            "source": str(source.resolve()), "source_tag": tag, "bounds": bounds,
        }

    return cache_or_compute(artifact, compute, force=force, file_format="json")


