"""Site bounds are built from complete, pre-split marker files."""

import numpy as np
import pandas as pd
import pytest

from scripts.build_parametric_bounds import run
from utils.site_bounds import get_site_bounds


def test_a1c_uses_provided_site_percentiles_and_clinical_sigma(tmp_path):
    input_dir = tmp_path / "data"
    marker_dir = input_dir / "cache" / "splits_by_marker"
    marker_dir.mkdir(parents=True)
    rows = [
        ("p1", "2015-01-01", 4.5), ("p2", "2015-01-01", 5.0),
        ("p3", "2015-01-01", 6.0), ("p4", "2015-01-01", 10.0),
    ]
    pd.DataFrame(rows, columns=["anon_id", "ts", "result_value"]).assign(test_code="A1C", sex="F").to_csv(marker_dir / "A1C.csv", index=False)

    manifest = run(input_dir=input_dir, output_dir=tmp_path / "out")
    saved = get_site_bounds("A1C", "ALL", input_dir=input_dir)

    assert manifest["missing_marker_files"] == ["HDL"]
    assert saved["n_measurements"] == 4
    assert saved["bounds"]["min_mu"] == pytest.approx(np.percentile([4.5, 5, 6, 10], 0.5))
    assert saved["bounds"]["max_mu"] == pytest.approx(np.percentile([4.5, 5, 6, 10], 99.5))
    assert saved["bounds"]["max_sigma"] == pytest.approx((5.6 - 4.0) / 3.92)
