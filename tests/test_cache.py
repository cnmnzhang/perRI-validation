"""Tests for utils/cache.py: cache_or_compute + hash_dataframe.

Uses the low-level perri_validation.utils.cache.cache_or_compute (which takes a
full path) rather than perri_validation.utils.setpoints's rooted wrapper (which
always roots at the real data/cache/) -- tests must never touch
the real, shared cache directory. These tests are deliberately isolated via
tmp_path.
"""

import pandas as pd
import pytest

from utils.cache import cache_or_compute, hash_dataframe


def test_cache_miss_then_hit(tmp_path):
    path = tmp_path / "result.csv"
    calls = []

    def _compute():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2, 3]})

    first = cache_or_compute(path, _compute, file_format="csv")
    second = cache_or_compute(path, _compute, file_format="csv")

    assert len(calls) == 1  # only computed once
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_force_recomputes_even_on_hit(tmp_path):
    path = tmp_path / "result.csv"
    calls = []

    def _compute():
        calls.append(1)
        return pd.DataFrame({"a": [len(calls)]})

    cache_or_compute(path, _compute, file_format="csv")
    cache_or_compute(path, _compute, file_format="csv", force=True)

    assert len(calls) == 2


def test_pickle_format_roundtrips_arbitrary_objects(tmp_path):
    path = tmp_path / "result.pkl"
    out = cache_or_compute(path, lambda: {"nested": [1, 2, {"x": "y"}]}, file_format="pickle")
    assert out == {"nested": [1, 2, {"x": "y"}]}

    cached = cache_or_compute(path, lambda: pytest.fail("should not recompute"), file_format="pickle")
    assert cached == {"nested": [1, 2, {"x": "y"}]}


def test_hash_dataframe_is_order_independent():
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = df1.iloc[::-1].reset_index(drop=True)
    assert hash_dataframe(df1) == hash_dataframe(df2)


def test_hash_dataframe_differs_on_content_change():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1, 3]})
    assert hash_dataframe(df1) != hash_dataframe(df2)
