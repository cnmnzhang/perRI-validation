"""Generic caching for compute-heavy functions.

Primary entry point: ``cache_or_compute``.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Union

import pandas as pd


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash a DataFrame's full CSV representation (stable, order-independent via sort)."""
    try:
        df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    except TypeError:
        # Mixed types that can't be compared — fall back to unsorted
        df_sorted = df.reset_index(drop=True)
    csv_bytes = df_sorted.to_csv(index=False).encode()
    return hashlib.sha256(csv_bytes).hexdigest()


def cache_or_compute(
    path: Union[str, Path],
    compute_fn: Callable[[], Any],
    force: bool = False,
    file_format: str = "auto",
) -> Any:
    """Generic caching pattern: compute once, then read from `path` on later calls.

    Args:
        path: Path to cache file.
        compute_fn: Function to compute data if not cached.
        force: If True, recompute even if cache exists.
        file_format: File format for saving. Options: "auto", "pickle", "csv", "parquet".
            "auto" infers from file extension.

    Returns:
        Cached or computed data.
    """
    path = Path(path)

    if file_format == "auto":
        suffix = path.suffix.lower()
        if suffix in (".pkl", ".pickle"):
            file_format = "pickle"
        elif suffix == ".csv":
            file_format = "csv"
        elif suffix == ".parquet":
            file_format = "parquet"
        else:
            file_format = "pickle"

    if path.exists() and not force:
        print(f"[cache] hit {path.name}")
        if file_format == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif file_format == "csv":
            return pd.read_csv(path)
        elif file_format == "parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unknown file format: {file_format}")

    if path.exists() and force:
        print(f"[cache] force recompute {path.name}")
    else:
        print(f"[cache] miss {path.name}")
    data = compute_fn()

    path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif file_format == "csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            raise ValueError("CSV format requires DataFrame")
    elif file_format == "parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path)
        else:
            raise ValueError("Parquet format requires DataFrame")

    print(f"[cache] saved {path.name}")

    return data
