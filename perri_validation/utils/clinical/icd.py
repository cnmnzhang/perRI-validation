"""ICD9/ICD10 -> first-diagnosis-date mapping.

``dx_all_to_first_fast`` and its helpers (the vectorized path).
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from perri_validation.constants.icd_config import ICD10_2_DX, ICD9_2_DX
from perri_validation.constants.runtime import ID_COL


def _flatten_columns_inplace(df: pd.DataFrame) -> None:
    """If columns are a MultiIndex (e.g., ('epic_pat_id',)), flatten to strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [(c[0] if isinstance(c, tuple) and len(c) == 1 else "_".join(map(str, c))) for c in df.columns]


def _normalize_code_series(s: pd.Series) -> pd.Series:
    # keep missing as NA, use pandas string dtype, strip spaces, remove dots, uppercase
    s = s.astype("string")
    s = s.str.strip().str.upper().str.replace(".", "", regex=False)
    return s


def _build_prefix_layers(mapping: Dict[str, List[str]]) -> List[tuple[int, Dict[str, List[str]]]]:
    """
    Normalize keys (remove dots, uppercase) and group by prefix length.
    Returns list of (length, dict) sorted by length desc so longest-prefix matches first.
    """
    norm = {str(k).replace(".", "").upper(): v for k, v in mapping.items()}
    layers: Dict[int, Dict[str, List[str]]] = {}
    for k, v in norm.items():
        L = len(k)
        if L == 0:  # skip empty keys defensively
            continue
        d = layers.get(L)
        if d is None:
            layers[L] = {k: v}
        else:
            d[k] = v
    return [(L, layers[L]) for L in sorted(layers.keys(), reverse=True)]


def _vectorized_map_by_prefix(codes: pd.Series, layers: List[tuple[int, Dict[str, List[str]]]]) -> pd.Series:
    """
    Vectorized longest-prefix match.
    - Normalizes codes (remove dots, uppercase)
    - Iterates a few passes by prefix length (longest -> shortest)
    - Uses slicing + dict map; no per-row loops.
    Returns a Series of python lists (dx names) or <NA>.
    """
    vals = _normalize_code_series(codes)
    out = pd.Series(pd.NA, index=vals.index, dtype="object")
    for L, d in layers:
        mask = out.isna() & vals.notna() & (vals.str.len() >= L)
        if not mask.any():
            continue
        sl = vals[mask].str[:L]
        mapped = sl.map(d)  # list or NA
        has = mapped.notna()
        if has.any():
            out.loc[mapped.index[has]] = mapped[has]
    return out


def dx_all_to_first_fast(
    dx_all: pd.DataFrame,
    *,
    id_col: str = ID_COL,
    icd9_col: str = "icd9",
    icd10_col: str = "icd10",
    ts_col: str = "diagnosis_ts",
    prefer: str = "icd10",  # or "icd10"
) -> pd.DataFrame:
    """
    Fast conversion of all diagnoses -> first occurrence per (patient, diagnosis_name).
    Uses layered longest-prefix matching against ICD9_2_DX and ICD10_2_DX.
    - Maps once per unique code (via vectorized passes), then joins back
    - No row-wise apply
    - Earliest date via groupby min (no global sort)

    Returns columns: [id_col, "diagnosis_name", "earliest_contact_date"]
    """

    # 0) Flatten columns if needed (fixes KeyError on MultiIndex like ('epic_pat_id',))
    _flatten_columns_inplace(dx_all)

    # 1) Deduplicate obvious duplicates
    subset_cols = [c for c in [id_col, icd9_col, icd10_col, ts_col] if c in dx_all.columns]
    if subset_cols:
        dx_all = dx_all.drop_duplicates(subset=subset_cols).copy()
    else:
        dx_all = dx_all.copy()

    # 2) Build layered prefix lookups once
    icd9_layers = _build_prefix_layers(ICD9_2_DX)
    icd10_layers = _build_prefix_layers(ICD10_2_DX)

    # 3) Map per unique code and broadcast back
    if icd9_col in dx_all.columns and icd9_layers:
        u9 = pd.Series(dx_all[icd9_col].dropna().astype("string").unique(), name=icd9_col)
        m9 = pd.DataFrame({icd9_col: u9})
        m9["dx_list9"] = _vectorized_map_by_prefix(m9[icd9_col], icd9_layers)
        dx_all = dx_all.merge(m9, on=icd9_col, how="left")
    else:
        dx_all["dx_list9"] = pd.NA

    if icd10_col in dx_all.columns and icd10_layers:
        u10 = pd.Series(dx_all[icd10_col].dropna().astype("string").unique(), name=icd10_col)
        m10 = pd.DataFrame({icd10_col: u10})
        m10["dx_list10"] = _vectorized_map_by_prefix(m10[icd10_col], icd10_layers)
        dx_all = dx_all.merge(m10, on=icd10_col, how="left")
    else:
        dx_all["dx_list10"] = pd.NA

    # 4) Choose ICD9 first or ICD10 first per your preference
    has9 = dx_all["dx_list9"].notna()
    has10 = dx_all["dx_list10"].notna()
    if prefer.lower() == "icd10":
        dx_all["diagnosis_name_list"] = np.where(has10, dx_all["dx_list10"], np.where(has9, dx_all["dx_list9"], pd.NA))
        dx_all["matched_by"] = np.where(has10, "icd10", np.where(has9, "icd9", pd.NA))
    else:
        dx_all["diagnosis_name_list"] = np.where(has9, dx_all["dx_list9"], np.where(has10, dx_all["dx_list10"], pd.NA))
        dx_all["matched_by"] = np.where(has9, "icd9", np.where(has10, "icd10", pd.NA))

    # quick stats (optional)
    n_total = len(dx_all)
    n_matched = int((has9 | has10).sum())
    print(f"Matched: {n_matched} / {n_total} ({n_total - n_matched} unmatched)")
    print(dx_all["matched_by"].value_counts(dropna=False).to_string())
    print("with preference:", prefer)

    # 5) Explode and reduce to first date
    need_cols = [id_col, "diagnosis_name_list", ts_col]
    need_cols = [c for c in need_cols if c in dx_all.columns]
    out = dx_all.loc[dx_all["diagnosis_name_list"].notna(), need_cols].copy()
    out = out.explode("diagnosis_name_list", ignore_index=True).rename(columns={"diagnosis_name_list": "diagnosis_name"})
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")

    dx_incident = out.groupby([id_col, "diagnosis_name"], observed=True)[ts_col].min().reset_index().rename(columns={ts_col: "earliest_contact_date"})
    dx_incident["earliest_contact_date"] = pd.to_datetime(dx_incident["earliest_contact_date"])
    dx_incident["diagnosis_name"] = dx_incident["diagnosis_name"].astype("category")

    return dx_incident
