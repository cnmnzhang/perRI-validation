"""Logistic-regression odds-ratio fitting for the fig4 forest-plot panel.

extract_coef + fit_logit_and_report.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def extract_coef(model, variable, alpha=0.05):
    """Extract coefficient, OR, and CI for a single predictor from a statsmodels Logit model.

    Handles both numeric predictors and categorical predictors encoded by patsy
    (e.g. `sex` -> `sex[T.M]` or `C(sex)[T.M]`).
    """
    param_index = model.params.index

    if variable in param_index:
        pname = variable
    else:
        candidates = [name for name in param_index if (name.startswith(variable + "[") or name.startswith(f"C({variable})") or f"{variable}[" in name)]
        if not candidates:
            raise KeyError(f"Variable '{variable}' not found in model params. Available params: {list(param_index)}")
        pname = candidates[0]

    coef = model.params[pname]
    se = model.bse[pname]
    z = norm.ppf(1 - alpha / 2)
    ci_low = coef - z * se
    ci_high = coef + z * se

    max_log = np.log(np.finfo(float).max)

    def _safe_exp(x: float) -> float:
        if pd.isna(x):
            return np.nan
        if x >= max_log:
            return np.inf
        if x <= -max_log:
            return 0.0
        return float(np.exp(x))

    or_val = _safe_exp(coef)
    or_low = _safe_exp(ci_low)
    or_high = _safe_exp(ci_high)

    return {
        "feature": variable,
        "coef": coef,
        "se": se,
        "odds_ratio": or_val,
        "ci_lower": or_low,
        "ci_upper": or_high,
        "log_odds_ratio": np.log(or_val),
        "log_ci_lower": np.log(or_low),
        "log_ci_upper": np.log(or_high),
        "log_ci_error": [np.log(or_val) - np.log(or_low), np.log(or_high) - np.log(or_val)],
    }


def fit_logit_and_report(
    outcome,
    df,
    exposures,
    covariates,
    multivariable_exposures=None,
    *,
    standardize: bool = False,
    standardize_cols=None,
):
    """
    Fit:
      1) One age/sex-adjusted univariate model for EACH exposure in `exposures`.
      2) One Multivariate model including ALL exposures + ALL covariates.
    """
    df = df.copy()

    if outcome not in df.columns:
        raise ValueError(f"[fit_logit_and_report] Outcome column '{outcome}' not found in df.")

    col = df[outcome]
    if col.dtype == bool:
        df[outcome] = col.astype(int)
    else:
        col_norm = col.astype(str).str.strip().str.lower().replace({"true": "1", "false": "0", "t": "1", "f": "0", "yes": "1", "no": "0", "y": "1", "n": "0"})
        df[outcome] = pd.to_numeric(col_norm, errors="coerce")

    df = df[df[outcome].isin([0, 1])]
    if df.empty or df[outcome].nunique() < 2:
        print(f"[fit_logit_and_report] Outcome '{outcome}' has insufficient variation after coercion. Skipping all models.")
        return pd.DataFrame()

    exposures = list(dict.fromkeys(exposures))
    covariates = list(dict.fromkeys(covariates))

    scales: dict = {}
    if standardize:
        rhs_all = list(dict.fromkeys((exposures or []) + (covariates or []) + (multivariable_exposures or [])))
        cols_to_scale = rhs_all if standardize_cols is None else list(dict.fromkeys(standardize_cols))

        for col_name in cols_to_scale:
            if col_name == outcome or col_name not in df.columns:
                continue

            s = df[col_name]
            if not pd.api.types.is_numeric_dtype(s):
                s_num = pd.to_numeric(s, errors="coerce")
                if float(s_num.notna().mean()) < 0.9:
                    continue
                df[col_name] = s_num
                s = df[col_name]

            if not pd.api.types.is_numeric_dtype(s):
                continue

            vals = s.dropna().to_numpy()
            if vals.size == 0:
                continue

            uniq = np.unique(vals)
            if uniq.size <= 2 and set(uniq.tolist()).issubset({0, 1}):
                continue

            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=0))
            if std == 0.0 or np.isnan(std):
                continue

            df[col_name] = (s - mean) / std
            scales[col_name] = {"mean": mean, "std": std}

    results = []

    for var in exposures:
        rhs_terms = [var] + covariates
        cols_needed = [outcome] + rhs_terms
        d = df[cols_needed].dropna()
        if d.empty or d[outcome].nunique() < 2:
            print(f"[fit_logit_and_report] Skipping univariate model for '{var}' (insufficient data).")
            continue

        formula = f"{outcome} ~ " + " + ".join(rhs_terms)
        uni_model = sm.Logit.from_formula(formula, data=d).fit(disp=0)

        coef = extract_coef(uni_model, var)
        results.append(
            {
                "variable": var,
                "model_type": "uni",
                "standardized": bool(standardize and (var in scales)),
                "x_mean": scales.get(var, {}).get("mean", np.nan),
                "x_std": scales.get(var, {}).get("std", np.nan),
                **coef,
            }
        )

    multi_exposures = multivariable_exposures if multivariable_exposures is not None else exposures

    if multi_exposures:
        rhs_terms = multi_exposures + covariates
        cols_needed = [outcome] + rhs_terms
        d = df[cols_needed].dropna()
        if d.empty or d[outcome].nunique() < 2:
            print("[fit_logit_and_report] Skipping multivariable model (insufficient data).")
        else:
            full_formula = f"{outcome} ~ " + " + ".join(rhs_terms)
            try:
                full_model = sm.Logit.from_formula(full_formula, data=d).fit(disp=0)
            except Exception as e:
                print(f"[fit_logit_and_report] Multivariable model failed: {e}")
            else:
                for var in exposures + covariates:
                    if var not in full_model.params.index:
                        continue
                    coef = extract_coef(full_model, var)
                    results.append(
                        {
                            "variable": var,
                            "model_type": "multi",
                            "standardized": bool(standardize and (var in scales)),
                            "x_mean": scales.get(var, {}).get("mean", np.nan),
                            "x_std": scales.get(var, {}).get("std", np.nan),
                            **coef,
                        }
                    )

    return pd.DataFrame(results)
