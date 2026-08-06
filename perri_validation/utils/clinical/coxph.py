"""Mortality Cox Proportional Hazards analysis.

Used by fig3a/b's hazard-ratio panels: one Cox fit per (marker, variable in
{mu, sigma, cv}[, baseline_index]), covariate standardized + adjusted for
age/sex, event = death within SURVIVAL_YEARS of the setpoint estimate's
timestamp.
"""

import datetime
import warnings

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceError
from numpy.linalg import cond
from sklearn.preprocessing import StandardScaler

from perri_validation.constants.runtime import MAX_FIT_DATE, SEX_COL, SURVIVAL_YEARS, TS_COL


def coxph_analysis(
    cox_df: pd.DataFrame,
    eval_col: str,
    survival_time_col: str = "duration",
    event_col: str = "event",
    additional_covariates: list = None,
) -> dict:
    """Fit a Cox Proportional Hazards model and return its hazard-ratio summary."""
    predictors = [eval_col] + (additional_covariates if additional_covariates else [])
    cph_data = cox_df[[survival_time_col, event_col] + predictors].dropna()

    cph = CoxPHFitter()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            cph.fit(cph_data, duration_col=survival_time_col, event_col=event_col)
        except ConvergenceError as e:
            print(f"Convergence error: {str(e)}")
            return {}
        except Exception as e:
            print(f"Unexpected error during fitting: {e}")
            return {}

    if cph is None:
        print("CoxPHFitter is None after fitting")
        return {}

    coef = cph.summary.loc[eval_col, "exp(coef)"]
    ci_lower = cph.summary.loc[eval_col, "exp(coef) lower 95%"]
    ci_upper = cph.summary.loc[eval_col, "exp(coef) upper 95%"]
    p = cph.summary.loc[eval_col, "p"]

    if any(pd.isna([ci_lower, ci_upper, coef])):
        print("CoxPHFitter summary contains NaN values")
        return {}

    return {
        "exp(coef)": coef,
        "exp(coef) lower 95%": ci_lower,
        "exp(coef) upper 95%": ci_upper,
        "p": p,
        "n": cph_data.shape[0],
    }


def run_coxph_analysis(
    results_df: pd.DataFrame,
    variable: str,
    date_calculation=MAX_FIT_DATE,
    years: float = SURVIVAL_YEARS,
    date_col: str = TS_COL,
    use_personalized_logic: bool = True,
) -> dict:
    """Run mortality Cox PH analysis for one setpoint variable (mu/sigma/cv).

    Expects results_df to already carry death_ts, birth_date, sex, and `variable`.
    """
    results_df2 = results_df.dropna(subset=[variable], inplace=False).copy()

    if len(results_df2) == 0:
        print("Skipping due to empty results_df2")
        return {}

    if results_df2[variable].nunique() <= 1 or results_df2[variable].std() < 1e-6:
        print(f"Skipping due to low variance in {variable}")
        return {}

    results_df2[variable] = StandardScaler().fit_transform(results_df2[variable].values.reshape(-1, 1))

    cox_df = results_df2.copy()
    cox_df.loc[cox_df["death_ts"].isnull(), "death_ts"] = datetime.datetime.now()
    cox_df["death_ts"] = pd.to_datetime(cox_df["death_ts"], format="mixed").copy()

    if "age" not in cox_df.columns:
        cox_df["birth_date"] = pd.to_datetime(cox_df["birth_date"]).copy()
        cox_df["age"] = (cox_df["death_ts"] - cox_df["birth_date"]).dt.days // 365

    if use_personalized_logic:
        start_dates = pd.to_datetime(cox_df[date_col])
    else:
        start_dates = pd.to_datetime(date_calculation)

    cox_df["duration"] = (pd.to_datetime(cox_df["death_ts"]) - start_dates).dt.days
    cox_df["event"] = cox_df["duration"] <= years * 365

    prepared_df = cox_df.drop(columns=["death_ts"])

    if prepared_df["event"].value_counts().shape[0] < 2:
        print(f"Skipping due to lack of events: {variable}")
        return {}

    additional_covariates = ["age"]

    for col in additional_covariates:
        if prepared_df[col].nunique() <= 1 or prepared_df[col].isnull().all():
            print(f"Skipping due to constant or null covariate: {col}")
            return {}

    if SEX_COL in prepared_df.columns:
        unique_sex_values = prepared_df[SEX_COL].dropna().unique()
        if set(unique_sex_values) == {"F", "M"}:
            prepared_df[SEX_COL] = prepared_df[SEX_COL].map({"F": 0, "M": 1})
            additional_covariates.append(SEX_COL)

    x_cols = [variable] + additional_covariates
    x = prepared_df[x_cols].dropna().values
    x = StandardScaler().fit_transform(x)

    condition_number = cond(x)
    if condition_number > 1e4:
        print(f"Skipping due to high collinearity: condition number = {condition_number:.2e}")
        return {}

    return coxph_analysis(prepared_df, eval_col=variable, additional_covariates=additional_covariates)


def run_cox_summary(results_df: pd.DataFrame) -> dict:
    """Run Cox PH for mu, sigma, and cv on one (test_code, model) group."""
    results_df = results_df.copy()
    if "cv" not in results_df.columns:
        results_df["cv"] = (results_df["sigma"] / results_df["mu"]).clip(0.001, 0.999)
    return {
        "mu": run_coxph_analysis(results_df=results_df, variable="mu"),
        "sigma": run_coxph_analysis(results_df=results_df, variable="sigma"),
        "cv": run_coxph_analysis(results_df=results_df, variable="cv"),
    }
