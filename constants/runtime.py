"""Runtime and schema constants."""

# -----------------------
# Column names (schema)
# -----------------------
ID_COL = "anon_id"
TS_COL = "ts"
MEASUREMENT_COL = "result_value"
MU = "mu"
SIGMA = "sigma"
CV_COL = "cv"
TEST_CODE_COL = "test_code"
SEX_COL = "sex"
MODEL_COL = "model"
INDEX_COL = "index"
PRESENT_VAL_COL = "presenting_value"
PRESENT_TS_COL = "presenting_ts"

# -----------------------
# Dx table column names
# -----------------------
ICD9_COL = "icd9"
ICD10_COL = "icd10"
DIAGNOSIS_TS_COL = "diagnosis_ts"

# -----------------------
# Clinical output column names (from add_oo)
# -----------------------
PERRI_Z_SCORE_COL = "perRI_z_score"  # Continuous z-score: (result - setpoint) / sigma
POP_LO_COL = "pop_lo"  # Population reference range lower bound
POP_HI_COL = "pop_hi"  # Population reference range upper bound
OUT_POPRI_COL = "out_popri"  # Binary: outside population reference interval
OUT_POPRI_LOWER_COL = "out_popri_lower"  # Binary: below population RI
OUT_POPRI_UPPER_COL = "out_popri_upper"  # Binary: above population RI
DELTA_COL = "delta"  # Difference: result - setpoint

# These are built dynamically based on confidence level but commonly use p=0.95
OUT_PERRI_P_PREFIX = "out_perri_p"  # Base for personal RI flags; full names e.g., "out_perri_p95"
OUT_PERRI_LOWER_SUFFIX = "_lower"  # Appended for lower bound flags
OUT_PERRI_UPPER_SUFFIX = "_upper"  # Appended for upper bound flags

# Default confidence level columns (p=0.95)
OUT_PERRI_P95_COL = "out_perri_p95"  # Binary: outside personal RI at 95% confidence
OUT_PERRI_P95_LOWER_COL = "out_perri_p95_lower"  # Binary: below personal RI at 95% confidence
OUT_PERRI_P95_UPPER_COL = "out_perri_p95_upper"  # Binary: above personal RI at 95% confidence

# -----------------------
# RI classification label strings
# -----------------------
INSIDE_STR = "In"
OUTSIDE_STR = "Out"
ABOVE_STR = "Above"
BELOW_STR = "Below"

# -----------------------
# Runtime settings
# -----------------------
RANDOM_SEED = 42

# compute_sp_df's joblib worker count for setpoint fitting (patient-level parallelism).
# Benchmarked on a 128-core machine: chunking patients across 16 workers gave ~6x
# wall-clock speedup over serial; going higher (32/64/128) was flat-to-worse, since each
# worker pays its own numba JIT warmup for perri's @njit bayesian() on first call -- more
# workers means more redundant warmups without a matching gain (each per-patient fit is
# sub-millisecond once warm). Set to 1 to force serial (e.g. for debugging).
N_JOBS = 3

# compute_sp_df's default minimum isolated measurements per patient to fit a setpoint --
# shared with is_fitted/is_fitted_canonical so their cache-key lookups match what
# compute_sp_df actually cached under, without both sides having to be kept in sync by hand.
DEFAULT_MIN_MEASUREMENTS = 3

# -----------------------
# Date constants
# -----------------------
MAX_FIT_DATE = "2019-01-01"
SURVIVAL_YEARS = 5


__all__ = sorted(name for name, value in globals().items() if name.isupper())
