"""ICD9/ICD10 -> diagnosis-name mapping (DX2ICD).

Only the 11 diagnoses actually consumed downstream: fig3_dx's 8 non-cancer
DX2SETPOINT diagnoses, plus fig4_dx_cases's 3 outcomes (acute kidney injury,
leukemia, hypothyroidism). Also includes reverse mappings from ICD9 and ICD10
codes to diagnosis names.
"""

from typing import Dict, List

DX2ICD = {
    "Hypothyroidism": {
        "ICD9": ["244"],  # acquired hypothyroidism family
        "ICD10": ["E03"],  # hypothyroidism (noncongenital) family
    },
    "Acute kidney injury": {
        "ICD9": ["584"],
        "ICD10": ["N17"],
    },
    "Iron Deficiency anemia": {
        "ICD9": ["280"],
        "ICD10": ["D50"],
    },
    "Macrocytic anemia": {
        # Macrocytic is most often megaloblastic from B12/folate, so include those families.
        "ICD9": ["281"],  # pernicious/B12/folate deficiency anemias, megaloblastic
        "ICD10": ["D51", "D52", "D53.1"],  # B12, folate, other megaloblastic anemia,
        #  excluding non‑megaloblastic macrocytic anemia
    },
    "Chronic kidney disease": {
        "ICD9": ["585", "585.1", "585.2", "585.3", "585.4", "585.5", "585.6", "585.9"],
        "ICD10": ["N18", "N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"],
    },
    "Leukemia": {
        "ICD9": ["204", "205", "206", "207", "208", "202.4"],  # 202.4x = leukemic reticuloendotheliosis (hairy cell leukemia), not lymphoma
        "ICD10": ["C91", "C92", "C93", "C94", "C95"],
    },
    "Heart failure": {
        "ICD9": ["428"],
        "ICD10": ["I50"],
    },
    "Cirrhosis": {
        "ICD9": ["571.2", "571.5", "571.6"],
        "ICD10": ["K74.3", "K74.4", "K74.5", "K74.6"],
    },
    "Nonalcoholic fatty liver disease": {
        "ICD9": ["571.8"],
        "ICD10": ["K76.0", "K75.81"],
    },
    "Neuropathy": {
        "ICD9": ["356.9"],
        "ICD10": ["G62.9"],
    },
    "Myeloproliferative neoplasms": {
        "ICD9": ["238.4"],
        "ICD10": ["D45"],
    },
}

ICD9_2_DX: Dict[str, List[str]] = {}
for diagnosis_name, info in DX2ICD.items():
    for icd9_code in info.get("ICD9", []):
        ICD9_2_DX.setdefault(icd9_code, []).append(diagnosis_name)

ICD10_2_DX: Dict[str, List[str]] = {}
for diagnosis_name, info in DX2ICD.items():
    for icd10_code in info.get("ICD10", []):
        ICD10_2_DX.setdefault(icd10_code, []).append(diagnosis_name)


# fig3_dx's non-cancer KM panel: diagnosis -> setpoint marker + percentile cutoff.
DX2SETPOINT = {
    "Iron Deficiency anemia": [{"setpoint_type": "HB", "pct_cutoff": 25}],
    "Myeloproliferative neoplasms": [{"setpoint_type": "TNEUT", "pct_cutoff": 75}],
    "Cirrhosis": [{"setpoint_type": "ALB", "pct_cutoff": 25}],
    "Nonalcoholic fatty liver disease": [{"setpoint_type": "ALT", "pct_cutoff": 75}],
    "Macrocytic anemia": [{"setpoint_type": "MCV", "pct_cutoff": 75}],
    "Chronic kidney disease": [{"setpoint_type": "P", "pct_cutoff": 75}],
    "Neuropathy": [{"setpoint_type": "GLU", "pct_cutoff": 75}],
    "Heart failure": [{"setpoint_type": "K", "pct_cutoff": 75}],
}

COMBINED_ORDER = [
    "Iron Deficiency anemia",
    "Myeloproliferative neoplasms",
    "Cirrhosis",
    "Nonalcoholic fatty liver disease",
    "Macrocytic anemia",
    "Chronic kidney disease",
    "Neuropathy",
    "Heart failure",
]

__all__ = ["DX2ICD", "ICD9_2_DX", "ICD10_2_DX", "DX2SETPOINT", "COMBINED_ORDER"]
