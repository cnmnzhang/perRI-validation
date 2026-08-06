# ── Master marker configuration ───────────────────────────────────────────────
# Single source of truth for battery grouping, units, IOI display ordering, and
# sex-specific population reference intervals. Edit this file to adapt to a different
# population -- see README.md's "Adapting to a different population".
#
#
# Fields
#   battery  – panel/battery name
#   units    – display units string
#   ioi      – index-of-individuality score used to order markers in fig3a/b
#   pop_ri   – sex-keyed reference interval tuples  {sex: (lower, upper)}
MARKER_CONFIG = {
    # ── CBC ───────────────────────────────────────────────────────────────────
    "MCH":   {"battery": "CBC",          "units": "pg",        "full_name": "Mean Corpuscular Hemoglobin",             "pop_ri": {"F": (27.3, 33.6),  "M": (27.3, 33.6)}},
    "WBC":   {"battery": "CBC",          "units": "x10^3/uL",  "full_name": "White Blood Cell Count",                  "pop_ri": {"F": (4.3, 10.0),   "M": (4.3, 10.0)}},
    "PLT":   {"battery": "CBC",          "units": "x10^3/uL",  "full_name": "Platelets",                               "pop_ri": {"F": (150, 400),    "M": (150, 400)}},
    "HB":    {"battery": "CBC",          "units": "g/dL",      "full_name": "Hemoglobin",                              "pop_ri": {"F": (11.5, 15.5),  "M": (13.0, 18.0)}},
    "HCT":   {"battery": "CBC",          "units": "%",         "full_name": "Hematocrit",                              "pop_ri": {"F": (36.0, 45.0),  "M": (38.0, 50.0)}},
    "MCV":   {"battery": "CBC",          "units": "fL",        "full_name": "Mean Corpuscular Volume",                 "pop_ri": {"F": (81, 98),       "M": (81, 98)}},
    "MCHC":  {"battery": "CBC",          "units": "g/dL",      "full_name": "Mean Corpuscular Hemoglobin Concentration","pop_ri": {"F": (32.2, 36.5),  "M": (32.2, 36.5)}},
    "RBC":   {"battery": "CBC",          "units": "x10^6/uL",  "full_name": "Red Blood Cell Count",                    "pop_ri": {"F": (3.8, 6.0),    "M": (4.4, 5.6)}},
    "RDWCV": {"battery": "CBC",          "units": "%",         "full_name": "Red Cell Distribution Width (CV)",        "pop_ri": {"F": (10, 14.5),    "M": (10, 14.5)}},

    # ── BMP ───────────────────────────────────────────────────────────────────
    "NA":    {"battery": "BMP",          "units": "mEq/L",     "full_name": "Sodium",                                  "pop_ri": {"F": (135, 145),    "M": (135, 145)}},
    "K":     {"battery": "BMP",          "units": "mEq/L",     "full_name": "Potassium",                               "pop_ri": {"F": (3.6, 5.2),    "M": (3.6, 5.2)}},
    "CL":    {"battery": "BMP",          "units": "mEq/L",     "full_name": "Chloride",                                "pop_ri": {"F": (98, 108),      "M": (98, 108)}},
    "CO2":   {"battery": "BMP",          "units": "mEq/L",     "full_name": "Carbon Dioxide",                           "pop_ri": {"F": (22, 32),       "M": (22, 32)}},
    "IGAP":  {"battery": "BMP",          "units": "mEq/L",     "full_name": "Anion Gap",                               "pop_ri": {"F": (4, 12),        "M": (4, 12)}},
    "GLU":   {"battery": "BMP",          "units": "mg/dL",     "full_name": "Glucose",                                 "pop_ri": {"F": (62, 125),      "M": (62, 125)}, "log_transform": True},
    "BUN":   {"battery": "BMP",          "units": "mg/dL",     "full_name": "Blood Urea Nitrogen",                     "pop_ri": {"F": (8, 21),        "M": (8, 21)}},
    "CRE":   {"battery": "BMP",          "units": "mg/dL",     "full_name": "Creatinine",                              "pop_ri": {"F": (0.38, 1.02),  "M": (0.51, 1.18)}},
    "CA":    {"battery": "BMP",          "units": "mg/dL",     "full_name": "Calcium",                                 "pop_ri": {"F": (8.9, 10.2),   "M": (8.9, 10.2)}},

    # ── WBC differential ──────────────────────────────────────────────────────
    "TNEUT": {"battery": "WBC diff",     "units": "x10^3/uL",  "full_name": "Total Neutrophils",                       "pop_ri": {"F": (1.80, 7),     "M": (1.80, 7)}},
    "LYMPH": {"battery": "WBC diff",     "units": "x10^3/uL",  "full_name": "Lymphocytes",                             "pop_ri": {"F": (1.00, 4.80),  "M": (1.00, 4.80)}},
    "MONOC": {"battery": "WBC diff",     "units": "x10^3/uL",  "full_name": "Monocytes",                               "pop_ri": {"F": (0.00, 0.80),  "M": (0.00, 0.80)}},

    # ── Hepatic panel ─────────────────────────────────────────────────────────
    "ALB":    {"battery": "Hepatic",     "units": "g/dL",      "full_name": "Albumin",                                 "pop_ri": {"F": (3.5, 5.2),    "M": (3.5, 5.2)}},
    "AST":    {"battery": "Hepatic",     "units": "U/L",       "full_name": "Aspartate Aminotransferase",              "pop_ri": {"F": (9, 33),        "M": (9, 33)}, "log_transform": True},
    "ALT":    {"battery": "Hepatic",     "units": "U/L",       "full_name": "Alanine Aminotransferase",                "pop_ri": {"F": (7, 33),        "M": (10, 64)}, "log_transform": True},

    "ALK": {
        "battery": "Hepatic",
        "units": "U/L",
        "full_name": "Alkaline Phosphatase",
        "log_transform": True,
        "pop_ri_age": {
            "F": [
                ((18, 25),            (26, 98)),
                ((25, 35),            (25, 100)),
                ((35, 45),            (25, 112)),
                ((45, 55),            (34, 121)),
                ((55, 65),            (31, 132)),
                ((65, 75),            (38, 172)),
                ((75, 200),           (49, 199)),
                ((200, float("inf")), (36, 122)),
            ],
            "M": [
                ((18, 25),            (42, 136)),
                ((25, 35),            (35, 109)),
                ((35, 45),            (36, 122)),
                ((45, 55),            (39, 139)),
                ((55, 65),            (37, 159)),
                ((65, 75),            (36, 161)),
                ((75, 200),           (52, 227)),
                ((200, float("inf")), (36, 122)),
            ],
        },
    },
    "BIL":    {"battery": "Hepatic",     "units": "mg/dL",     "full_name": "Total Bilirubin",                         "pop_ri": {"F": (0.2, 1.3),    "M": (0.2, 1.3)}, "log_transform": True},
    "BILD":   {"battery": "Hepatic",     "units": "mg/dL",     "full_name": "Direct Bilirubin",                        "pop_ri": {"F": (0.0, 0.3),    "M": (0.0, 0.3)}, "log_transform": True},

   "TP":    {"battery": "Hepatic",     "units": "g/dL",       "full_name": "Total Protein",                           "pop_ri": {"F": (6.0, 8.2),    "M": (6.0, 8.2)}},
    # ── Lipid panel ───────────────────────────────────────────────────────────
    "CHOL":   {"battery": "Lipid",       "units": "mg/dL",     "full_name": "Total Cholesterol",                       "pop_ri": {"F": (0, 210),       "M": (0, 210)}},
    "TRIG":   {"battery": "Lipid",       "units": "mg/dL",     "full_name": "Triglycerides",                           "pop_ri": {"F": (0, 175),       "M": (0, 175)}, "log_transform": True},
    "HDL":    {"battery": "Lipid",       "units": "mg/dL",     "full_name": "HDL Cholesterol",                         "pop_ri": {"F": (49, float("inf")), "M": (39, float("inf"))}},
    "NONHDL": {"battery": "Lipid",       "units": "mg/dL",     "full_name": "Non-HDL Cholesterol",                     "pop_ri": {"F": (0, 190),       "M": (0, 190)}},
    # "LDLN":   {"battery": "Lipid",       "units": "mg/dL",     "pop_ri": {"F": (0, 160),       "M": (0, 160)}},
    "LDL":    {"battery": "Lipid",       "units": "mg/dL",     "full_name": "LDL Cholesterol",                         "pop_ri": {"F": (0, 130),       "M": (0, 130)}},

    # ── Coagulation ───────────────────────────────────────────────────────────
    "PROPAT": {"battery": "Coag",        "units": "sec",       "full_name": "Prothrombin Time",                        "pop_ri": {"F": (10.7, 15.6),  "M": (10.7, 15.6)}, "log_transform": True},
    "PROINR": {"battery": "Coag",        "units": "ratio",     "full_name": "INR",                                     "pop_ri": {"F": (0.8, 1.3),    "M": (0.8, 1.3)}, "log_transform": True},

    # ── Misc outpatient markers (incl. inflammatory) ──────────────────────────
    "HSCRP":  {"battery": "Misc",        "units": "mg/L",      "full_name": "High-Sensitivity C-Reactive Protein",     "pop_ri": {"F": (0.0, 10.0),   "M": (0.0, 10.0)}, "log_transform": True},
    "LD":     {"battery": "Misc",        "units": "U/L",       "full_name": "Lactate Dehydrogenase",                   "pop_ri": {"F": (0, 210),       "M": (0, 210)},    "log_transform": True},
    # "LDRD":   {"battery": "Misc",        "units": "U/L",       "pop_ri": {"F": (0, 225),       "M": (0, 225)}},      # Roche LD direct
    "MG":     {"battery": "Misc",        "units": "mg/dL",     "full_name": "Magnesium",                               "pop_ri": {"F": (1.8, 2.4),    "M": (1.8, 2.4)}},
    "P":      {"battery": "Misc",        "units": "mg/dL",     "full_name": "Phosphate",                               "pop_ri": {"F": (2.5, 4.5),    "M": (2.5, 4.5)}},
    "TSH":    {"battery": "Misc",        "units": "mIU/L",     "full_name": "Thyroid-Stimulating Hormone",             "pop_ri": {"F": (0.4, 5.0),    "M": (0.4, 5.0)}, "log_transform": True},
    "FER":    {"battery": "Misc",        "units": "ng/mL",     "full_name": "Ferritin",                                "pop_ri": {"F": (10, 180),      "M": (20, 230)}, "log_transform": True},
    "VITDT":  {"battery": "Misc",        "units": "ng/mL",     "full_name": "Vitamin D (25-OH)",                       "pop_ri": {"F": (20.1, 50.0),  "M": (20.1, 50.0)}},
    "A1C":    {"battery": "Misc",        "units": "%",         "full_name": "Hemoglobin A1c",                          "pop_ri": {"F": (4.0, 5.6),    "M": (4.0, 5.6)}, "log_transform": True},
}

# Index-of-individuality scores are fixed marker metadata. 
_MARKER_IOI = {
    "LYMPH": 0.1240856150740024, "CRE": 0.215252942155578, "FER": 0.2602852985631083,
    "BILD": 0.2664928162128014, "PROINR": 0.2965466807815566, "PROPAT": 0.3037294087730144,
    "TSH": 0.3326601130580597, "WBC": 0.3812644572617786, "MCH": 0.3876969640554623,
    "HDL": 0.3944065446788048, "ALK": 0.4013030626568585, "MCV": 0.401919169034569,
    "LDL": 0.4335901683325938, "A1C": 0.4337080400142871, "PLT": 0.4386564571095497,
    "AST": 0.4776962231762662, "RDWCV": 0.4910737411663658, "RBC": 0.4934490295373994,
    "GLU": 0.4963864628480854, "BIL": 0.5112988577381046, "ALT": 0.526406963387422,
    "TRIG": 0.5275096724791868, "HB": 0.5493032845512203, "BUN": 0.5591772391010709,
    "HCT": 0.589274833602067, "HSCRP": 0.6205212879792517, "LD": 0.6362586935132956,
    "MONOC": 0.6447740892727473, "CHOL": 0.6751771542283006, "MCHC": 0.6798611254803866,
    "TP": 0.6827770490892363, "ALB": 0.6955637720803165, "TNEUT": 0.6969498808828416,
    "NONHDL": 0.7151275142806887, "VITDT": 0.7764068403858958, "CL": 0.8372227004786118,
    "MG": 0.8769385887577749, "CA": 0.891704707588067, "NA": 0.8982162208920138,
    "CO2": 0.9421859623817695, "P": 0.9853582851971668, "K": 1.0048144894615505,
    "IGAP": 1.2415616726559728,
}

for _test_code, _ioi in _MARKER_IOI.items():
    MARKER_CONFIG[_test_code]["ioi"] = _ioi
