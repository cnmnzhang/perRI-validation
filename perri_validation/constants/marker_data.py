# ── Master marker configuration ───────────────────────────────────────────────
# Single source of truth for battery grouping, units, and sex-specific
# population reference intervals. Edit this file to adapt to a different
# population -- see README.md's "Adapting to a different population".
#
# Everything derived from this (BATTERY2TESTCODE, POP_REF_INTERVAL, etc.) lives
# in marker_config.py, which imports MARKER_CONFIG from here -- nothing in this
# file needs to know about that, and nothing here needs editing to add a
# derived constant.
#
# Fields
#   battery  – panel/battery name
#   units    – display units string
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
    # "TPRD":   {"battery": "Hepatic",     "units": "g/dL",      "pop_ri": {"F": (6.4, 8.3),    "M": (6.4, 8.3)}},
 # "ASTRD":  {"battery": "Hepatic",     "units": "U/L",       "pop_ri": {"F": (0, 35),        "M": (0, 35)}},       # provisional
        # "ALTRD":  {"battery": "Hepatic",     "units": "U/L",       "pop_ri": {"F": (0, 43),        "M": (0, 63)}},     # provisional
    # "BILDRD": {"battery": "Hepatic",     "units": "mg/dL",     "pop_ri": {"F": (0.0, 0.4),    "M": (0.0, 0.4)}},    # Roche (adult 1m+) Direct bilirubin
    # "BILRD":  {"battery": "Hepatic",     "units": "mg/dL",     "pop_ri": {"F": (0.0, 1.1),    "M": (0.0, 1.1)}},    # Roche (adult 1m+) Total bilirubin
    # provisional
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
