# ── Derived constants ─────────────────────────────────────────────────────────
# Everything here is built from MARKER_CONFIG (marker_lab_config.py) -- to adapt to a
# different population, edit marker_lab_config.py, not this file. Nothing here
# should need hand-editing; if a marker's battery/units/pop_ri look wrong,
# the fix belongs in marker_lab_config.py.

from perri_validation.constants.marker_lab_config import MARKER_CONFIG

BATTERY2TESTCODE: dict[str, list[str]] = {}
for _tc, _cfg in MARKER_CONFIG.items():
    if _cfg.get("battery"):
        BATTERY2TESTCODE.setdefault(_cfg["battery"], []).append(_tc)

MARKER_UNITS: dict[str, str] = {_tc: _cfg["units"] for _tc, _cfg in MARKER_CONFIG.items() if "units" in _cfg}

MARKER_FULL_NAMES: dict[str, str] = {_tc: _cfg["full_name"] for _tc, _cfg in MARKER_CONFIG.items() if "full_name" in _cfg}

MARKER_IOI: dict[str, float] = {_tc: _cfg["ioi"] for _tc, _cfg in MARKER_CONFIG.items() if "ioi" in _cfg}
MARKER_IOI_ORDER: list[str] = sorted(MARKER_IOI, key=MARKER_IOI.get)

TESTCODE_DISPLAY: dict[str, str] = {
    "VITDT": "VITD",
    "PROPAT": "PT",
    "PROINR": "INR",
    "RDWCV": "RDW",
    "MONOC": "MONO",
    "TNEUT": "NEUT",
    "HSCRP": "CRP",
}

TESTCODES_LIST: list[str] = [_tc for _tcs in BATTERY2TESTCODE.values() for _tc in _tcs]

TESTCODE2BATTERY: dict[str, str] = {_tc: _cfg["battery"] for _tc, _cfg in MARKER_CONFIG.items() if _cfg.get("battery")}

POP_REF_INTERVAL: dict[str, dict[str, tuple]] = {}
for _tc, _cfg in MARKER_CONFIG.items():
    for _sex, _ri in _cfg.get("pop_ri", {}).items():
        POP_REF_INTERVAL.setdefault(_sex, {})[_tc] = _ri

POP_REF_INTERVAL["ALL"] = {
    _tc: (min(_v[0] for _v in _cfg["pop_ri"].values()), max(_v[1] for _v in _cfg["pop_ri"].values()))
    for _tc, _cfg in MARKER_CONFIG.items()
    if _cfg.get("pop_ri")
}
POP_REF_INTERVAL["All"] = POP_REF_INTERVAL["ALL"]

AGE_STRATIFIED_RI: dict[str, dict[str, list]] = {
    _tc: _cfg["pop_ri_age"]
    for _tc, _cfg in MARKER_CONFIG.items()
    if "pop_ri_age" in _cfg
}

# Inject global (widest) bounds for age-stratified markers into POP_REF_INTERVAL so
# the optimization path gets a valid wide prior with no code changes downstream.
for _tc, _age_ri in AGE_STRATIFIED_RI.items():
    for _sex, _bands in _age_ri.items():
        _global_lo = min(_ri[0] for _, _ri in _bands)
        _global_hi = max(_ri[1] for _, _ri in _bands)
        POP_REF_INTERVAL.setdefault(_sex, {})[_tc] = (_global_lo, _global_hi)
    _all_lo = min(POP_REF_INTERVAL[_s][_tc][0] for _s in ("F", "M") if _tc in POP_REF_INTERVAL.get(_s, {}))
    _all_hi = max(POP_REF_INTERVAL[_s][_tc][1] for _s in ("F", "M") if _tc in POP_REF_INTERVAL.get(_s, {}))
    POP_REF_INTERVAL.setdefault("ALL", {})[_tc] = (_all_lo, _all_hi)
POP_REF_INTERVAL["All"] = POP_REF_INTERVAL["ALL"]  # keep alias in sync after injection

LOG_TRANSFORM_MARKERS: set[str] = {_tc for _tc, _cfg in MARKER_CONFIG.items() if _cfg.get("log_transform")}

BATTERY2LABEL: dict[str, str] = {
    "CBC": "CBC",
    "BMP": "BMP",
    "Coag": "COAG",
    "Hepatic": "LFT",
    "Lipid": "LIPID",
    "WBC diff": "WCD",
    "Misc": "MISC",
}

__all__ = sorted(name for name, value in globals().items() if name.isupper())
