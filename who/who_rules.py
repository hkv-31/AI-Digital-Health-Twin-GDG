# who_rules.py
import json
import os

def _load_who():
    candidates = [
        os.path.join("who", "who_standard.json"),
        "who_standard.json"
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    raise FileNotFoundError("who_standard.json not found in ./who/ or project root.")

WHO = _load_who()

def in_range(val, lo, hi):
    return val is not None and lo <= val <= hi

def classify_fasting_glucose(val):
    fg = WHO["diabetes_indicators"]["fasting_glucose"]
    if val is None:
        return "Unknown"
    if in_range(val, fg["normal"][0], fg["normal"][1]):
        return "Normal"
    if in_range(val, fg["diabetes_indicator"][0], fg["diabetes_indicator"][1]):
        return "Diabetes Indicator"
    return "Unknown"

def classify_hba1c(val):
    h = WHO["diabetes_indicators"]["hba1c"]
    if val is None:
        return "Unknown"
    if in_range(val, h["normal"][0], h["normal"][1]):
        return "Normal"
    if in_range(val, h["diabetes_indicator"][0], h["diabetes_indicator"][1]):
        return "Diabetes Indicator"
    return "Unknown"

def classify_systolic_bp(val):
    if val is None:
        return "Unknown"
    ranges = WHO["cardiovascular_risk_factors"]["systolic_blood_pressure"]["ranges"]
    for r in ranges:
        if "min" in r and "max" in r and r["min"] <= val <= r["max"]:
            return r["label"]
        if "max" in r and val < r["max"]:
            return r["label"]
    return "Unknown"

def classify_lipids(ldl=None, hdl=None, triglycerides=None):
    out = {}
    lip = WHO.get("lipid_profile_standards", {})

    def _bucket(name, val):
        if val is None:
            return "Unknown"
        spec = lip.get(name, {})
        for label, rr in spec.items():
            if isinstance(rr, list) and rr[0] <= val <= rr[1]:
                return label.replace("_", " ").title()
        return "Unknown"

    if "ldl_cholesterol" in lip:
        out["LDL"] = _bucket("ldl_cholesterol", ldl)
    if "hdl_cholesterol" in lip:
        out["HDL"] = _bucket("hdl_cholesterol", hdl)
    if "triglycerides" in lip:
        out["Triglycerides"] = _bucket("triglycerides", triglycerides)

    return out

def classify_all(values: dict) -> dict:
    return {
        "Fasting Glucose": classify_fasting_glucose(values.get("fasting_glucose")),
        "HbA1c": classify_hba1c(values.get("hba1c")),
        "Systolic BP": classify_systolic_bp(values.get("systolic_bp")),
        **classify_lipids(values.get("ldl"), values.get("hdl"), values.get("triglycerides"))
    }
