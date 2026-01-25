# predictor.py
import os
import json
import numpy as np
import joblib

WHO_PATHS = [os.path.join("who", "who_standard.json"), "who_standard.json"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "diabetes": os.path.join(BASE_DIR, "diabetes_model.pkl"),
    "hypertension": os.path.join(BASE_DIR, "hypertension_model.pkl"),
    "cvd": os.path.join(BASE_DIR, "cvd_model.pkl"),
}


FEATURES = [
    "age", "gender", "bmi", "waist_circumference",
    "sys_bp_avg", "dia_bp_avg",
    "fasting_glucose", "hba1c",
    "total_cholesterol", "triglycerides",
    "tg_hdl_ratio", "map", "age_bmi",
    "WHO_Glucose_Code", "WHO_BP_Code", "WHO_BMI_Code"
]


def load_who():
    for p in WHO_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError("who_standard.json not found in ./who/ or project root.")

WHO = load_who()


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def who_glucose_code(fg, a1c) -> int:
    fg = _safe_float(fg, None)
    a1c = _safe_float(a1c, None)
    if fg is None or a1c is None:
        return 2

    fg_std = WHO["diabetes_indicators"]["fasting_glucose"]
    a1c_std = WHO["diabetes_indicators"]["hba1c"]

    fg_normal = fg_std["normal"][0] <= fg <= fg_std["normal"][1]
    a1c_normal = a1c_std["normal"][0] <= a1c <= a1c_std["normal"][1]

    fg_indicator = fg_std["diabetes_indicator"][0] <= fg <= fg_std["diabetes_indicator"][1]
    a1c_indicator = a1c_std["diabetes_indicator"][0] <= a1c <= a1c_std["diabetes_indicator"][1]

    if fg_normal and a1c_normal:
        return 1
    if fg_indicator or a1c_indicator:
        return 3
    return 2


def who_bp_code(sys_bp, dia_bp) -> int:
    sys_bp = _safe_float(sys_bp, None)
    dia_bp = _safe_float(dia_bp, None)
    if sys_bp is None or dia_bp is None:
        return 2

    cls = WHO["hypertension_engine"]["classification"]

    def in_band(b):
        return (b["systolic"][0] <= sys_bp <= b["systolic"][1]) and (b["diastolic"][0] <= dia_bp <= b["diastolic"][1])

    if in_band(cls["normal"]) or in_band(cls["elevated"]):
        return 1
    if in_band(cls["hypertension_stage_1"]):
        return 2
    if in_band(cls["hypertension_stage_2"]) or in_band(cls["hypertensive_crisis"]):
        return 3
    return 2


def who_bmi_code(bmi) -> int:
    bmi = _safe_float(bmi, None)
    if bmi is None:
        return 2
    if bmi < 25:
        return 1
    if bmi < 30:
        return 2
    return 3


def build_feature_row(values: dict) -> np.ndarray:
    age = _safe_float(values.get("age"))
    gender = values.get("gender")
    if isinstance(gender, str):
        gender = 1.0 if gender.strip().lower().startswith("m") else 0.0
    gender = _safe_float(gender)

    bmi = _safe_float(values.get("bmi"))
    waist = _safe_float(values.get("waist_circumference"))

    sys_bp = _safe_float(values.get("sys_bp_avg", values.get("systolic_bp")))
    dia_bp = _safe_float(values.get("dia_bp_avg", values.get("diastolic_bp")))

    fg = _safe_float(values.get("fasting_glucose"))
    a1c = _safe_float(values.get("hba1c"))

    tc = _safe_float(values.get("total_cholesterol"))
    tg = _safe_float(values.get("triglycerides"))
    hdl = _safe_float(values.get("hdl"))

    tg_hdl_ratio = tg / (hdl + 1e-9)
    map_val = (sys_bp + 2 * dia_bp) / 3
    age_bmi = age * bmi

    row = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "waist_circumference": waist,
        "sys_bp_avg": sys_bp,
        "dia_bp_avg": dia_bp,
        "fasting_glucose": fg,
        "hba1c": a1c,
        "total_cholesterol": tc,
        "triglycerides": tg,
        "tg_hdl_ratio": tg_hdl_ratio,
        "map": map_val,
        "age_bmi": age_bmi,
        "WHO_Glucose_Code": who_glucose_code(fg, a1c),
        "WHO_BP_Code": who_bp_code(sys_bp, dia_bp),
        "WHO_BMI_Code": who_bmi_code(bmi),
    }

    return np.array([[row[f] for f in FEATURES]], dtype=float)


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def predict_all_risks(values: dict) -> dict:
    x = build_feature_row(values)

    diab = _load_model(MODEL_PATHS["diabetes"])
    hyp = _load_model(MODEL_PATHS["hypertension"])
    cvd = _load_model(MODEL_PATHS["cvd"])

    d = float(diab.predict_proba(x)[0][1]) * 100.0
    h = float(hyp.predict_proba(x)[0][1]) * 100.0
    c = float(cvd.predict_proba(x)[0][1]) * 100.0

    return {
        "Diabetes_Risk_%": round(d, 2),
        "Hypertension_Risk_%": round(h, 2),
        "CVD_Risk_%": round(c, 2),
    }

# ---- ADD BELOW your existing code in predictor.py ----
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def risk_level_from_pct(p):
    # same bins you used earlier
    if p < 20: return "Low Risk"
    if p < 40: return "Moderate Risk"
    return "High Risk"


def plot_risk_distribution_from_dataset(df: pd.DataFrame, probs: np.ndarray, title: str):
    """
    probs: predicted probability for class 1 (0..1) for each row in df
    Creates the same Low/Moderate/High distribution plot you had earlier.
    """
    levels = pd.Series((probs * 100.0)).apply(risk_level_from_pct)
    categories = ["Low Risk", "Moderate Risk", "High Risk"]
    counts = levels.value_counts().reindex(categories, fill_value=0)
    perc = (counts / counts.sum()) * 100.0

    fig = plt.figure(figsize=(7, 5))
    bars = plt.bar(categories, counts.values)

    for bar, p in zip(bars, perc.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.03,
            f"{int(bar.get_height())}\n({p:.1f}%)",
            ha="center"
        )

    plt.title(title)
    plt.ylabel("Number of Patients")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    return fig


def plot_risk_correlation_heatmap(diabetes_probs, hypertension_probs, cvd_probs):
    """
    Replicates your old risk correlation heatmap style.
    """
    corr_df = pd.DataFrame({
        "Diabetes": diabetes_probs * 100.0,
        "Hypertension": hypertension_probs * 100.0,
        "CVD": cvd_probs * 100.0
    })
    corr = corr_df.corr()

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(3), corr.columns)
    plt.yticks(range(3), corr.index)

    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    plt.title("Correlation Between Predicted Risks")
    return fig


def plot_confusion_matrix_heatmap(y_true, y_prob, threshold=0.5, title="Confusion Matrix"):
    """
    Replicates your old confusion matrix heatmap style.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(cm)
    plt.colorbar(label="Patients")
    plt.xticks([0, 1], ["Predicted No", "Predicted Yes"])
    plt.yticks([0, 1], ["Actual No", "Actual Yes"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title(title)
    return fig


def evaluate_models_on_dataset(csv_path: str):
    """
    Loads dataset, builds X using the same build_feature_row logic row-wise,
    runs all 3 models, and returns probabilities + labels (if available).
    """
    df = pd.read_csv(csv_path)

    # required features to run predictions
    required = ["age","gender","bmi","waist_circumference","sys_bp_avg","dia_bp_avg",
                "fasting_glucose","hba1c","total_cholesterol","hdl","triglycerides"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns needed for evaluation: {missing}")

    # build X matrix
    X_rows = []
    for _, row in df.iterrows():
        vals = row.to_dict()
        X_rows.append(build_feature_row(vals)[0])
    X = np.array(X_rows, dtype=float)

    # load models
    diab = _load_model(MODEL_PATHS["diabetes"])
    hyp = _load_model(MODEL_PATHS["hypertension"])
    cvd = _load_model(MODEL_PATHS["cvd"])

    d_prob = diab.predict_proba(X)[:, 1]
    h_prob = hyp.predict_proba(X)[:, 1]
    c_prob = cvd.predict_proba(X)[:, 1]

    # labels if present
    y_diab = df["diabetes_label"].astype(int).values if "diabetes_label" in df.columns else None
    y_hyp = df["hypertension_label"].astype(int).values if "hypertension_label" in df.columns else None
    y_cvd = df["cvd_label"].astype(int).values if "cvd_label" in df.columns else None

    return (d_prob, h_prob, c_prob, y_diab, y_hyp, y_cvd)
