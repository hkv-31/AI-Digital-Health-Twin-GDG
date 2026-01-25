# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("final_combined_dataset.csv")
df = df.dropna(subset=["diabetes_label"])

# =====================================================
# BASIC PREPROCESSING
# =====================================================
df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(0)

# =====================================================
# WHO TARGET LABELS (FOR TRAINING)
# =====================================================
def who_hypertension_label(sys, dia):
    return 1 if sys >= 130 or dia >= 80 else 0

def who_cvd_label(sys, dia, hdl, tg):
    return 1 if (sys >= 130 or dia >= 80 or hdl < 40 or tg >= 200) else 0

df["hypertension_label"] = df.apply(
    lambda r: who_hypertension_label(r["sys_bp_avg"], r["dia_bp_avg"]), axis=1
)
df["cvd_label"] = df.apply(
    lambda r: who_cvd_label(r["sys_bp_avg"], r["dia_bp_avg"], r["hdl"], r["triglycerides"]), axis=1
)

# =====================================================
# WHO CODES (FEATURES)
# =====================================================
def who_glucose_code(fg, a1c):
    if fg < 100 and a1c < 5.7: return 1
    if fg >= 126 or a1c >= 6.5: return 3
    return 2

def who_bp_code(sys, dia):
    if sys < 120 and dia < 80: return 1
    if sys < 140: return 2
    return 3

def who_bmi_code(bmi):
    if bmi < 25: return 1
    if bmi < 30: return 2
    return 3

df["WHO_Glucose_Code"] = df.apply(
    lambda r: who_glucose_code(r["fasting_glucose"], r["hba1c"]), axis=1
)
df["WHO_BP_Code"] = df.apply(
    lambda r: who_bp_code(r["sys_bp_avg"], r["dia_bp_avg"]), axis=1
)
df["WHO_BMI_Code"] = df["bmi"].apply(who_bmi_code)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl"] + 1e-9)
df["map"] = (df["sys_bp_avg"] + 2 * df["dia_bp_avg"]) / 3
df["age_bmi"] = df["age"] * df["bmi"]

FEATURES = [
    "age","gender","bmi","waist_circumference",
    "sys_bp_avg","dia_bp_avg","fasting_glucose","hba1c",
    "total_cholesterol","triglycerides",
    "tg_hdl_ratio","map","age_bmi",
    "WHO_Glucose_Code","WHO_BP_Code","WHO_BMI_Code"
]

X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# =====================================================
# TRAIN FUNCTION
# =====================================================
def train_model(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    model = RandomForestClassifier(
        n_estimators=600,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    joblib.dump(model, f"{name}.pkl")
    print(f"✅ Saved {name}.pkl")

    y_pred = (model.predict_proba(X_te)[:,1] >= 0.5).astype(int)
    print(f"{name.upper()} Accuracy:", round(accuracy_score(y_te, y_pred)*100,2), "%")
    print(f"{name.upper()} Recall:", round(recall_score(y_te, y_pred)*100,2), "%")

    return model, X_te

# =====================================================
# TRAIN 3 MODELS
# =====================================================
diabetes_model, X_test = train_model(X, df["diabetes_label"], "diabetes_model")
hypertension_model, _ = train_model(X, df["hypertension_label"], "hypertension_model")
cvd_model, _ = train_model(X, df["cvd_label"], "cvd_model")

# =====================================================
# RISK PERCENTAGES (PURE ML)
# =====================================================
df.loc[X_test.index, "Diabetes_Risk_%"] = diabetes_model.predict_proba(X_test)[:,1] * 100
df.loc[X_test.index, "Hypertension_Risk_%"] = hypertension_model.predict_proba(X_test)[:,1] * 100
df.loc[X_test.index, "CVD_Risk_%"] = cvd_model.predict_proba(X_test)[:,1] * 100

# =====================================================
# RISK LEVELS
# =====================================================
def risk_level(p):
    if p < 20: return "Low Risk"
    if p < 40: return "Moderate Risk"
    return "High Risk"

for d in ["Diabetes","Hypertension","CVD"]:
    df.loc[X_test.index, f"{d}_Risk_Level"] = df.loc[X_test.index, f"{d}_Risk_%"].apply(risk_level)

# =====================================================
# FINAL PERSON-WISE REPORT
# =====================================================
final_report = df.loc[X_test.index, [
    "Diabetes_Risk_%","Diabetes_Risk_Level",
    "Hypertension_Risk_%","Hypertension_Risk_Level",
    "CVD_Risk_%","CVD_Risk_Level"
]]

print("\n🩺 FINAL INDIVIDUAL HEALTH RISK REPORT\n")
print(final_report.head(10))

final_report.to_csv("final_health_risk_report.csv")
print("📁 Saved final_health_risk_report.csv")

# =====================================================
# VISUALIZATIONS (UNCHANGED)
# =====================================================
RISK_COLORS = {
    "Low Risk": "#2ECC71",
    "Moderate Risk": "#F1C40F",
    "High Risk": "#E74C3C"
}

def plot_risk_bar(df, idx, col, title):
    categories = ["Low Risk","Moderate Risk","High Risk"]
    counts = df.loc[idx,col].value_counts().reindex(categories, fill_value=0)
    perc = counts / counts.sum() * 100

    plt.figure(figsize=(7,5))
    bars = plt.bar(categories, counts.values,
                   color=[RISK_COLORS[c] for c in categories])

    for bar,p in zip(bars, perc):
        plt.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+counts.max()*0.03,
                 f"{int(bar.get_height())}\n({p:.1f}%)",
                 ha="center")

    plt.title(title)
    plt.ylabel("Number of Patients")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()

plot_risk_bar(df, X_test.index, "Diabetes_Risk_Level", "Diabetes Risk Distribution")
plot_risk_bar(df, X_test.index, "Hypertension_Risk_Level", "Hypertension Risk Distribution")
plot_risk_bar(df, X_test.index, "CVD_Risk_Level", "Cardiovascular Risk Distribution")

# =====================================================
# HEATMAP: DISEASE CORRELATION
# =====================================================
corr = df.loc[X_test.index,
    ["Diabetes_Risk_%","Hypertension_Risk_%","CVD_Risk_%"]].corr()

plt.figure(figsize=(6,5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(3), ["Diabetes","Hypertension","CVD"])
plt.yticks(range(3), ["Diabetes","Hypertension","CVD"])

for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}",
                 ha="center", va="center")

plt.title("Correlation Between Disease Risks")
plt.show()

# =====================================================
# HEATMAP: MODEL RELIABILITY (DIABETES)
# =====================================================
cm = confusion_matrix(
    df.loc[X_test.index, "diabetes_label"],
    (diabetes_model.predict_proba(X_test)[:,1] >= 0.5).astype(int)
)

plt.figure(figsize=(5,5))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Patients")
plt.xticks([0,1], ["Predicted No","Predicted Yes"])
plt.yticks([0,1], ["Actual No","Actual Yes"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")

plt.title("Model Reliability: Diabetes Prediction")
plt.show()