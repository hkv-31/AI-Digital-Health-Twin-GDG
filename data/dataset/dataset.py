import pandas as pd
from functools import reduce
from sklearn.impute import SimpleImputer

def load_xpt(file, cols):
    df = pd.read_sas(file, format="xport")
    return df[cols]

demo = load_xpt(
    "P_DEMO.xpt",
    ["SEQN", "RIDAGEYR", "RIAGENDR"]
)

bmx = load_xpt(
    "P_BMX.xpt",
    ["SEQN", "BMXBMI", "BMXWAIST"]
)

bp_raw = pd.read_sas("P_BPXO.xpt", format="xport")
bp = bp_raw[
    [
        "SEQN",
        "BPXOSY1", "BPXOSY2", "BPXOSY3",
        "BPXODI1", "BPXODI2", "BPXODI3",
    ]
]

bp["avg_systolic"] = bp[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
bp["avg_diastolic"] = bp[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)

bp = bp[["SEQN", "sys_bp_avg", "dia_bp_avg"]]

glucose = load_xpt(
    "P_GLU.xpt",
    ["SEQN", "LBXGLU"]
)

hba1c = load_xpt(
    "P_GHB.xpt",
    ["SEQN", "LBXGH"]
)

chol = load_xpt(
    "P_TCHOL.xpt",
    ["SEQN", "LBXTC"]
)

hdl = load_xpt(
    "P_HDL.xpt",
    ["SEQN", "LBDHDD"]
)

trig = load_xpt(
    "P_TRIGLY.xpt",
    ["SEQN", "LBXTR"]
)

diabetes = load_xpt(
    "P_DIQ.xpt",
    ["SEQN", "DIQ010"]  # 1 = Yes, 2 = No
)

dfs = [
    demo,
    bmx,
    bp,
    glucose,
    hba1c,
    chol,
    hdl,
    trig,
    diabetes
]

final_df = reduce(
    lambda left, right: pd.merge(left, right, on="SEQN", how="left"),
    dfs
)

# Binary diabetes label
final_df["diabetes_label"] = final_df["DIQ010"].map({1: 1, 2: 0})

# Drop original questionnaire column
final_df.drop(columns=["DIQ010"], inplace=True)

final_df.rename(columns={
    "RIDAGEYR": "age",
    "RIAGENDR": "gender",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_circumference",
    "LBXGLU": "fasting_glucose",
    "LBXGH": "hba1c",
    "LBXTC": "total_cholesterol",
    "LBDHDD": "hdl",
    "LBXTR": "triglycerides"
}, inplace=True)

features = [
    "age",
    "gender",
    "bmi",
    "waist_circumference",
    "sys_bp_avg",
    "dia_bp_avg",
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "hdl",
    "triglycerides"
]

X = final_df[features]
y = final_df["diabetes_label"]

# Median imputation (best for medical data)
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Recombine
final_dataset = pd.concat([X_imputed, y.reset_index(drop=True)], axis=1)

final_dataset.to_csv("nhanes_final_rf_dataset.csv", index=False)

print("Dataset created: nhanes_final_dataset.csv")
print("Rows:", final_dataset.shape[0])
print("Columns:", final_dataset.shape[1])
print(final_dataset.head())