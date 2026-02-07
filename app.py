import streamlit as st
import json
import os
import re
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

from ocr.ocr_engine import extract_text
from ml.predictor import predict_all_risks  # uses 3 pkls (diabetes/hypertension/cvd)
from ai.prompts import EXTRACTION_PROMPT, REASONING_PROMPT
from gemini_client import get_text_model

@st.cache_data
def cached_eval(path):
    return evaluate_models_on_dataset(path)

@st.cache_data
def cached_gemini_call(prompt):
    model = get_text_model()
    return model.generate_content(prompt).text

st.set_page_config(
    page_title="WHO-Aligned AI Health Digital Twin",
    layout="wide"
)

def load_who():
    for p in ["who/who_standard.json", "who_standard.json"]:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError("who_standard.json not found in ./who/ or project root.")

WHO = load_who()

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "chat" not in st.session_state:
    st.session_state.chat = []
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "risks" not in st.session_state:
    st.session_state.risks = {}
if "classifications" not in st.session_state:
    st.session_state.classifications = {}

def classify_fasting_glucose(val):
    fg = WHO["diabetes_indicators"]["fasting_glucose"]
    if val is None:
        return "Unknown"
    if fg["normal"][0] <= val <= fg["normal"][1]:
        return "Normal"
    if fg["diabetes_indicator"][0] <= val <= fg["diabetes_indicator"][1]:
        return "Diabetes Indicator"
    return "Unknown"

def classify_hba1c(val):
    h = WHO["diabetes_indicators"]["hba1c"]
    if val is None:
        return "Unknown"
    if h["normal"][0] <= val <= h["normal"][1]:
        return "Normal"
    if h["diabetes_indicator"][0] <= val <= h["diabetes_indicator"][1]:
        return "Diabetes Indicator"
    return "Unknown"

def classify_bp(sys_bp, dia_bp):
    if sys_bp is None or dia_bp is None:
        return "Unknown"

    cls = WHO["hypertension_engine"]["classification"]

    def in_band(b):
        return (b["systolic"][0] <= sys_bp <= b["systolic"][1]) and (b["diastolic"][0] <= dia_bp <= b["diastolic"][1])

    if in_band(cls["normal"]):
        return "Normal"
    if in_band(cls["elevated"]):
        return "Elevated"
    if in_band(cls["hypertension_stage_1"]):
        return "Hypertension Stage 1"
    if in_band(cls["hypertension_stage_2"]):
        return "Hypertension Stage 2"
    if in_band(cls["hypertensive_crisis"]):
        return "Hypertensive Crisis"
    return "Unknown"

def classify_lipids(ldl, hdl, tg):
    out = {}
    lip = WHO.get("lipid_profile_standards", {})

    def in_spec(val, spec):
        """
        spec can be:
        - [min, max]
        - {"min": x, "max": y}
        - {"min": x}
        - {"max": y}
        """
        if val is None:
            return False

        # list/tuple: [min, max]
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            lo, hi = spec
            return lo <= val <= hi

        # dict: min/max keys
        if isinstance(spec, dict):
            lo = spec.get("min", None)
            hi = spec.get("max", None)

            if lo is not None and hi is not None:
                return lo <= val <= hi
            if lo is not None:
                return val >= lo
            if hi is not None:
                return val <= hi

        return False

    def bucket(field_name, val):
        if val is None:
            return "Unknown"

        field = lip.get(field_name, {})
        # field is like {"optimal": [0,99], "borderline_high": {...}, ...}
        for label, spec in field.items():
            if in_spec(val, spec):
                return label.replace("_", " ").title()

        return "Unknown"

    # Only classify keys that exist in JSON (safe)
    if "ldl_cholesterol" in lip:
        out["LDL"] = bucket("ldl_cholesterol", ldl)
    if "hdl_cholesterol" in lip:
        out["HDL"] = bucket("hdl_cholesterol", hdl)
    if "triglycerides" in lip:
        out["Triglycerides"] = bucket("triglycerides", tg)

    return out

TARGET_FIELDS = [
    "age", "gender", "bmi", "waist_circumference",
    "sys_bp_avg", "dia_bp_avg",
    "fasting_glucose", "hba1c",
    "total_cholesterol", "ldl", "hdl", "triglycerides"
]

def safe_json_load(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except:
        return {}

def extract_from_report_with_gemini(raw_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not configured in .env")
        return {}

    if not raw_text or not raw_text.strip():
        st.error("No text extracted from the report. Try a clearer PDF/image.")
        return {}

    try:
        from gemini_client import get_text_model
        model = get_text_model()

        prompt = EXTRACTION_PROMPT.format(
            target_features=json.dumps(TARGET_FIELDS, indent=2),
            text=raw_text[:6000]
        )

        text = cached_gemini_call(prompt)
        return safe_json_load(text)


    except Exception as e:
        st.error(f"Gemini extraction failed: {e}")
        return {}

def health_assistant_reply(question, values, who_labels, risks):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "AI assistant unavailable (API key not configured)."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        context = {
            "user_data": values,
            "who_classification": who_labels,
            "ml_risks": risks
        }

        prompt = REASONING_PROMPT.format(
            user_data=json.dumps(values, indent=2),
            who_labels=json.dumps(who_labels, indent=2),
            risk_score=round(float(risks.get("Diabetes_Risk_%", 0)), 2)
        )

        final_prompt = f"{prompt}\n\nExtra Context:\n{json.dumps(context, indent=2)}\n\nUser Question:\n{question}"
        return cached_gemini_call(final_prompt)

    except:
        return "Unable to generate response at the moment."
    
def generate_pdf(user_data, risks, classifications):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "WHO-Aligned Preventive Health Summary (ML Risk)", ln=True)

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "User Health Metrics", ln=True)
    pdf.set_font("Arial", "", 10)
    for k, v in user_data.items():
        pdf.cell(0, 7, f"{k.replace('_',' ').title()}: {v}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "WHO Classification", ln=True)
    pdf.set_font("Arial", "", 10)
    for k, v in classifications.items():
        if isinstance(v, dict):
            pdf.cell(0, 7, f"{k}:", ln=True)
            for kk, vv in v.items():
                pdf.cell(0, 7, f"  - {kk}: {vv}", ln=True)
        else:
            pdf.cell(0, 7, f"{k}: {v}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "ML Risk Predictions (3 Models)", ln=True)
    pdf.set_font("Arial", "", 10)

    pdf.cell(0, 7, f"Diabetes Risk: {risks.get('Diabetes_Risk_%', 'NA')}%", ln=True)
    pdf.cell(0, 7, f"Hypertension Risk: {risks.get('Hypertension_Risk_%', 'NA')}%", ln=True)
    pdf.cell(0, 7, f"CVD Risk: {risks.get('CVD_Risk_%', 'NA')}%", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0, 6,
        "This report is generated for preventive risk-awareness only and does not constitute a medical diagnosis. "
        "Please consult a healthcare professional."
    )

    return pdf.output(dest="S").encode("latin-1")

st.title("WHO-Aligned AI Health Digital Twin")
st.warning("Preventive risk awareness only. This is not a medical diagnosis.")

tab1, tab2, tab3 = st.tabs(["Input & Analyze", "Results", "Download Report"])

with tab1:
    st.subheader("Option A — Upload Report (PDF/Image)")
    up = st.file_uploader("Upload report", type=["pdf", "png", "jpg", "jpeg"])
    if up is not None:
        if st.button("Extract values from report"):
            raw = extract_text(up)
            extracted = extract_from_report_with_gemini(raw)

            for k, v in extracted.items():
                if v is not None:
                    st.session_state.user_data[k] = v

            st.success("Extracted values added. Please review/edit below.")

    st.divider()
    st.subheader("Option B — Enter / Edit Values")

    age = st.number_input("Age", 18, 100, int(st.session_state.user_data.get("age", 30)))
    gender = st.selectbox("Gender", ["Female", "Male"],
                          index=1 if str(st.session_state.user_data.get("gender", "Male")).lower().startswith("m") else 0)

    bmi = st.number_input("BMI", 10.0, 50.0, float(st.session_state.user_data.get("bmi", 22.0)))
    waist = st.number_input("Waist Circumference (cm)", 0.0, 200.0, float(st.session_state.user_data.get("waist_circumference", 0.0)))

    sys_bp = st.number_input("Systolic BP (mmHg)", 80.0, 250.0, float(st.session_state.user_data.get("sys_bp_avg", 120.0)))
    dia_bp = st.number_input("Diastolic BP (mmHg)", 40.0, 150.0, float(st.session_state.user_data.get("dia_bp_avg", 80.0)))

    fg = st.number_input("Fasting Glucose (mg/dL)", 50.0, 400.0, float(st.session_state.user_data.get("fasting_glucose", 90.0)))
    hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, float(st.session_state.user_data.get("hba1c", 5.2)))

    tc = st.number_input("Total Cholesterol (mg/dL)", 50.0, 600.0, float(st.session_state.user_data.get("total_cholesterol", 170.0)))
    ldl = st.number_input("LDL (mg/dL)", 0.0, 400.0, float(st.session_state.user_data.get("ldl", 100.0)))
    hdl = st.number_input("HDL (mg/dL)", 0.0, 150.0, float(st.session_state.user_data.get("hdl", 45.0)))
    tg = st.number_input("Triglycerides (mg/dL)", 0.0, 2000.0, float(st.session_state.user_data.get("triglycerides", 120.0)))

    if st.button("Analyze (WHO + 3 ML Models)"):
        st.session_state.user_data = {
            "age": int(age),
            "gender": gender,
            "bmi": float(bmi),
            "waist_circumference": float(waist),
            "sys_bp_avg": float(sys_bp),
            "dia_bp_avg": float(dia_bp),
            "fasting_glucose": float(fg),
            "hba1c": float(hba1c),
            "total_cholesterol": float(tc),
            "ldl": float(ldl),
            "hdl": float(hdl),
            "triglycerides": float(tg),
        }

        st.session_state.classifications = {
            "Fasting Glucose": classify_fasting_glucose(float(fg)),
            "HbA1c": classify_hba1c(float(hba1c)),
            "Blood Pressure": classify_bp(float(sys_bp), float(dia_bp)),
            "Lipids": classify_lipids(float(ldl), float(hdl), float(tg))
        }

        st.session_state.risks = predict_all_risks(st.session_state.user_data)

        st.session_state.analyzed = True
        st.success("Analysis complete. Go to Results tab.")


with tab2:
    if not st.session_state.analyzed:
        st.info("Run analysis first in the Input & Analyze tab.")
    else:
        st.subheader("ML Risk Predictions (3 Models)")
        st.json(st.session_state.risks)

        st.subheader("Risk Chart")
        import matplotlib.pyplot as plt

        labels = ["Diabetes", "Hypertension", "CVD"]
        values = [
            st.session_state.risks.get("Diabetes_Risk_%", 0),
            st.session_state.risks.get("Hypertension_Risk_%", 0),
            st.session_state.risks.get("CVD_Risk_%", 0),
        ]

        fig = plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.ylim(0, 100)
        plt.ylabel("Risk %")
        plt.title("Predicted Risk (%)")
        st.pyplot(fig)
        st.subheader("WHO Classification")
        st.json(st.session_state.classifications)

        st.divider()
        st.subheader("💬 Health Assistant")

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if q := st.chat_input("Ask about your health results"):
            st.session_state.chat.append({"role": "user", "content": q})
            reply = health_assistant_reply(
                q,
                st.session_state.user_data,
                st.session_state.classifications,
                st.session_state.risks
            )
            st.session_state.chat.append({"role": "assistant", "content": reply})
            st.rerun()

import os
from ml.predictor import (
    evaluate_models_on_dataset,
    plot_risk_distribution_from_dataset,
    plot_risk_correlation_heatmap,
    plot_confusion_matrix_heatmap
)

DATASET_PATH = "final_combined_dataset.csv"  

st.divider()
st.subheader("Model Insights (Dataset-level graphs)")

if os.path.exists(DATASET_PATH):
    with st.expander("Show model insight graphs (from dataset)", expanded=False):
        try:
            d_prob, h_prob, c_prob, y_d, y_h, y_c = evaluate_models_on_dataset(DATASET_PATH)

            # 1) Risk distributions
            st.markdown("### Risk Distribution (Low / Moderate / High)")
            st.pyplot(plot_risk_distribution_from_dataset(None, d_prob, "Diabetes Risk Distribution"))
            st.pyplot(plot_risk_distribution_from_dataset(None, h_prob, "Hypertension Risk Distribution"))
            st.pyplot(plot_risk_distribution_from_dataset(None, c_prob, "CVD Risk Distribution"))

            # 2) Correlation heatmap
            st.markdown("### Correlation Between Predicted Risks")
            st.pyplot(plot_risk_correlation_heatmap(d_prob, h_prob, c_prob))

            # 3) Confusion matrices (only if labels exist)
            st.markdown("### Model Reliability (Confusion Matrix)")
            if y_d is not None:
                st.pyplot(plot_confusion_matrix_heatmap(y_d, d_prob, title="Diabetes Model Reliability"))
            else:
                st.info("diabetes_label not found in dataset. Skipping diabetes confusion matrix.")

            if y_h is not None:
                st.pyplot(plot_confusion_matrix_heatmap(y_h, h_prob, title="Hypertension Model Reliability"))
            else:
                st.info("hypertension_label not found in dataset. Skipping hypertension confusion matrix.")

            if y_c is not None:
                st.pyplot(plot_confusion_matrix_heatmap(y_c, c_prob, title="CVD Model Reliability"))
            else:
                st.info("cvd_label not found in dataset. Skipping CVD confusion matrix.")

        except Exception as e:
            st.error(f"Could not generate model insight graphs: {e}")
else:
    st.info("Place final_combined_dataset.csv in your project root to enable dataset-level model graphs.")

with tab3:
    if st.session_state.analyzed:
        pdf_bytes = generate_pdf(
            st.session_state.user_data,
            st.session_state.risks,
            st.session_state.classifications
        )

        st.download_button(
            "Download PDF Health Summary",
            data=pdf_bytes,
            file_name="WHO_Health_Twin_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Run analysis first.")
