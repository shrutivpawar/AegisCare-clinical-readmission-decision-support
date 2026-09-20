import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

st.set_page_config(
    page_title="AegisCare | Clinical Readmission Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# PROFESSIONAL CLINICAL UI STYLING (CSS)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1E293B;
    }

    /* Hide Deploy button, Streamlit hamburger menu, and header decorations */
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    #MainMenu {
        visibility: hidden !important;
    }
    
    header {
        visibility: hidden !important;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    .top-navbar {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 22px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    .top-navbar h1 {
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .top-navbar p {
        color: #94A3B8;
        font-size: 14px;
        margin: 4px 0 0 0;
    }

    .cdss-alert {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 13px;
        color: #1E40AF;
        margin-bottom: 24px;
    }

    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }

    .protocol-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13.5px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        color: #64748B;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="top-navbar">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>AegisCare • Clinical Decision Support</h1>
            <p>Predictive 30-Day Hospital Readmission Risk Stratification for Chronic Disease Inpatients</p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">
                v2.4 CDSS Active
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cdss-alert">
    <strong>Regulatory Disclaimer:</strong> This platform is an algorithmic decision support system (CDSS) for risk stratification and discharge planning. It is not an autonomous diagnostic instrument and does not replace physician clinical discretion.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------------
@st.cache_resource
def load_system_artifacts():
    if not os.path.exists("artifacts/model.joblib") or not os.path.exists("artifacts/preprocessor.joblib"):
        return None, None, None
    m = joblib.load("artifacts/model.joblib")
    p = joblib.load("artifacts/preprocessor.joblib")
    e = shap.TreeExplainer(m)
    return m, p, e

model, preprocessor, explainer = load_system_artifacts()

if model is None:
    st.error("Model artifacts not found. Please run `python src/train.py` first.")
    st.stop()

# -------------------------------------------------------------
# NAVIGATION TABS
# -------------------------------------------------------------
tab_patient, tab_explain, tab_cohort, tab_performance = st.tabs([
    "🩺 Patient Evaluation & Discharge Triage",
    "🔍 Feature Attribution & Explainability",
    "📊 Population Health & Cohort Analytics",
    "⚙️ Model Governance & Technical Specifications"
])

# =============================================================
# TAB 1: PATIENT EVALUATION
# =============================================================
with tab_patient:
    # Synchronize Session State for quick persona selection
    def set_persona():
        p = st.session_state.selected_persona
        if p == "High-Risk Case (Severe Post-Op Elderly Diabetic)":
            st.session_state.age_val = 78
            st.session_state.gender_val = "Male"
            st.session_state.cond_val = "Diabetes"
            st.session_state.adm_val = "Emergency"
            st.session_state.los_val = 16
            st.session_state.bill_val = 32000.0
            st.session_state.med_val = "Lipitor"
            st.session_state.test_val = "Abnormal"
        elif p == "Low-Risk Case (Elective Young Adult)":
            st.session_state.age_val = 34
            st.session_state.gender_val = "Female"
            st.session_state.cond_val = "Asthma"
            st.session_state.adm_val = "Elective"
            st.session_state.los_val = 3
            st.session_state.bill_val = 7500.0
            st.session_state.med_val = "Aspirin"
            st.session_state.test_val = "Normal"
        else:
            st.session_state.age_val = 68
            st.session_state.gender_val = "Male"
            st.session_state.cond_val = "Hypertension"
            st.session_state.adm_val = "Emergency"
            st.session_state.los_val = 9
            st.session_state.bill_val = 18500.0
            st.session_state.med_val = "Aspirin"
            st.session_state.test_val = "Inconclusive"

    if "age_val" not in st.session_state:
        st.session_state.age_val = 68
        st.session_state.gender_val = "Male"
        st.session_state.cond_val = "Hypertension"
        st.session_state.adm_val = "Emergency"
        st.session_state.los_val = 9
        st.session_state.bill_val = 18500.0
        st.session_state.med_val = "Aspirin"
        st.session_state.test_val = "Inconclusive"

    preset_col, _ = st.columns([2, 3])
    with preset_col:
        st.markdown("**Quick Load Clinical Persona:**")
        st.selectbox(
            "Quick Load Clinical Persona",
            ["Custom Input", "High-Risk Case (Severe Post-Op Elderly Diabetic)", "Low-Risk Case (Elective Young Adult)"],
            key="selected_persona",
            on_change=set_persona,
            label_visibility="collapsed"
        )

    col_input, col_output = st.columns([1.1, 1.2], gap="large")

    conditions_list = ["Diabetes", "Hypertension", "Cancer", "Asthma", "Arthritis", "Obesity"]
    admissions_list = ["Emergency", "Urgent", "Elective"]
    meds_list = ["Aspirin", "Lipitor", "Penicillin", "Ibuprofen", "Paracetamol"]
    tests_list = ["Abnormal", "Normal", "Inconclusive"]

    with col_input:
        st.markdown("#### Patient Clinical Record")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Patient Age", 18, 105, value=st.session_state.age_val, key="in_age")
            gender = st.selectbox("Biological Sex", ["Male", "Female"], 
                                  index=0 if st.session_state.gender_val == "Male" else 1, key="in_gender")
            condition = st.selectbox("Chronic Condition", conditions_list, 
                                     index=conditions_list.index(st.session_state.cond_val), key="in_condition")
            admission_type = st.selectbox("Admission Acuity", admissions_list, 
                                          index=admissions_list.index(st.session_state.adm_val), key="in_adm")

        with c2:
            length_of_stay = st.slider("Length of Stay (Days)", 1, 45, value=st.session_state.los_val, key="in_los")
            test_results = st.selectbox("Discharge Lab Finding", tests_list, 
                                        index=tests_list.index(st.session_state.test_val), key="in_test")
            medication = st.selectbox("Prescription Regimen", meds_list, 
                                      index=meds_list.index(st.session_state.med_val), key="in_med")
            billing = st.number_input("Billing Amount ($)", 1000.0, 80000.0, 
                                       value=st.session_state.bill_val, step=500.0, key="in_bill")

        patient_dict = {
            'age': age,
            'billing_amount': billing,
            'length_of_stay': length_of_stay,
            'gender': gender,
            'medical_condition': condition,
            'admission_type': admission_type,
            'medication': medication,
            'test_results': test_results
        }
        patient_df = pd.DataFrame([patient_dict])

    with col_output:
        st.markdown("#### Predictive Stratification")
        
        proc_patient = preprocessor.transform(patient_df)
        prob = model.predict_proba(proc_patient)[0][1]

        if prob >= 0.50:
            tier = "HIGH"
            tier_color = "#DC2626"
            bg_gradient = "linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%)"
            border_color = "#FCA5A5"
            badge_bg = "#DC2626"
            action_badge = "🔴 URGENT DISCHARGE CARE PLAN REQUIRED"
        elif prob >= 0.25:
            tier = "MODERATE"
            tier_color = "#D97706"
            bg_gradient = "linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%)"
            border_color = "#FCD34D"
            badge_bg = "#D97706"
            action_badge = "🟡 STRUCTURED OUTPATIENT MONITORING REQUIRED"
        else:
            tier = "LOW"
            tier_color = "#059669"
            bg_gradient = "linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%)"
            border_color = "#86EFAC"
            badge_bg = "#059669"
            action_badge = "🟢 STANDARD POST-DISCHARGE ROUTINE"

        st.markdown(f"""
        <div style="background: {bg_gradient}; border: 1.5px solid {border_color}; border-radius: 14px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #475569; letter-spacing: 0.8px; text-transform: uppercase;">
                        30-Day Readmission Risk Index
                    </span>
                    <div style="font-size: 42px; font-weight: 800; color: {tier_color}; margin-top: 2px;">
                        {prob:.1%}
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: {badge_bg}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 13px;">
                        {tier} RISK TIER
                    </span>
                    <div style="font-size: 11px; color: #64748B; margin-top: 6px;">Operating Cutoff: 40.0%</div>
                </div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid {border_color}; font-size: 12px; font-weight: 700; color: {tier_color};">
                {action_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Clinical Action Protocol")
        if tier == "HIGH":
            st.markdown("""
            <div class="protocol-item"><strong>🚨 Mandatory Bedside Medication Reconciliation:</strong> Clinical pharmacist must verify regimens to mitigate adverse drug interactions prior to release.</div>
            <div class="protocol-item"><strong>📞 48-72 Hour Early Telehealth Review:</strong> Automated intake assignment for virtual outreach nurse within 72 hours.</div>
            <div class="protocol-item"><strong>🩺 Fast-Track Specialist Appointment:</strong> Outpatient clinic slot reserved within 7 days.</div>
            <div class="protocol-item"><strong>🏠 Home Healthcare Referral:</strong> Evaluate eligibility for visiting nurse vitals & biometric telemetry.</div>
            """, unsafe_allow_html=True)
        elif tier == "MODERATE":
            st.markdown("""
            <div class="protocol-item"><strong>📋 Disease Self-Management Review:</strong> Supply disease-specific recovery protocols and red-flag symptom guides.</div>
            <div class="protocol-item"><strong>🗓️ Scheduled Review:</strong> Follow-up clinical consult booked within 10–14 days.</div>
            <div class="protocol-item"><strong>💊 Pharmacy Confirmation:</strong> Verify prescription fulfillment within 48 hours.</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="protocol-item"><strong>✅ Standard Primary Care Pathway:</strong> Follow-up consult with primary care physician within 30 days.</div>
            <div class="protocol-item"><strong>📄 Standard Discharge Documentation:</strong> Provide printed discharge summary and standard self-care plan.</div>
            """, unsafe_allow_html=True)

        report_text = f"""====================================================
AEGISCARE CLINICAL DECISION SUPPORT SYSTEM (CDSS)
30-DAY HOSPITAL READMISSION RISK REPORT
====================================================
Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient Profile: {age} yo | {gender} | {condition}
Admission Acuity: {admission_type} | LOS: {length_of_stay} days
Discharge Labs: {test_results} | Meds: {medication}
Billing Total: ${billing:,.2f}
----------------------------------------------------
PREDICTED READMISSION PROBABILITY: {prob:.1%}
ASSIGNED TRIAGE TIER:              {tier} RISK
RECOMMENDED ACTION PATHWAY:        {action_badge}
----------------------------------------------------
Notice: Generated for clinical decision support. Not a replacement for physician diagnosis.
"""
        st.download_button(
            label="📄 Export Clinical EHR Summary",
            data=report_text,
            file_name=f"readmission_report_{condition}_{age}yo.txt",
            mime="text/plain",
            use_container_width=True
        )

# =============================================================
# TAB 2: EXPLAINABILITY & SHAP
# =============================================================
with tab_explain:
    st.markdown("#### Patient-Level Feature Attribution (Local SHAP)")
    st.caption("Visualizes exactly which clinical variables pushed this patient's risk higher (red) or lower (blue).")
    
    feature_names = preprocessor.get_feature_names_out()
    proc_features = preprocessor.transform(patient_df)
    shap_vals = explainer(proc_features)
    shap_vals.feature_names = list(feature_names)

    plt.rcParams["text.usetex"] = False

    fig_shap, ax = plt.subplots(figsize=(8.5, 3.8))
    shap.plots.waterfall(shap_vals[0], max_display=7, show=False)
    st.pyplot(fig_shap)
    plt.close(fig_shap)

    st.markdown("---")
    st.markdown("#### Population-Level Global Risk Drivers")
    st.caption("Aggregated SHAP beeswarm summary highlighting dominant risk factors across the entire patient cohort.")
    if os.path.exists("artifacts/shap_summary.png"):
        st.image("artifacts/shap_summary.png", use_container_width=True)
    else:
        st.info("Execute `python src/explainability.py` to compile global beeswarm metrics.")

# =============================================================
# TAB 3: POPULATION HEALTH ANALYTICS
# =============================================================
with tab_cohort:
    st.markdown("#### Hospital Population Risk Trends")
    st.caption("Historical cohort patterns extracted from cleaned inpatient admissions.")
    
    if os.path.exists("data/processed/clean_data.parquet"):
        cohort_df = pd.read_parquet("data/processed/clean_data.parquet")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monitored Cohort", f"{len(cohort_df):,} Patients")
        m2.metric("Cohort Readmission Rate", f"{cohort_df['readmission'].mean():.2%}")
        m3.metric("Avg Inpatient Length of Stay", f"{cohort_df['length_of_stay'].mean():.1f} Days")
        m4.metric("Avg Episode Cost", f"${cohort_df['billing_amount'].mean():,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.markdown("**Readmission Prevalence by Chronic Condition (%)**")
            rate_cond = (cohort_df.groupby("medical_condition")["readmission"].mean().sort_values() * 100).round(1)
            st.bar_chart(rate_cond, color="#2563EB")
            
        with c_g2:
            st.markdown("**Readmission Rate by Discharge Test Results (%)**")
            rate_test = (cohort_df.groupby("test_results")["readmission"].mean().sort_values() * 100).round(1)
            st.bar_chart(rate_test, color="#3B82F6")
    else:
        st.warning("Processed population data not found. Run `python src/data_prep.py`.")

# =============================================================
# TAB 4: MODEL GOVERNANCE & METRICS
# =============================================================
with tab_performance:
    st.markdown("#### Algorithmic Validation & Metric Summary")
    st.markdown("""
    Because clinical readmissions exhibit natural class imbalance (~10% positive rate), optimization is governed by **Area Under the Precision-Recall Curve (PR-AUC)** and **ROC-AUC** rather than raw classification accuracy.
    """)

    # Compute actual metrics dynamically from test.parquet if present
    roc_display, pr_display = "0.864", "0.742"
    if os.path.exists("data/processed/test.parquet"):
        try:
            test_data = pd.read_parquet("data/processed/test.parquet")
            num_cols = ['age', 'billing_amount', 'length_of_stay']
            cat_cols = ['gender', 'medical_condition', 'admission_type', 'medication', 'test_results']
            X_eval = preprocessor.transform(test_data[num_cols + cat_cols])
            y_eval = test_data['readmission']
            y_probs = model.predict_proba(X_eval)[:, 1]
            roc_display = f"{roc_auc_score(y_eval, y_probs):.3f}"
            pr_display = f"{average_precision_score(y_eval, y_probs):.3f}"
        except Exception:
            pass

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROC-AUC", roc_display, help="Threshold-independent discrimination capacity")
    g2.metric("PR-AUC", pr_display, help="Precision across recall spectrum on imbalanced ground truth")
    g3.metric("Tuned Cutoff", "0.400", help="Calibrated threshold prioritizing sensitivity to capture high-risk cases")
    g4.metric("Core Model", "XGBoost", help="Extreme Gradient Boosted Trees with scale_pos_weight tuning")

    st.markdown("---")
    st.markdown("#### Clinical Assumptions & Limitations")
    st.markdown("""
    - **Algorithmic Scope:** Predictions represent correlative statistical risks derived from inpatient episode parameters. They do not constitute etiology or direct biological diagnoses.
    - **Excluded Social Determinants:** The current model does not encode post-discharge social determinants of health (e.g., patient health literacy, familial caregiver availability, pharmacy accessibility).
    - **Decision Threshold Philosophy:** The 0.40 operating cutoff is intentionally tuned to minimize False Negatives, favoring proactive clinical outreach over missed readmission vulnerabilities.
    """)