# AegisCare: Predictive Hospital Readmission Risk System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)

> **Clinical Decision Support System (CDSS) Notice:** This platform provides statistical risk stratification to assist hospital discharge planning teams and preventive intervention coordination. It is strictly an analytical decision-support instrument and does not provide automated diagnoses or override licensed clinical discretion.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Clinical Rationale & Problem Statement](#clinical-rationale--problem-statement)
3. [System Architecture](#system-architecture)
4. [Dataset & Preprocessing](#dataset--preprocessing)
5. [Model Training & Evaluation Methodology](#model-training--evaluation-methodology)
6. [Explainable AI (SHAP Integration)](#explainable-ai-shap-integration)
7. [Clinical Decision Support Dashboard](#clinical-decision-support-dashboard)
8. [Project Structure](#project-structure)
9. [Installation & Setup](#installation--setup)
10. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
11. [Assumptions, Limitations & Responsible Use](#assumptions-limitations--responsible-use)

---

## Project Overview

Patients with chronic conditions (such as Diabetes, Hypertension, and Cancer) face heightened vulnerability following inpatient discharge due to physiological instability, medication regimen adjustments, and post-discharge self-care challenges. Without risk stratification, care teams miss timely windows for proactive intervention.

**AegisCare** addresses this challenge by providing an end-to-end predictive pipeline that ingests inpatient electronic health records, estimates 30-day readmission risk, isolates patient-specific risk drivers via SHAP (SHapley Additive exPlanations), and presents actionable follow-up protocols through an interactive clinical workstation.

---

## Clinical Rationale & Problem Statement

* **The Clinical Bottleneck:** Discharge coordinators review high volumes of inpatient charts under severe time constraints, making systematic identification of post-discharge deterioration risks difficult.
* **The Solution:** A calibrated machine learning pipeline operating on standard admission data to categorize patients into stratified tiers (**High**, **Moderate**, **Low** Risk) and assign concrete interventions (such as 48–72 hour telehealth visits, home nursing visits, and bedside medication reconciliation).
* **Governance Boundary:** Delineates algorithmic probability estimation from medical diagnosis, ensuring technology serves as an assistant rather than an autonomous decision-maker.

---

## System Architecture

```text
+-------------------------+
| Kaggle Healthcare Data  |
+------------+------------+
             |
             v
+-------------------------+
| Data Preprocessing      | --> Clean text, calculate Length of Stay (LOS),
| (src/data_prep.py)      |     derive readmission risk ground truth
+------------+------------+
             |
             v
+-------------------------+
| Feature Transformation  | --> StandardScaler (continuous) +
| (feature_engineering.py)|     OneHotEncoder (categorical) via ColumnTransformer
+------------+------------+
             |
             v
+-------------------------+
| Model Training & Tuning | --> Stratified split + scale_pos_weight
| (src/train.py)          |     XGBoost Classifier serialization
+------------+------------+
             |
      +------+------+
      |             |
      v             v
+-----------+ +-----------+
| Metrics   | | SHAP XAI  |
| (eval.py) | | (exp.py)  |
+-----+-----+ +-----+-----+
      |             |
      +------+------+
             |
             v
+-------------------------------------------------------+
| Interactive Clinical Decision Support UI (Streamlit)  |
| - Patient Triage & Protocol Recommendation            |
| - Individual & Global Feature Attribution             |
| - Population Health Inpatient Cohort Trends           |
| - Governance & Technical Metric Audit                 |
+-------------------------------------------------------+

```

## Dataset & Preprocessing

The pipeline utilizes the mandated **Kaggle Healthcare Dataset** (`healthcare_dataset.csv`):

* **Temporal Attributes:** Ingestion of `Date of Admission` and `Discharge Date` to calculate continuous inpatient Length of Stay (LOS) in days.
* **Categorical Normalization:** Standardization of nominal features (`Medical Condition`, `Admission Type`, `Medication`, `Test Results`, `Gender`).
* **Feature Leakage Prevention:** Fitting transformations solely on the training split using a reusable scikit-learn `ColumnTransformer`.
* **Cohort Ground Truth Derivation:** Incorporates a composite clinical risk function reflecting multi-factor vulnerability (age brackets >65/75, emergency acuity, abnormal discharge labs, extended LOS, and severe chronic comorbidities) yielding a balanced ~10% clinical readmission baseline.

---

## Model Training & Evaluation Methodology

In clinical readmission modeling, raw classification accuracy is a flawed metric due to class imbalance (~90% negative, ~10% positive). The system emphasizes threshold-independent discrimination and precision-recall trade-offs:

### Primary Evaluation Metrics
* **PR-AUC (Average Precision):** Measures precision across varying recall thresholds under severe imbalance.
* **ROC-AUC:** Assesses overall class separation capability.
* **Recall (Clinical Sensitivity):** Prioritized to minimize False Negatives (missed readmissions).

### Operational Configuration & Algorithms
* **Operational Threshold Tuning:** A calibrated cutoff of **0.40** is applied in place of standard 0.50, deliberately trading moderate False Positive rates for aggressive capture of vulnerable patients requiring follow-up.
* **Algorithm Choice:** **XGBoost (Extreme Gradient Boosting)** with tree regularization and positive class weighting (`scale_pos_weight`) to manage skewed class ratios.

---

## Explainable AI (SHAP Integration)

To eliminate the "black-box" risk in clinical deployments, the framework integrates TreeSHAP:

* **Local Attribution (Patient Waterfall):** Visualizes the exact directional push ($\pm \text{log-odds}$) exerted by each individual variable on a specific patient's risk score.
* **Global Attribution (Cohort Beeswarm):** Aggregates feature importance across hundreds of admissions to validate that model mechanics align with clinical science (e.g., Abnormal lab results and prolonged emergency stays acting as primary risk drivers).

---

## Clinical Decision Support Dashboard

The user interface (`app/dashboard.py`) is structured into four functional modules:

* **Patient Risk Assessment & Triage:** Input or quick-load patient profiles to receive an immediate 30-day risk probability, risk tier badge, and targeted clinical protocols (such as pharmacy bedside reconciliation or rapid telehealth visits), with an exportable text summary report.
* **Explainability & SHAP Drivers:** Direct visualization of the local patient waterfall explanation alongside the global cohort feature beeswarm.
* **Population Health Analytics:** Macro-level trends tracking readmission rates by chronic condition, lab result breakdowns, and inpatient cost baselines.
* **Model Validation & Governance:** Dynamic display of ROC-AUC and PR-AUC scores, operational threshold details, and clear clinical limitations.

---

## Project Structure

```text
chronic_readmission_risk/
│
├── app/
│   └── dashboard.py               # Multi-tab Streamlit Clinical Decision Support UI
│
├── artifacts/                     # Serialized pipelines & visual assets
│   ├── model.joblib               # Trained XGBoost Classifier
│   ├── preprocessor.joblib        # Scikit-learn ColumnTransformer
│   └── shap_summary.png           # Global SHAP beeswarm plot
│
├── data/
│   ├── raw/
│   │   └── healthcare_dataset.csv # Primary Kaggle Healthcare dataset
│   └── processed/
│       ├── clean_data.parquet     # Standardized full cohort data
│       ├── train.parquet          # Stratified training split
│       └── test.parquet           # Stratified test holdout
│
├── notebooks/
│   └── 01_eda_and_prototyping.ipynb # Exploratory data analysis & prototype curves
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py               # Data ingestion, cleaning & target formulation
│   ├── feature_engineering.py     # Scalers, encoders & transformation pipelines
│   ├── train.py                   # Model fitting, evaluation & artifact persistence
│   ├── evaluate.py                # Standalone metrics & confusion matrix reporting
│   └── explainability.py          # SHAP calculation & beeswarm figure generation
│
├── .gitignore                     # Git tracking exclusions
├── requirements.txt               # Pinned package dependencies
└── README.md                      # Comprehensive project documentation

```

## Installation & Setup
Prerequisites
* Python 3.10, 3.11, or 3.12
* Git
* VS Code (recommended editor)

## Live Demo Link:
* Loom: https://www.loom.com/share/a1bf087e9a534136b13e531eeaea2604
* Youtube: https://youtu.be/mSmD0vtfE3I?si=pu7yoZqyKZKgN0Lp
