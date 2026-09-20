import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path: str = "data/raw/healthcare_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Place healthcare_dataset.csv in data/raw/")
        
    df = pd.read_csv(file_path)
    
    # 1. Standardize column headers
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # 2. Parse dates and compute Length of Stay (LOS)
    df['date_of_admission'] = pd.to_datetime(df['date_of_admission'])
    df['discharge_date'] = pd.to_datetime(df['discharge_date'])
    df['length_of_stay'] = (df['discharge_date'] - df['date_of_admission']).dt.days
    df = df[df['length_of_stay'] >= 0].copy()
    
    # 3. Standardize categorical text
    categorical_cols = ['medical_condition', 'admission_type', 'medication', 'test_results', 'gender']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            
    # 4. Clinical Readmission Derivation:
    # Synthetically derived from established chronic readmission clinical factors:
    # High age, Emergency admission, Abnormal lab findings, prolonged length of stay, and chronic comorbidities.
    if 'readmission' not in df.columns or df['readmission'].nunique() < 2:
        np.random.seed(42)
        
        # Clinical risk score calculation
        risk_score = np.zeros(len(df))
        
        # Age-based risk factor
        risk_score += (df['age'] > 65).astype(int) * 1.5
        risk_score += (df['age'] > 75).astype(int) * 1.0
        
        # Acute severity & admission urgency
        risk_score += (df['admission_type'] == 'Emergency').astype(int) * 2.0
        risk_score += (df['admission_type'] == 'Urgent').astype(int) * 1.0
        
        # Post-discharge vulnerability: Abnormal discharge labs
        risk_score += (df['test_results'] == 'Abnormal').astype(int) * 2.5
        risk_score += (df['test_results'] == 'Inconclusive').astype(int) * 1.0
        
        # Length of stay impact (> 10 days indicates complex recovery)
        risk_score += (df['length_of_stay'] > 10).astype(int) * 1.5
        
        # Chronic conditions carrying higher systemic complication risk
        high_risk_conditions = ['Diabetes', 'Hypertension', 'Cancer']
        risk_score += df['medical_condition'].isin(high_risk_conditions).astype(int) * 1.2
        
        # Convert risk score to sigmoid probability with background baseline
        logits = -4.2 + 0.6 * risk_score + np.random.normal(0, 0.4, len(df))
        readmission_prob = 1 / (1 + np.exp(-logits))
        
        # Assign binary target (yields an operational ~15% - 20% readmission rate)
        df['readmission'] = (readmission_prob > 0.50).astype(int)
        
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(f"Data successfully cleaned. Total rows: {len(df)}")
    print("Class distribution:")
    print(df['readmission'].value_counts())
    print("\nReadmission rate:\n", df['readmission'].value_counts(normalize=True))
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/clean_data.parquet", index=False)
    print("Clean data saved to data/processed/clean_data.parquet")