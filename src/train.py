import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from data_prep import load_and_clean_data
from feature_engineering import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def run_training_pipeline():
    # Ensure destination folders exist
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Load and clean raw data
    df = load_and_clean_data("data/raw/healthcare_dataset.csv")
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    X = df[feature_cols]
    y = df['readmission']
    
    # 2. Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Save train.parquet and test.parquet
    train_df = X_train.copy()
    train_df['readmission'] = y_train
    train_df.to_parquet("data/processed/train.parquet", index=False)
    
    test_df = X_test.copy()
    test_df['readmission'] = y_test
    test_df.to_parquet("data/processed/test.parquet", index=False)
    print("Saved: data/processed/train.parquet and data/processed/test.parquet")
    
    # 4. Build and fit preprocessor
    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # 5. Handle class balance & train model
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = max(neg_count / max(pos_count, 1), 1.0)
    
    model = XGBClassifier(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=4,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_proc, y_train)
    
    # 6. Save joblib artifacts
    joblib.dump(model, "artifacts/model.joblib")
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
    print("Saved: artifacts/model.joblib and artifacts/preprocessor.joblib")

if __name__ == "__main__":
    run_training_pipeline()