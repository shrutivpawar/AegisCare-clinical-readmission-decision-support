import joblib
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def evaluate_model():
    model = joblib.load("artifacts/model.joblib")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df['readmission']
    
    X_test_proc = preprocessor.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
    
    # Clinical decision threshold (0.40 favors catching readmissions)
    THRESHOLD = 0.40
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)
    
    print("=" * 45)
    print("CLINICAL READMISSION EVALUATION REPORT")
    print("=" * 45)
    print(f"ROC-AUC Score:          {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_pred_proba):.4f}")
    print(f"Operational Cutoff:     {THRESHOLD}")
    print("-" * 45)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    print("-" * 45)
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    evaluate_model()