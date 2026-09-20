import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def generate_shap_summary():
    model = joblib.load("artifacts/model.joblib")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    
    # Updated filename to test.parquet
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].sample(min(300, len(test_df)), random_state=42)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_proc)
    shap_values.feature_names = list(feature_names)
    
    # Save SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary.png", dpi=300)
    plt.close()
    print("SHAP beeswarm summary plot saved to artifacts/shap_summary.png")

if __name__ == "__main__":
    generate_shap_summary()