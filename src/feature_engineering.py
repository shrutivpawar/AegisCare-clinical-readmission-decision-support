from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = ['age', 'billing_amount', 'length_of_stay']
CATEGORICAL_FEATURES = ['gender', 'medical_condition', 'admission_type', 'medication', 'test_results']

def build_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES):
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor