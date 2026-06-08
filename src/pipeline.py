import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def generate_mock_data(n_samples=2000):
    """Generates synthetic telecom customer data for churn testing."""
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, size=n_samples),
        'Tenure': np.random.randint(1, 72, size=n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, size=n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.5, 0.3, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], size=n_samples)
    }
    df = pd.DataFrame(data)
    
    # Induce logical dependencies so the model has patterns to learn
    churn_prob = (
        (df['Contract'] == 'Month-to-month') * 0.4 +
        (df['MonthlyCharges'] > 80) * 0.3 +
        (df['Tenure'] < 12) * 0.2 +
        np.random.normal(0, 0.1, size=n_samples)
    )
    df['Churn'] = (churn_prob > 0.5).astype(int)
    return df

def build_model_pipeline():
    """Constructs an isolated preprocessing and XGBoost estimation pipeline."""
    numeric_features = ['Age', 'Tenure', 'MonthlyCharges']
    categorical_features = ['Contract', 'PaymentMethod', 'PaperlessBilling']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            scale_pos_weight=3, 
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    return model_pipeline, numeric_features, categorical_features