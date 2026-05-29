import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import joblib
import json

def load_data():
    print("Generating synthetic dataset...")
    np.random.seed(42)
    n = 10000
    
    age = np.random.randint(18, 70, n)
    workclass = np.random.choice(['Private', 'Self-emp-not-inc', 'Local-gov'], n)
    fnlwgt = np.random.randint(50000, 300000, n)
    education = np.random.choice(['Bachelors', 'HS-grad', 'Masters'], n)
    education_num = np.where(education == 'Bachelors', 13, np.where(education == 'Masters', 14, 9))
    marital_status = np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced'], n)
    occupation = np.random.choice(['Exec-managerial', 'Prof-specialty', 'Craft-repair'], n)
    relationship = np.random.choice(['Husband', 'Wife', 'Not-in-family'], n)
    race = np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n)
    sex = np.random.choice(['Male', 'Female'], n)
    capital_gain = np.random.exponential(1000, n)
    capital_loss = np.random.exponential(100, n)
    hours_per_week = np.random.randint(20, 60, n)
    native_country = np.random.choice(['United-States', 'Mexico', 'Canada', 'India', 'Germany', 'United-Kingdom', 'China', 'Japan', 'France', 'Italy', 'Philippines', 'El-Salvador', 'Cuba'], n)
    
    # Create target with intentional bias to demonstrate fairness metrics
    score = (age * 0.05) + (education_num * 0.2) + (capital_gain * 0.001) + (hours_per_week * 0.02)
    score += np.where(sex == 'Male', 1.0, 0.0) # Bias
    score += np.where(race == 'White', 0.5, 0.0) # Bias
    score += np.random.normal(0, 1, n)
    
    target = (score > np.median(score)).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country,
        'target': target
    })
    
    return df

def train_and_evaluate():
    df = load_data()
    df.to_csv('data.csv', index=False)
    
    y = df['target']
    X = df.drop(columns=['target'])
    
    # Identify sensitive feature
    sensitive_feature = 'sex'
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    A_train = X_train[sensitive_feature]
    A_test = X_test[sensitive_feature]
    
    # Define preprocessing
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # 1. Train Standard Model (Biased)
    print("Training standard model...")
    pipeline_standard = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline_standard.fit(X_train, y_train)
    
    # Evaluate Standard Model
    y_pred_standard = pipeline_standard.predict(X_test)
    
    dp_diff_standard = demographic_parity_difference(y_test, y_pred_standard, sensitive_features=A_test)
    eo_diff_standard = equal_opportunity_difference(y_test, y_pred_standard, sensitive_features=A_test)
    
    print(f"Standard Model - Demographic Parity Diff: {dp_diff_standard:.4f}")
    print(f"Standard Model - Equal Opportunity Diff: {eo_diff_standard:.4f}")
    
    # 2. Train Mitigated Model (Remove sensitive attribute 'sex' and 'race' as a baseline mitigation)
    print("Training mitigated model (blind to sensitive attributes)...")
    # A simple mitigation technique: remove sensitive columns
    X_train_mitigated = X_train.drop(columns=['sex', 'race'], errors='ignore')
    X_test_mitigated = X_test.drop(columns=['sex', 'race'], errors='ignore')
    
    cat_mitigated = [c for c in categorical_features if c not in ['sex', 'race']]
    num_mitigated = [n for n in numeric_features if n not in ['sex', 'race']]
    
    preprocessor_mitigated = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_mitigated),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_mitigated)
        ])
        
    pipeline_mitigated = Pipeline(steps=[
        ('preprocessor', preprocessor_mitigated),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline_mitigated.fit(X_train_mitigated, y_train)
    
    # Evaluate Mitigated Model
    y_pred_mitigated = pipeline_mitigated.predict(X_test_mitigated)
    dp_diff_mitigated = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=A_test)
    eo_diff_mitigated = equal_opportunity_difference(y_test, y_pred_mitigated, sensitive_features=A_test)
    
    print(f"Mitigated Model - Demographic Parity Diff: {dp_diff_mitigated:.4f}")
    print(f"Mitigated Model - Equal Opportunity Diff: {eo_diff_mitigated:.4f}")
    
    # Save metrics
    metrics = {
        "standard": {
            "demographic_parity_diff": dp_diff_standard,
            "equal_opportunity_diff": eo_diff_standard
        },
        "mitigated": {
            "demographic_parity_diff": dp_diff_mitigated,
            "equal_opportunity_diff": eo_diff_mitigated
        }
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    # Save models and preprocessors
    joblib.dump(pipeline_standard, 'model_standard.joblib')
    joblib.dump(pipeline_mitigated, 'model_mitigated.joblib')
    joblib.dump(X_train.columns.tolist(), 'feature_names.joblib')
    joblib.dump(X_train_mitigated.columns.tolist(), 'feature_names_mitigated.joblib')
    
    # Background dataset for SHAP (subset)
    X_train_transformed = preprocessor.transform(X_train)
    background_data = X_train_transformed[:100]
    joblib.dump(background_data, 'shap_background.joblib')
    
    X_train_mit_transformed = preprocessor_mitigated.transform(X_train_mitigated)
    background_data_mitigated = X_train_mit_transformed[:100]
    joblib.dump(background_data_mitigated, 'shap_background_mitigated.joblib')
    
    print("Training complete. Models and metrics saved.")

if __name__ == "__main__":
    train_and_evaluate()
