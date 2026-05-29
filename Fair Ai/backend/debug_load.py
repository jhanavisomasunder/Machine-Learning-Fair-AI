import joblib
import pandas as pd
import numpy as np
import shap
import os

print("Starting debug script...")
try:
    print("Loading model_standard.joblib...")
    model_standard = joblib.load('model_standard.joblib')
    print("Loading model_mitigated.joblib...")
    model_mitigated = joblib.load('model_mitigated.joblib')
    print("Loading feature_names.joblib...")
    feature_names = joblib.load('feature_names.joblib')
    print("Loading feature_names_mitigated.joblib...")
    feature_names_mitigated = joblib.load('feature_names_mitigated.joblib')
    print("Loading shap_background.joblib...")
    background_data = joblib.load('shap_background.joblib')
    print("Loading shap_background_mitigated.joblib...")
    background_data_mitigated = joblib.load('shap_background_mitigated.joblib')
    
    print("Initializing explainers...")
    explainer_standard = shap.LinearExplainer(model_standard.named_steps['classifier'], background_data)
    print("Standard explainer done.")
    explainer_mitigated = shap.LinearExplainer(model_mitigated.named_steps['classifier'], background_data_mitigated)
    print("Mitigated explainer done.")
    print("All loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
