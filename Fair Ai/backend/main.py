from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import io
import joblib
import pandas as pd
import numpy as np
import shap
import json
import os
from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference

app = FastAPI(title="Unbiased AI Decision System")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models and Artifacts
try:
    model_standard = joblib.load('model_standard.joblib')
    model_mitigated = joblib.load('model_mitigated.joblib')
    feature_names = joblib.load('feature_names.joblib')
    feature_names_mitigated = joblib.load('feature_names_mitigated.joblib')
    background_data = joblib.load('shap_background.joblib')
    background_data_mitigated = joblib.load('shap_background_mitigated.joblib')
    
    # Initialize SHAP explainers
    # Using LinearExplainer for LogisticRegression
    explainer_standard = shap.LinearExplainer(model_standard.named_steps['classifier'], background_data)
    explainer_mitigated = shap.LinearExplainer(model_mitigated.named_steps['classifier'], background_data_mitigated)
except Exception as e:
    print(f"Error loading models. Make sure to run model_trainer.py first. {e}")

class PredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str
    use_mitigated_model: bool = False

class RegisterRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str
    actual_outcome: int

@app.get("/")
def root():
    return RedirectResponse(url="/app")

@app.post("/register")
def register_data(request: RegisterRequest):
    new_data = {
        'age': request.age,
        'workclass': request.workclass,
        'fnlwgt': request.fnlwgt,
        'education': request.education,
        'education-num': request.education_num,
        'marital-status': request.marital_status,
        'occupation': request.occupation,
        'relationship': request.relationship,
        'race': request.race,
        'sex': request.sex,
        'capital-gain': request.capital_gain,
        'capital-loss': request.capital_loss,
        'hours-per-week': request.hours_per_week,
        'native-country': request.native_country,
        'target': request.actual_outcome
    }
    
    try:
        # Load and append
        df = pd.read_csv('data.csv')
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv('data.csv', index=False)
        
        # Recalculate metrics on the new dataset
        y = df['target']
        X = df.drop(columns=['target'])
        A = X['sex'] # sensitive feature
        
        y_pred_standard = model_standard.predict(X)
        dp_diff_standard = demographic_parity_difference(y, y_pred_standard, sensitive_features=A)
        eo_diff_standard = equal_opportunity_difference(y, y_pred_standard, sensitive_features=A)
        
        X_mit = X.drop(columns=['sex', 'race'], errors='ignore')
        y_pred_mitigated = model_mitigated.predict(X_mit)
        dp_diff_mitigated = demographic_parity_difference(y, y_pred_mitigated, sensitive_features=A)
        eo_diff_mitigated = equal_opportunity_difference(y, y_pred_mitigated, sensitive_features=A)
        
        metrics = {
            "standard": {
                "demographic_parity_diff": float(dp_diff_standard),
                "equal_opportunity_diff": float(eo_diff_standard)
            },
            "mitigated": {
                "demographic_parity_diff": float(dp_diff_mitigated),
                "equal_opportunity_diff": float(eo_diff_mitigated)
            }
        }
        
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        return {"message": "Data registered & Metrics updated!", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert input to DataFrame
    input_data = {
        'age': [request.age],
        'workclass': [request.workclass],
        'fnlwgt': [request.fnlwgt],
        'education': [request.education],
        'education-num': [request.education_num],
        'marital-status': [request.marital_status],
        'occupation': [request.occupation],
        'relationship': [request.relationship],
        'race': [request.race],
        'sex': [request.sex],
        'capital-gain': [request.capital_gain],
        'capital-loss': [request.capital_loss],
        'hours-per-week': [request.hours_per_week],
        'native-country': [request.native_country]
    }
    df = pd.DataFrame(input_data)
    
    try:
        if request.use_mitigated_model:
            # Prepare data for mitigated model
            df_mitigated = df.drop(columns=['sex', 'race'], errors='ignore')
            
            # Predict
            prob = model_mitigated.predict_proba(df_mitigated)[0][1]
            prediction = 1 if prob > 0.5 else 0
            
            # Explain with SHAP
            X_transformed = model_mitigated.named_steps['preprocessor'].transform(df_mitigated)
            shap_values = explainer_mitigated.shap_values(X_transformed)[0]
            
            # Get transformed feature names
            cat_features = model_mitigated.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
            num_features = model_mitigated.named_steps['preprocessor'].transformers_[0][2]
            all_feat_names = list(num_features) + list(cat_features)
            
        else:
            # Predict
            prob = model_standard.predict_proba(df)[0][1]
            prediction = 1 if prob > 0.5 else 0
            
            # Explain with SHAP
            X_transformed = model_standard.named_steps['preprocessor'].transform(df)
            shap_values = explainer_standard.shap_values(X_transformed)[0]
            
            # Get transformed feature names
            cat_features = model_standard.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
            num_features = model_standard.named_steps['preprocessor'].transformers_[0][2]
            all_feat_names = list(num_features) + list(cat_features)
        
        # Format SHAP values
        shap_dict = [{"feature": f, "value": float(v)} for f, v in zip(all_feat_names, shap_values)]
        shap_dict = sorted(shap_dict, key=lambda x: abs(x['value']), reverse=True)[:10] # Top 10 features
        
        return {
            "prediction": "Approved (>50K)" if prediction == 1 else "Rejected (<=50K)",
            "probability": float(prob),
            "shap_values": shap_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    if not os.path.exists('metrics.json'):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)

    try:
        df = pd.read_csv('data.csv')

        # Gender stats
        male_total    = len(df[df['sex'] == 'Male'])
        male_approved = len(df[(df['sex'] == 'Male') & (df['target'] == 1)])
        male_rate     = (male_approved / male_total) if male_total > 0 else 0

        female_total    = len(df[df['sex'] == 'Female'])
        female_approved = len(df[(df['sex'] == 'Female') & (df['target'] == 1)])
        female_rate     = (female_approved / female_total) if female_total > 0 else 0

        # Race stats
        white_total    = len(df[df['race'] == 'White'])
        white_approved = len(df[(df['race'] == 'White') & (df['target'] == 1)])
        white_rate     = (white_approved / white_total) if white_total > 0 else 0

        nonwhite_total    = len(df[df['race'] != 'White'])
        nonwhite_approved = len(df[(df['race'] != 'White') & (df['target'] == 1)])
        nonwhite_rate     = (nonwhite_approved / nonwhite_total) if nonwhite_total > 0 else 0

        metrics['dataset_stats'] = {
            "gender": {"Male": float(male_rate), "Female": float(female_rate)},
            "race":   {"White": float(white_rate), "Non-White": float(nonwhite_rate)}
        }

        # ── Bias Analysis ─────────────────────────────────────────
        dp_diff    = abs(metrics['standard']['demographic_parity_diff'])
        eo_diff    = abs(metrics['standard']['equal_opportunity_diff'])
        gender_gap = abs(male_rate - female_rate)
        race_gap   = abs(white_rate - nonwhite_rate)

        # Severity
        if dp_diff > 0.25 or eo_diff > 0.20:
            severity, severity_color = "High", "danger"
        elif dp_diff > 0.10 or eo_diff > 0.08:
            severity, severity_color = "Medium", "warning"
        else:
            severity, severity_color = "Low", "success"

        # Root causes
        causes = []
        if gender_gap > 0.08:
            disadvantaged = "Female" if female_rate < male_rate else "Male"
            causes.append({
                "type": "Gender Disparity",
                "icon": "⚧",
                "detail": (
                    f"The dataset shows a {gender_gap*100:.1f}% approval gap between "
                    f"Male ({male_rate*100:.1f}%) and Female ({female_rate*100:.1f}%) applicants. "
                    f"The model learned this historical pattern, systematically disadvantaging {disadvantaged} applicants."
                )
            })
        if race_gap > 0.05:
            disadvantaged_race = "Non-White" if nonwhite_rate < white_rate else "White"
            causes.append({
                "type": "Racial Disparity",
                "icon": "🌍",
                "detail": (
                    f"White applicants have a {white_rate*100:.1f}% approval rate vs "
                    f"{nonwhite_rate*100:.1f}% for Non-White applicants — a gap of {race_gap*100:.1f}%. "
                    f"This imbalance in training data directly propagates into model predictions, disadvantaging {disadvantaged_race} applicants."
                )
            })
        if dp_diff > 0.10:
            causes.append({
                "type": "Historical Data Imbalance",
                "icon": "📊",
                "detail": (
                    f"The Demographic Parity Difference of {dp_diff:.4f} indicates the model approves "
                    f"different demographic groups at unequal rates. This stems from structural inequalities "
                    f"encoded in the historical training dataset."
                )
            })
        if eo_diff > 0.08:
            causes.append({
                "type": "Unequal True Positive Rates",
                "icon": "📉",
                "detail": (
                    f"The Equal Opportunity Difference of {eo_diff:.4f} shows that qualified applicants "
                    f"from some groups are approved less often than equally qualified applicants from other groups — "
                    f"a form of systemic under-prediction."
                )
            })
        if not causes:
            causes.append({
                "type": "Minimal Detected Bias",
                "icon": "✅",
                "detail": "No significant bias detected in the current dataset. The model treats demographic groups comparably. Continue monitoring as new data is added."
            })

        # Ranked solutions
        mit_dp = metrics['mitigated']['demographic_parity_diff']
        solutions = [
            {
                "rank": 1,
                "name": "Blind Modeling",
                "tag": "Active via Toggle",
                "effectiveness": "High",
                "type": "Pre-processing",
                "description": (
                    f"Remove sensitive attributes (Sex, Race) so the model cannot directly use them. "
                    f"Toggle the switch in the sidebar to activate. "
                    f"This reduces Demographic Parity Difference from {dp_diff:.4f} to {abs(mit_dp):.4f}."
                ),
                "tradeoff": "May still use correlated proxies (e.g., occupation, zip code) indirectly."
            },
            {
                "rank": 2,
                "name": "Reweighing Training Data",
                "tag": "Recommended",
                "effectiveness": "High",
                "type": "Pre-processing",
                "description": (
                    f"Assign higher weights to under-represented positive outcomes (e.g., approved Female "
                    f"or Non-White applicants) before training. Directly counters the {gender_gap*100:.1f}% "
                    f"gender gap and {race_gap*100:.1f}% race gap in the dataset."
                ),
                "tradeoff": "Requires full training pipeline access. May slightly reduce overall accuracy."
            },
            {
                "rank": 3,
                "name": "Threshold Optimization",
                "tag": "Post-processing",
                "effectiveness": "Medium",
                "type": "Post-processing",
                "description": (
                    "Apply different decision thresholds per demographic group. Instead of a universal 50% "
                    "confidence cut-off, a disadvantaged group might only need 40% to be approved — "
                    "equalizing True Positive Rates across groups."
                ),
                "tradeoff": "Requires knowing the sensitive attribute at inference time."
            },
            {
                "rank": 4,
                "name": "Collect Representative Data",
                "tag": "Long-term Fix",
                "effectiveness": "Very High",
                "type": "Data Strategy",
                "description": (
                    f"Root cause: historical imbalance. Actively collect more data from under-represented groups "
                    f"(Non-White: {nonwhite_rate*100:.1f}% approval, Female: {female_rate*100:.1f}% approval) "
                    f"to reflect the true population and reduce bias at the source."
                ),
                "tradeoff": "Time-consuming — requires deliberate, long-term data collection strategy."
            }
        ]

        metrics['bias_analysis'] = {
            "severity": severity,
            "severity_color": severity_color,
            "dp_diff": float(dp_diff),
            "eo_diff": float(eo_diff),
            "gender_gap": float(gender_gap),
            "race_gap": float(race_gap),
            "causes": causes,
            "solutions": solutions
        }

    except Exception as e:
        print(f"Error calculating bias analysis: {e}")
        metrics['dataset_stats'] = None
        metrics['bias_analysis'] = None

    return metrics

@app.get("/report")
def get_report():
    if not os.path.exists('metrics.json'):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
        
    report = f"""
    ================================================
    FAIRNESS & BIAS MITIGATION REPORT
    ================================================
    
    1. STANDARD MODEL (BIASED)
    - Demographic Parity Difference: {metrics['standard']['demographic_parity_diff']:.4f}
    - Equal Opportunity Difference:  {metrics['standard']['equal_opportunity_diff']:.4f}
    
    2. MITIGATED MODEL (BIAS BLIND)
    - Demographic Parity Difference: {metrics['mitigated']['demographic_parity_diff']:.4f}
    - Equal Opportunity Difference:  {metrics['mitigated']['equal_opportunity_diff']:.4f}
    
    Conclusion:
    The mitigated model removes the sensitive attributes ('sex', 'race') 
    to reduce the disparities across demographic groups.
    ================================================
    """
    return {"report": report}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # ── Auto-detect columns ──────────────────────────────────
    cols = [c.lower() for c in df.columns]
    
    # Target detection (More aggressive)
    target_col = None
    target_keywords = ['target', 'label', 'approved', 'class', 'y', 'outcome', 'output', 'status', 'decision', 'prediction', 'accepted', 'result']
    for t in target_keywords:
        if t in cols:
            target_col = df.columns[cols.index(t)]
            break
    
    # Fallback: Check the last few columns for binary-like data (often where the label is)
    if not target_col:
        for col in reversed(df.columns):
            try:
                unique_vals = set(df[col].dropna().unique())
                if len(unique_vals) == 2:
                    # Check for common binary pairs
                    if any(pair.issubset(unique_vals) for pair in [{0, 1}, {'0', '1'}, {'Yes', 'No'}, {'Approved', 'Rejected'}, {'Male', 'Female'}]):
                        # But don't pick gender/sex as target
                        if col.lower() not in ['sex', 'gender', 'race']:
                            target_col = col
                            break
            except: continue
    
    # Sex detection
    sex_col = None
    for s in ['sex', 'gender', 'male', 'female', 's']:
        if s in cols:
            sex_col = df.columns[cols.index(s)]
            break
            
    # Race detection
    race_col = None
    for r in ['race', 'ethnicity', 'origin', 'r']:
        if r in cols:
            race_col = df.columns[cols.index(r)]
            break

    try:
        # ── Statistical Defaults ─────────────────────────────────
        full_template = pd.read_csv('data.csv')
        defaults = {}
        for col in feature_names:
            if pd.api.types.is_numeric_dtype(full_template[col]):
                defaults[col] = full_template[col].mean()
            else:
                defaults[col] = full_template[col].mode()[0]

        def align_features(input_df, expected_features):
            aligned = pd.DataFrame(index=input_df.index)
            for feat in expected_features:
                if feat in input_df.columns:
                    aligned[feat] = input_df[feat]
                else:
                    aligned[feat] = defaults.get(feat, 0)
            return aligned

        X_std = align_features(df, feature_names)
        X_mit = align_features(df, feature_names_mitigated)
        
        # Target/Historical Ground Truth
        y_raw = df[target_col] if target_col else pd.Series([0]*len(df))
        
        # Aggressive mapping to 0/1
        def map_y(val):
            v = str(val).lower().strip()
            if v in ['1', '1.0', 'approved', 'yes', 'true', 'accept', 'accepted']: return 1
            if v in ['0', '0.0', 'rejected', 'no', 'false', 'deny', 'denied']: return 0
            return 1 if val == df[target_col].unique()[0] else 0 # Fallback
            
        y = y_raw.apply(map_y)
        
        # ── Intelligence: Use CSV's own prediction as 'Standard' if possible ─────
        # If the target_col exists and looks like a model output, use it as y_pred_std
        # Otherwise run our own standard model
        if target_col:
            y_pred_std = y
        else:
            y_pred_std = model_standard.predict(X_std)

        # ── Run Our Mitigated Model as Comparison ────────────────
        y_pred_mit = model_mitigated.predict(X_mit)
        
        # Sensitive feature
        A_col = sex_col or race_col or (df.select_dtypes(include=['object']).columns[0] if not df.select_dtypes(include=['object']).columns.empty else df.columns[0])
        A = df[A_col]

        # ── Calculate Metrics ────────────────────────────────────
        dp_std = demographic_parity_difference(y, y_pred_std, sensitive_features=A)
        eo_std = equal_opportunity_difference(y, y_pred_std, sensitive_features=A) if target_col else 0

        dp_mit = demographic_parity_difference(y, y_pred_mit, sensitive_features=A)
        eo_mit = equal_opportunity_difference(y, y_pred_mit, sensitive_features=A) if target_col else 0

        # Dataset Stats
        vals = A.unique()
        stats = {}
        dataset_dp = 0
        if len(vals) >= 2:
            v1, v2 = vals[0], vals[1]
            d1 = df[df[A_col] == v1]; d2 = df[df[A_col] == v2]
            r1 = len(d1[d1[target_col] == 1]) / len(d1) if (target_col and len(d1) > 0) else 0
            r2 = len(d2[d2[target_col] == 1]) / len(d2) if (target_col and len(d2) > 0) else 0
            stats = {str(v1): float(r1), str(v2): float(r2)}
            dataset_dp = abs(r1 - r2)
        elif len(vals) == 1:
            stats = {str(vals[0]): 0.0}

        severity, color = ("High", "danger") if max(dp_std, dataset_dp) > 0.20 else ("Medium", "warning") if max(dp_std, dataset_dp) > 0.08 else ("Low", "success")

        analysis = {
            "severity": severity,
            "severity_color": color,
            "dp_diff": float(dp_std),
            "eo_diff": float(eo_std),
            "dataset_bias": float(dataset_dp),
            "gender_gap": float(dataset_dp),
            "race_gap": 0.0,
            "standard": {"dp": float(dp_std), "eo": float(eo_std)},
            "mitigated": {"dp": float(dp_mit), "eo": float(eo_mit)},
            "dataset_stats": {"detected_attribute": A_col, "values": stats},
            "causes": [
                {"type": "Column Detection", "icon": "🔍", "detail": f"Target: {'Detected ('+target_col+')' if target_col else 'Not Found (Using dummy)'}. Sensitive Feature: Detected ({A_col})."},
                {"type": "Inherent Data Bias", "icon": "📊", "detail": f"The input CSV itself contains a {dataset_dp*100:.1f}% disparity in historical outcomes between '{A_col}' groups."}
            ],
            "solutions": [
                {"rank": 1, "name": "Reweighing", "tag": "Recommended", "effectiveness": "High", "type": "Pre-processing", "description": "Adjust the importance of biased rows in the dataset to balance the historical outcome disparity.", "tradeoff": "Moderate impact on accuracy."}
            ]
        }
        return {"status": "success", "analysis": analysis}

    except Exception as e:
        print(f"ERROR in upload_csv: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Serve frontend static files (must be LAST — after all API routes)
app.mount("/app", StaticFiles(directory="../frontend", html=True), name="frontend")
