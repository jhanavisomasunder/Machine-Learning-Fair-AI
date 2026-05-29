import joblib
import pandas as pd

try:
    feature_names = joblib.load('feature_names.joblib')
    print(f"Feature Names: {feature_names}")

    df = pd.read_csv('data.csv')
    print("\nData Types:")
    print(df[feature_names].dtypes)

    for col in feature_names:
        print(f"\nProcessing column: {col}, dtype: {df[col].dtype}")
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Mean: {df[col].mean()}")
        else:
            print(f"Mode: {df[col].mode()[0]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
