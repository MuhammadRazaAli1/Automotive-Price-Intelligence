"""
Automative Price Intelligence - Model Training Script
Author: Muhammad Raza Ali
Description: This script cleans the raw car dataset, engineers features, 
and trains a scikit-learn machine learning pipeline using a Random Forest Regressor.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------- Configuration ----------
EXPECTED_COLS = [
    "price", "year", "odometer", "make", "model", "fuel", "transmission", "condition",
    "engine_cc", "power_hp", "seats", "body_type", "state", "title", "description", "seller_type"
]

COLUMN_ALIASES = {
    "mileage_km": "odometer", "mileage": "odometer", "kms_driven": "odometer",
    "engine_size_cc": "engine_cc", "hp": "power_hp", "power_bhp": "power_hp",
    "province": "state", "location_state": "state", "desc": "description",
    "name": "title", "seller": "seller_type", "Price": "price", "Year": "year"
}

RANDOM_STATE = 42

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to lowercase and applies known aliases."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    for col in list(df.columns):
        if col in COLUMN_ALIASES:
            new_col = COLUMN_ALIASES[col]
            if new_col not in df.columns:
                df[new_col] = df[col]
    return df

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers new features like car age and processes text descriptions."""
    df = df.copy()
    current_year = datetime.now().year
    
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["car_age"] = np.clip(current_year - df["year"], 0, None)
    else:
        df["car_age"] = np.nan

    if "odometer" in df.columns:
        df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

    # Combine title and description for NLP processing
    title = df["title"].fillna("") if "title" in df.columns else ""
    desc = df["description"].fillna("") if "description" in df.columns else ""
    df["combined_text"] = (title.astype(str) + " " + desc.astype(str)).str.strip()

    return df

def split_data(df: pd.DataFrame, target_col: str = "price"):
    """Splits the dataset into training and testing sets."""
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Builds a scikit-learn preprocessing and modeling pipeline."""
    numeric_cols = [c for c in ["year", "odometer", "engine_cc", "power_hp", "seats", "car_age"] if c in X.columns]
    categorical_cols = [c for c in ["make", "model", "fuel", "transmission", "condition", "body_type", "state", "seller_type"] if c in X.columns]
    text_col = "combined_text" if "combined_text" in X.columns else None

    transformers = []
    
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols))

    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols))

    if text_col:
        transformers.append(("txt", TfidfVectorizer(max_features=300, stop_words='english'), text_col))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Using Random Forest for better accuracy and robust predictions
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

def main():
    parser = argparse.ArgumentParser(description="Train the UniCarPrice Prediction Model")
    parser.add_argument("--csv", type=str, default="data/raw/sample_cars.csv", help="Path to input CSV")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the model")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Dataset not found at {args.csv}", file=sys.stderr)
        sys.exit(1)

    print("Loading and preparing data...")
    df = pd.read_csv(args.csv)
    df = standardize_columns(df)
    
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = derive_features(df)
    X_train, X_test, y_train, y_test = split_data(df, target_col="price")

    print("Building and training pipeline...")
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    preds = pipeline.predict(X_test)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2_Score": float(r2_score(y_test, preds))
    }
    
    print(f"Metrics: {metrics}")

    os.makedirs(args.save_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(args.save_dir, "model.joblib"))
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Model successfully saved to {args.save_dir}/model.joblib")

if __name__ == "__main__":
    main()
