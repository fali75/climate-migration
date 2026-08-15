#!/usr/bin/env python3
"""
analysis/03_model_training.py
=============================
Orchestrates the training of machine learning and deep learning models.
This is a standalone entry point for model training, separate from the Colab pipeline.
"""

import os
import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_PATH = BASE_DIR / "data" / "processed" / "master_dataset.csv"
METRICS_DIR = BASE_DIR / "results" / "metrics"
FIG_DIR = BASE_DIR / "results" / "figures"

def main():
    print("=" * 60)
    print("Machine Learning Model Training Orchestrator")
    print("=" * 60)
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    
    print("\nTraining Ensemble ML Models...")
    from models.migration_ensemble import train_ensemble_regression, train_ensemble_classification
    from models.explainability import run_full_explainability
    
    # 1. Regression
    print("\n--- Regression (Displacement Magnitude) ---")
    reg_models, reg_metrics, importance, scaler, feature_cols = train_ensemble_regression(
        df, output_dir=str(METRICS_DIR)
    )
    
    # 2. Classification
    print("\n--- Classification (Displacement Severity) ---")
    cls_models, cls_metrics, cls_scaler, cls_feature_cols, class_names = train_ensemble_classification(
        df, output_dir=str(METRICS_DIR)
    )
    
    # 3. Explainability
    print("\n--- Running SHAP Explainability ---")
    from models.migration_ensemble import prepare_regression_data
    from sklearn.preprocessing import StandardScaler
    
    X, y, y_log, feature_cols = prepare_regression_data(df)
    X_filled = X.fillna(X.median())
    
    ss = StandardScaler()
    X_scaled = pd.DataFrame(ss.fit_transform(X_filled), columns=feature_cols, index=X_filled.index)
    
    # We omit Stacking Ensemble from SHAP as TreeExplainer doesn't support it directly
    base_models = {k: v for k, v in reg_models.items() if k != "Stacking Ensemble"}
    run_full_explainability(base_models, X_scaled, y_log, output_dir=str(FIG_DIR))
    
    print("\nModel training and evaluation complete.")

if __name__ == "__main__":
    main()
