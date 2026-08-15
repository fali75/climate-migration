#!/usr/bin/env python3
"""
analysis/02_statistical_tests.py
================================
Performs baseline statistical testing and linear regression models
to establish the relationship between climate variables and displacement
prior to applying complex machine learning models.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Attempt to import statsmodels for statistical tests
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Some tests will be skipped.")

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "master_dataset.csv"
OUTPUT_DIR = BASE_DIR / "results" / "stats"

def load_data():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        sys.exit(1)
    return pd.read_csv(DATA_PATH)

def run_baseline_ols(df):
    """Run Ordinary Least Squares (OLS) regression as a baseline."""
    print("\n--- Running Baseline OLS Regression ---")
    
    if not STATSMODELS_AVAILABLE:
        print("statsmodels is required for OLS regression. Skipping.")
        return
    
    # Prepare data
    target = 'new_displacement_disasters'
    if target not in df.columns:
        print("Target variable not found.")
        return
        
    valid_df = df.dropna(subset=[target]).copy()
    
    # Use log displacement
    valid_df['log_disp'] = np.log1p(valid_df[target])
    
    # Select independent variables
    features = ['CESI', 'gdp_per_capita_usd', 'population_density', 'employment_agriculture_pct']
    features = [f for f in features if f in valid_df.columns]
    
    # Drop rows with NaNs in features
    reg_df = valid_df.dropna(subset=features)
    
    if len(reg_df) < 20:
        print("Not enough data points for regression after dropping NaNs.")
        return
        
    # Standardize features for comparable coefficients
    for f in features:
        reg_df[f] = (reg_df[f] - reg_df[f].mean()) / reg_df[f].std()
        
    # Build formula
    formula = f"log_disp ~ {' + '.join(features)}"
    
    # Fit model
    model = smf.ols(formula=formula, data=reg_df).fit()
    
    print(model.summary())
    
    # Save summary to text file
    with open(OUTPUT_DIR / "ols_regression_summary.txt", "w") as f:
        f.write(model.summary().as_text())
        
    print(f"Saved OLS summary to {OUTPUT_DIR / 'ols_regression_summary.txt'}")

def main():
    print("=" * 60)
    print("Statistical Tests & Baseline Models")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    
    run_baseline_ols(df)
    
    print("\nStatistical testing completed.")

if __name__ == "__main__":
    main()
