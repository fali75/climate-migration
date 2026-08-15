#!/usr/bin/env python3
"""
colab_full_pipeline.py
======================
Google Colab-Ready Full Pipeline Script

This script is designed to be run on Google Colab for computation-intensive tasks.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Upload this entire repository to Google Drive
2. Open this file in Google Colab (File > Open in Colab)
3. Connect to a GPU runtime (Runtime > Change runtime type > GPU)
4. Mount your Google Drive
5. Run cells sequentially

NOTE: This is a .py file (not .ipynb) for Git compatibility.
Convert to notebook in Colab or run as script.
"""

# ============================================================================
# CELL 1: Setup & Installation
# ============================================================================

"""
# Run in Colab:
!pip install -q xgboost lightgbm catboost shap mesa optuna plotly kaleido wbdata geopandas

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set project directory
import os
PROJECT_DIR = '/content/drive/MyDrive/climate-migration'  # Adjust path
os.chdir(PROJECT_DIR)
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Add project root to path (handle both script and notebook environments)
try:
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_DIR = os.getcwd()
sys.path.insert(0, PROJECT_DIR)


# ============================================================================
# CELL 2: Data Acquisition (Run once, then cached)
# ============================================================================

def run_data_acquisition():
    """Run all data acquisition scripts."""
    print("=" * 70)
    print("PHASE 1: DATA ACQUISITION")
    print("=" * 70)
    
    scripts = [
        ("NASA POWER Climate Data", "scripts/01_fetch_nasa_power.py"),
        ("World Bank Indicators", "scripts/02_fetch_worldbank.py"),
        ("ND-GAIN Vulnerability", "scripts/03_fetch_ndgain.py"),
        ("EM-DAT Disasters", "scripts/04_process_emdat.py"),
        ("IDMC Displacement", "scripts/05_process_idmc.py"),
        ("Conflict Data", "scripts/06_fetch_acled.py"),
    ]
    
    for name, script_path in scripts:
        full_path = os.path.join(PROJECT_DIR, script_path)
        if os.path.exists(full_path):
            print(f"\n{'─'*50}")
            print(f"Running: {name}")
            print(f"{'─'*50}")
            try:
                exec(open(full_path).read(), {"__name__": "__main__"})
            except Exception as e:
                print(f"  Error in {name}: {e}")
                print("  Continuing with next script...")
    
    # Merge all datasets
    print(f"\n{'─'*50}")
    print("Running: Master Data Merge")
    print(f"{'─'*50}")
    merge_path = os.path.join(PROJECT_DIR, "scripts/07_merge_datasets.py")
    if os.path.exists(merge_path):
        exec(open(merge_path).read(), {"__name__": "__main__"})


# ============================================================================
# CELL 3: Load Processed Data
# ============================================================================

def load_master_data():
    """Load the processed master dataset."""
    data_path = os.path.join(PROJECT_DIR, "data/processed/master_dataset.csv")
    
    if not os.path.exists(data_path):
        print("Master dataset not found. Running data acquisition...")
        run_data_acquisition()
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Master dataset loaded: {df.shape}")
        print(f"Countries: {df['iso3'].nunique()}")
        print(f"Years: {df['year'].min()}-{df['year'].max()}")
        print(f"Features: {len(df.columns)}")
        return df
    else:
        print("ERROR: Could not load or create master dataset.")
        return None


# ============================================================================
# CELL 4: Deep Learning Model Training
# ============================================================================

def train_deep_learning_models(df):
    """Train all three deep learning architectures."""
    print("\n" + "=" * 70)
    print("PHASE 2: DEEP LEARNING CLIMATE FORECASTING")
    print("=" * 70)
    
    from models.climate_lstm_attention import build_lstm_attention_model
    from models.climate_transformer import build_transformer_model
    from models.cnn_lstm_hybrid import build_cnn_lstm_model
    
    # Load daily climate data for sequence modeling
    daily_path = os.path.join(PROJECT_DIR, "data/raw/nasa_power_daily_all.csv")
    
    if os.path.exists(daily_path):
        daily = pd.read_csv(daily_path)
        print(f"Daily climate data: {len(daily)} records")
    else:
        print("Daily climate data not available. Skipping DL training.")
        print("NOTE: Run NASA POWER data fetch first (takes ~30 min)")
        return None
    
    from models.climate_lstm_attention import train_lstm_attention
    
    output_dir = os.path.join(PROJECT_DIR, "results/metrics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Train LSTM-Attention
    print("\n--- Training LSTM-Attention ---")
    try:
        model, history, metrics, scaler, features = train_lstm_attention(
            daily, sequence_length=12, epochs=50, batch_size=32,
            output_dir=output_dir
        )
        print(f"LSTM-Attention trained. Metrics: {metrics}")
    except Exception as e:
        print(f"LSTM-Attention training error: {e}")
        model, history, metrics = None, None, None
    
    return model, history, metrics


# ============================================================================
# CELL 5: Ensemble ML Training
# ============================================================================

def train_ensemble_models(df):
    """Train ensemble ML models with SHAP explainability."""
    print("\n" + "=" * 70)
    print("PHASE 3: ENSEMBLE ML MIGRATION PREDICTION")
    print("=" * 70)
    
    sys.path.insert(0, PROJECT_DIR)
    from models.migration_ensemble import (
        train_ensemble_regression, train_ensemble_classification
    )
    from models.explainability import run_full_explainability
    
    output_dir = os.path.join(PROJECT_DIR, "results/metrics")
    fig_dir = os.path.join(PROJECT_DIR, "results/figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Regression
    reg_models, reg_metrics, importance, scaler, features = train_ensemble_regression(
        df, output_dir=output_dir
    )
    
    # Classification
    cls_models, cls_metrics, cls_scaler, cls_features, class_names = train_ensemble_classification(
        df, output_dir=output_dir
    )
    
    # Explainability (SHAP)
    print("\n--- Running SHAP Explainability ---")
    from models.migration_ensemble import prepare_regression_data
    X, y, y_log, feature_cols = prepare_regression_data(df)
    X_filled = X.fillna(X.median())
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_scaled = pd.DataFrame(ss.fit_transform(X_filled), columns=feature_cols, index=X_filled.index)
    
    shap_results = run_full_explainability(
        {k: v for k, v in reg_models.items() if k != "Stacking Ensemble"},
        X_scaled, y_log,
        output_dir=fig_dir
    )
    
    return reg_models, reg_metrics, cls_models, cls_metrics, importance


# ============================================================================
# CELL 6: Agent-Based Model Experiments
# ============================================================================

def run_abm_experiments():
    """Run ABM scenario experiments."""
    print("\n" + "=" * 70)
    print("PHASE 4: AGENT-BASED MODEL EXPERIMENTS")
    print("=" * 70)
    
    sys.path.insert(0, PROJECT_DIR)
    from abm.model import run_all_experiments
    from abm.visualization import plot_scenario_comparison, plot_simulation_timeseries
    
    output_dir = os.path.join(PROJECT_DIR, "results/metrics")
    fig_dir = os.path.join(PROJECT_DIR, "results/figures")
    
    # Run experiments (reduce for faster testing)
    results = run_all_experiments(
        output_dir=output_dir,
        n_households=500,  # Increase to 1000 for final run
        steps=100,
        n_reps=10,  # Increase to 50 for final run
    )
    
    # Generate visualizations
    print("\nGenerating ABM visualizations...")
    
    plot_scenario_comparison(
        results,
        os.path.join(fig_dir, "abm_scenario_comparison.png")
    )
    
    for scenario in results["scenario"].unique():
        scenario_data = results[results["scenario"] == scenario]
        safe_name = scenario.replace(".", "_").replace("-", "_")
        plot_simulation_timeseries(
            scenario_data,
            os.path.join(fig_dir, f"abm_timeseries_{safe_name}.png"),
            scenario=scenario
        )
    
    return results


# ============================================================================
# CELL 7: Full Pipeline Execution
# ============================================================================

def run_full_pipeline():
    """Execute the complete research pipeline."""
    print("╔" + "═" * 68 + "╗")
    print("║  CLIMATE MIGRATION RESEARCH — FULL PIPELINE                       ║")
    print("║  Geo-Simulating Climate Extreme-Induced Human Migration           ║")
    print("╚" + "═" * 68 + "╝")
    
    # Step 1: Load data
    df = load_master_data()
    if df is None:
        print("Cannot proceed without data. Exiting.")
        return
    
    # Step 2: Deep learning (optional, needs daily data)
    try:
        dl_results = train_deep_learning_models(df)
    except Exception as e:
        print(f"Deep learning phase skipped: {e}")
        dl_results = None
    
    # Step 3: Ensemble ML
    try:
        ml_results = train_ensemble_models(df)
    except Exception as e:
        print(f"Ensemble ML error: {e}")
        ml_results = None
    
    # Step 4: ABM
    try:
        abm_results = run_abm_experiments()
    except Exception as e:
        print(f"ABM error: {e}")
        abm_results = None
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  PIPELINE COMPLETE                                                ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nResults saved to: {os.path.join(PROJECT_DIR, 'results/')}")
    
    return df, dl_results, ml_results, abm_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_full_pipeline()
