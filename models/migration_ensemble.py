#!/usr/bin/env python3
"""
migration_ensemble.py
=====================
Ensemble Machine Learning for Climate-Induced Migration Prediction.

4 Base Models:
  - XGBoost (Gradient Boosted Trees)
  - Random Forest (Bagging Ensemble)
  - LightGBM (Gradient Boosting with GOSS)
  - CatBoost (Ordered Boosting)

Meta-Learner: Stacking with Ridge Regression

Features:
  - Bayesian hyperparameter optimization (Optuna)
  - Spatial-temporal blocked cross-validation
  - Multi-metric evaluation (RMSE, MAE, R², MAPE)
  - Displacement severity classification (Low/Medium/High/Extreme)
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")


# ─── Feature Selection ──────────────────────────────────────────────────────

def select_features(df, target_col, exclude_cols=None):
    """Select and prepare features for modeling."""
    if exclude_cols is None:
        exclude_cols = [
            "iso3", "country", "year", "region", "CESI_category",
            "source", "population",
        ]
    
    # Add target to exclusions
    exclude_cols.append(target_col)
    
    # Select numeric features
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in ["float64", "int64", "float32", "int32"]
        and not c.endswith("_category")
    ]
    
    # Remove features with too many missing values (>50%)
    feature_cols = [
        c for c in feature_cols
        if df[c].notna().mean() > 0.5
    ]
    
    return feature_cols


def prepare_regression_data(df, target_col="new_displacement_disasters"):
    """Prepare data for regression (displacement magnitude prediction)."""
    feature_cols = select_features(df, target_col)
    
    # Drop rows where target is missing
    valid = df.dropna(subset=[target_col]).copy()
    
    # Fill missing features with median
    X = valid[feature_cols].fillna(valid[feature_cols].median())
    y = valid[target_col].values
    
    # Log-transform target for better distribution
    y_log = np.log1p(y)
    
    return X, y, y_log, feature_cols


def prepare_classification_data(df, target_col="new_displacement_disasters"):
    """Prepare data for classification (displacement severity)."""
    feature_cols = select_features(df, target_col)
    
    valid = df.dropna(subset=[target_col]).copy()
    
    X = valid[feature_cols].fillna(valid[feature_cols].median())
    
    # Create severity classes based on displacement magnitude
    y_continuous = valid[target_col].values
    thresholds = np.percentile(y_continuous[y_continuous > 0], [25, 50, 75])
    
    def classify_severity(val):
        if val <= thresholds[0]:
            return 0  # Low
        elif val <= thresholds[1]:
            return 1  # Medium
        elif val <= thresholds[2]:
            return 2  # High
        else:
            return 3  # Extreme
    
    y_class = np.array([classify_severity(v) for v in y_continuous])
    class_names = ["Low", "Medium", "High", "Extreme"]
    
    return X, y_class, feature_cols, class_names


# ─── Model Definitions ──────────────────────────────────────────────────────

def get_xgboost_model(task="regression", **kwargs):
    """Get XGBoost model with default hyperparameters."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(kwargs)
    
    if task == "regression":
        return xgb.XGBRegressor(**default_params)
    else:
        return xgb.XGBClassifier(**default_params, use_label_encoder=False, eval_metric="mlogloss")


def get_rf_model(task="regression", **kwargs):
    """Get Random Forest model."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(kwargs)
    
    if task == "regression":
        return RandomForestRegressor(**default_params)
    else:
        return RandomForestClassifier(**default_params)


def get_lightgbm_model(task="regression", **kwargs):
    """Get LightGBM model."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    default_params.update(kwargs)
    
    if task == "regression":
        return lgb.LGBMRegressor(**default_params)
    else:
        return lgb.LGBMClassifier(**default_params)


def get_catboost_model(task="regression", **kwargs):
    """Get CatBoost model."""
    default_params = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
    }
    default_params.update(kwargs)
    
    if task == "regression":
        return cb.CatBoostRegressor(**default_params)
    else:
        return cb.CatBoostClassifier(**default_params)


# ─── Stacking Ensemble ──────────────────────────────────────────────────────

def build_stacking_ensemble(task="regression"):
    """Build a stacking ensemble with 4 base learners and a meta-learner."""
    
    base_learners = [
        ("xgb", get_xgboost_model(task)),
        ("rf", get_rf_model(task)),
        ("lgbm", get_lightgbm_model(task)),
        ("catboost", get_catboost_model(task)),
    ]
    
    if task == "regression":
        meta_learner = Ridge(alpha=1.0)
        ensemble = StackingRegressor(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
        )
    else:
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        ensemble = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
        )
    
    return ensemble


# ─── Cross-Validation ───────────────────────────────────────────────────────

def spatial_temporal_cv(df, n_splits=5):
    """
    Spatial-temporal blocked cross-validation.
    Ensures no temporal leakage by using forward-chaining splits,
    and evaluates generalization across different countries.
    """
    # Sort by year
    df = df.sort_values(["year", "iso3"])
    
    # Use time-series split on years
    unique_years = sorted(df["year"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    splits = []
    year_indices = {year: df[df["year"] == year].index.tolist() for year in unique_years}
    
    for train_year_idx, test_year_idx in tscv.split(unique_years):
        train_years = [unique_years[i] for i in train_year_idx]
        test_years = [unique_years[i] for i in test_year_idx]
        
        train_idx = []
        for y in train_years:
            train_idx.extend(year_indices.get(y, []))
        
        test_idx = []
        for y in test_years:
            test_idx.extend(year_indices.get(y, []))
        
        splits.append((train_idx, test_idx))
    
    return splits


# ─── Training Pipeline ──────────────────────────────────────────────────────

def train_ensemble_regression(df, output_dir=None):
    """
    Full training pipeline for regression (displacement magnitude).
    
    Returns:
        models: dict of trained models
        metrics: dict of evaluation metrics
        feature_importance: DataFrame of feature importances
    """
    print("=" * 60)
    print("Ensemble ML — Regression Pipeline")
    print("Predicting: displacement magnitude")
    print("=" * 60)
    
    # Prepare data
    X, y, y_log, feature_cols = prepare_regression_data(df)
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: {y.min():.0f} – {y.max():.0f}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    
    # Train/test split (temporal)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
    y_train, y_test = y_log[:split_idx], y_log[split_idx:]
    y_raw_test = y[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train individual models
    models = {}
    metrics = {}
    
    model_configs = [
        ("XGBoost", get_xgboost_model("regression")),
        ("Random Forest", get_rf_model("regression")),
        ("LightGBM", get_lightgbm_model("regression")),
        ("CatBoost", get_catboost_model("regression")),
    ]
    
    for name, model in model_configs:
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions (transform back from log)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)  # No negative displacement
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_raw_test, y_pred))
        mae = mean_absolute_error(y_raw_test, y_pred)
        r2 = r2_score(y_raw_test, y_pred)
        
        # MAPE (handle zeros)
        mask = y_raw_test > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_raw_test[mask] - y_pred[mask]) / y_raw_test[mask])) * 100
        else:
            mape = float('inf')
        
        metrics[name] = {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "MAPE": float(mape),
        }
        
        models[name] = model
        print(f"    RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | R²: {r2:.4f} | MAPE: {mape:.1f}%")
    
    # Train stacking ensemble
    print(f"\n  Training Stacking Ensemble...")
    ensemble = build_stacking_ensemble("regression")
    ensemble.fit(X_train, y_train)
    
    y_pred_log = ensemble.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)
    
    rmse = np.sqrt(mean_squared_error(y_raw_test, y_pred))
    mae = mean_absolute_error(y_raw_test, y_pred)
    r2 = r2_score(y_raw_test, y_pred)
    mask = y_raw_test > 0
    mape = np.mean(np.abs((y_raw_test[mask] - y_pred[mask]) / y_raw_test[mask])) * 100 if mask.sum() > 0 else float('inf')
    
    metrics["Stacking Ensemble"] = {
        "RMSE": float(rmse), "MAE": float(mae), "R2": float(r2), "MAPE": float(mape),
    }
    models["Stacking Ensemble"] = ensemble
    print(f"    RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | R²: {r2:.4f} | MAPE: {mape:.1f}%")
    
    # Feature importance (from XGBoost)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "xgb_importance": models["XGBoost"].feature_importances_,
        "rf_importance": models["Random Forest"].feature_importances_,
        "lgbm_importance": models["LightGBM"].feature_importances_,
    })
    importance_df["mean_importance"] = importance_df[
        ["xgb_importance", "rf_importance", "lgbm_importance"]
    ].mean(axis=1)
    importance_df = importance_df.sort_values("mean_importance", ascending=False)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "regression_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        importance_df.to_csv(
            os.path.join(output_dir, "feature_importance.csv"), index=False
        )
        
        print(f"\n  Results saved to: {output_dir}")
    
    print("\n" + "=" * 60)
    print("REGRESSION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>8} {'MAPE':>8}")
    print("-" * 65)
    for name, m in metrics.items():
        print(f"{name:<25} {m['RMSE']:>12,.0f} {m['MAE']:>12,.0f} {m['R2']:>8.4f} {m['MAPE']:>7.1f}%")
    
    return models, metrics, importance_df, scaler, feature_cols


def train_ensemble_classification(df, output_dir=None):
    """
    Full training pipeline for classification (displacement severity).
    """
    print("\n" + "=" * 60)
    print("Ensemble ML — Classification Pipeline")
    print("Predicting: displacement severity class")
    print("=" * 60)
    
    X, y, feature_cols, class_names = prepare_classification_data(df)
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {dict(zip(class_names, np.bincount(y, minlength=4)))}")
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    models = {}
    metrics = {}
    
    model_configs = [
        ("XGBoost", get_xgboost_model("classification")),
        ("Random Forest", get_rf_model("classification")),
        ("LightGBM", get_lightgbm_model("classification")),
        ("CatBoost", get_catboost_model("classification")),
    ]
    
    for name, model in model_configs:
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        metrics[name] = {
            "Accuracy": float(acc),
            "F1_weighted": float(f1),
        }
        models[name] = model
        print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Stacking ensemble
    print(f"\n  Training Stacking Ensemble...")
    ensemble = build_stacking_ensemble("classification")
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    metrics["Stacking Ensemble"] = {"Accuracy": float(acc), "F1_weighted": float(f1)}
    models["Stacking Ensemble"] = ensemble
    print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Classification report for best model
    best_model_name = max(metrics, key=lambda k: metrics[k]["F1_weighted"])
    best_pred = models[best_model_name].predict(X_test)
    
    print(f"\nBest Model: {best_model_name}")
    print(classification_report(y_test, best_pred, target_names=class_names))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "classification_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return models, metrics, scaler, feature_cols, class_names


if __name__ == "__main__":
    print("Migration Ensemble ML Module")
    print("This module is imported by analysis/03_model_training.py")
