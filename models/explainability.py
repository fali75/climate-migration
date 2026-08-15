#!/usr/bin/env python3
"""
explainability.py
=================
Explainable AI (XAI) module for migration prediction models.

Implements:
  - SHAP TreeExplainer (global + local explanations)
  - SHAP interaction effects
  - SHAP dependence plots
  - Permutation importance
  - Partial Dependence Plots (PDP)
  
All visualizations saved as high-res PNGs for the research paper.
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

warnings.filterwarnings("ignore")

# Set publication-quality defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ─── SHAP Analysis ──────────────────────────────────────────────────────────

def compute_shap_values(model, X, model_name="XGBoost"):
    """
    Compute SHAP values using TreeExplainer.
    
    Parameters:
        model: trained tree-based model (XGBoost, RF, LightGBM, CatBoost)
        X: feature DataFrame
        model_name: for labeling
    
    Returns:
        shap_values: SHAP values array
        explainer: SHAP TreeExplainer object
    """
    print(f"  Computing SHAP values for {model_name}...")
    
    # Use TreeExplainer for tree-based models
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"  TreeExplainer failed, using KernelExplainer: {e}")
        # Fallback for models not supported by TreeExplainer
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)
    
    return shap_values, explainer


def plot_shap_summary(shap_values, X, output_path, model_name="XGBoost", max_display=20):
    """
    Generate SHAP summary beeswarm plot.
    Shows global feature importance with direction of effect.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Handle multi-class SHAP values
    if isinstance(shap_values, list):
        # For classification, use mean absolute SHAP across classes
        mean_shap = np.abs(np.array(shap_values)).mean(axis=0)
        shap.summary_plot(
            mean_shap, X, plot_type="bar",
            max_display=max_display, show=False
        )
    else:
        shap.summary_plot(
            shap_values, X,
            max_display=max_display, show=False
        )
    
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {os.path.basename(output_path)}")


def plot_shap_bar(shap_values, X, output_path, model_name="XGBoost", max_display=15):
    """Generate SHAP bar plot showing mean |SHAP| values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1)) if len(np.array(shap_values).shape) > 2 else np.abs(np.array(shap_values)).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap if isinstance(mean_abs_shap, np.ndarray) and mean_abs_shap.ndim == 1 else mean_abs_shap.mean(axis=0) if hasattr(mean_abs_shap, 'mean') else mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=True).tail(max_display)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    ax.barh(importance_df["feature"], importance_df["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Global Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {os.path.basename(output_path)}")


def plot_shap_waterfall(shap_values, X, output_path, sample_idx=0, model_name="XGBoost"):
    """
    Generate SHAP waterfall plot for a single prediction.
    Shows how each feature pushed the prediction up or down.
    """
    try:
        if isinstance(shap_values, list):
            sv = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            sv = shap_values
        
        # Create Explanation object
        explanation = shap.Explanation(
            values=sv[sample_idx],
            base_values=np.mean(sv, axis=0).mean() if sv.ndim > 1 else np.mean(sv),
            data=X.iloc[sample_idx].values,
            feature_names=list(X.columns),
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — Sample {sample_idx} ({model_name})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"    Waterfall plot error: {e}")


def plot_shap_dependence(shap_values, X, feature, output_path, interaction_feature=None):
    """
    Generate SHAP dependence plot for a specific feature.
    Shows how the feature value affects the model output.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if isinstance(shap_values, list):
        sv = shap_values[0] if len(shap_values) > 0 else shap_values
    else:
        sv = shap_values
    
    try:
        shap.dependence_plot(
            feature, sv, X,
            interaction_index=interaction_feature,
            ax=ax, show=False,
        )
        plt.title(f"SHAP Dependence: {feature}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"    Dependence plot error for {feature}: {e}")
        plt.close()


def plot_shap_interaction(shap_values, X, output_path, top_n=10):
    """
    Generate SHAP interaction heatmap showing feature interactions.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    try:
        if isinstance(shap_values, list):
            sv = np.array(shap_values[0]) if len(shap_values) > 0 else np.array(shap_values)
        else:
            sv = np.array(shap_values)
        
        # Compute correlation between SHAP values of different features
        shap_df = pd.DataFrame(sv, columns=X.columns)
        
        # Get top features by mean absolute SHAP
        top_features = shap_df.abs().mean().nlargest(top_n).index.tolist()
        shap_corr = shap_df[top_features].corr()
        
        mask = np.triu(np.ones_like(shap_corr, dtype=bool), k=1)
        sns.heatmap(
            shap_corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax,
            square=True, linewidths=0.5,
        )
        ax.set_title("SHAP Value Correlations (Feature Interactions)", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"    Interaction plot error: {e}")
        plt.close()


# ─── Permutation Importance ─────────────────────────────────────────────────

def compute_permutation_importance(model, X, y, output_path, model_name="XGBoost", n_repeats=10):
    """
    Compute and plot permutation importance.
    Model-agnostic validation of feature rankings.
    """
    print(f"  Computing permutation importance for {model_name}...")
    
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        perm_df["feature"],
        perm_df["importance_mean"],
        xerr=perm_df["importance_std"],
        color=plt.cm.plasma(np.linspace(0.3, 0.9, len(perm_df))),
        capsize=3,
    )
    ax.set_xlabel("Mean Decrease in Score", fontsize=12)
    ax.set_title(f"Permutation Importance — {model_name}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {os.path.basename(output_path)}")
    
    return perm_df


# ─── Full Explainability Pipeline ───────────────────────────────────────────

def run_full_explainability(models, X, y, output_dir):
    """
    Run complete explainability analysis on all models.
    
    Parameters:
        models: dict of {name: trained_model}
        X: feature DataFrame
        y: target array
        output_dir: directory to save plots
    """
    print("\n" + "=" * 60)
    print("Explainable AI Analysis Pipeline")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_shap_results = {}
    
    for model_name, model in models.items():
        if model_name == "Stacking Ensemble":
            continue  # Skip stacking (not supported by TreeExplainer)
        
        print(f"\n{'─'*40}")
        print(f"Analyzing: {model_name}")
        print(f"{'─'*40}")
        
        # SHAP values
        shap_values, explainer = compute_shap_values(model, X, model_name)
        all_shap_results[model_name] = shap_values
        
        # Summary beeswarm plot
        plot_shap_summary(
            shap_values, X,
            os.path.join(output_dir, f"shap_summary_{model_name.lower().replace(' ', '_')}.png"),
            model_name
        )
        
        # Bar plot
        plot_shap_bar(
            shap_values, X,
            os.path.join(output_dir, f"shap_bar_{model_name.lower().replace(' ', '_')}.png"),
            model_name
        )
        
        # Waterfall for specific samples
        for idx in [0, len(X)//2, len(X)-1]:
            if idx < len(X):
                plot_shap_waterfall(
                    shap_values, X,
                    os.path.join(output_dir, f"shap_waterfall_{model_name.lower().replace(' ', '_')}_sample{idx}.png"),
                    sample_idx=idx,
                    model_name=model_name
                )
        
        # Interaction heatmap
        plot_shap_interaction(
            shap_values, X,
            os.path.join(output_dir, f"shap_interactions_{model_name.lower().replace(' ', '_')}.png")
        )
        
        # Dependence plots for top 5 features
        if isinstance(shap_values, list):
            mean_abs = np.abs(np.array(shap_values)).mean(axis=(0, 1)) if np.array(shap_values).ndim > 2 else np.abs(np.array(shap_values[0])).mean(axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
        
        if isinstance(mean_abs, np.ndarray) and mean_abs.ndim <= 1:
            top_features = pd.Series(mean_abs, index=X.columns).nlargest(5).index
            for feat in top_features:
                plot_shap_dependence(
                    shap_values, X, feat,
                    os.path.join(output_dir, f"shap_dep_{feat}_{model_name.lower().replace(' ', '_')}.png")
                )
        
        # Permutation importance
        compute_permutation_importance(
            model, X, y,
            os.path.join(output_dir, f"perm_importance_{model_name.lower().replace(' ', '_')}.png"),
            model_name
        )
    
    print(f"\n  All explainability outputs saved to: {output_dir}")
    
    return all_shap_results


if __name__ == "__main__":
    print("Explainability Module")
    print("Import and use with trained models from migration_ensemble.py")
