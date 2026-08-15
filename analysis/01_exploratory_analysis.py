#!/usr/bin/env python3
"""
analysis/01_exploratory_analysis.py
===================================
Exploratory Data Analysis (EDA) of the Climate Migration Dataset.

Generates:
  - Descriptive statistics table
  - Correlation heatmaps
  - Time series trend plots for key variables
  - Distribution plots for CESI and Displacement
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "master_dataset.csv"
OUTPUT_DIR = BASE_DIR / "results" / "eda"

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

def load_data():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Please run data acquisition scripts first.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records with {len(df.columns)} features.")
    return df

def generate_descriptive_stats(df):
    print("\n--- Generating Descriptive Statistics ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T
    
    stats_path = OUTPUT_DIR / "descriptive_statistics.csv"
    stats.to_csv(stats_path)
    print(f"Saved descriptive statistics to {stats_path}")
    return stats

def plot_distributions(df):
    print("--- Plotting Distributions ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # CESI Distribution
    if 'CESI' in df.columns:
        sns.histplot(df['CESI'].dropna(), kde=True, ax=axes[0], color='coral')
        axes[0].set_title('Distribution of Climate Extreme Severity Index (CESI)')
        axes[0].set_xlabel('CESI')
        axes[0].set_ylabel('Frequency')
    
    # Displacement Distribution
    target = 'new_displacement_disasters'
    if target in df.columns:
        # Plot log displacement for better visualization
        log_disp = np.log1p(df[target].dropna())
        sns.histplot(log_disp, kde=True, ax=axes[1], color='teal')
        axes[1].set_title('Distribution of Disaster Displacement (Log Scale)')
        axes[1].set_xlabel('Log(Displacements + 1)')
        axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "distributions.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved distributions plot to {plot_path}")

def plot_correlation_matrix(df):
    print("--- Plotting Correlation Matrix ---")
    # Select key columns to avoid massive matrix
    key_cols = [
        'new_displacement_disasters', 'CESI', 'gdp_per_capita_usd', 
        'population_density', 'employment_agriculture_pct', 
        'conflict_intensity', 'HWDI', 'CDD', 'Rx5day'
    ]
    # Keep only columns that actually exist
    key_cols = [c for c in key_cols if c in df.columns]
    
    if len(key_cols) > 1:
        corr = df[key_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
        plt.title('Correlation Matrix of Key Variables')
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "correlation_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved correlation matrix to {plot_path}")

def plot_time_series_trends(df):
    print("--- Plotting Time Series Trends ---")
    if 'year' not in df.columns or 'new_displacement_disasters' not in df.columns:
        return
    
    # Aggregate displacement by year
    yearly_disp = df.groupby('year')['new_displacement_disasters'].sum() / 1e6  # in millions
    
    plt.figure(figsize=(10, 5))
    yearly_disp.plot(kind='bar', color='steelblue')
    plt.title('Total Disaster Displacement per Year (Study Region)')
    plt.xlabel('Year')
    plt.ylabel('Displacement (Millions)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "yearly_displacement_trend.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved time series trend to {plot_path}")

def main():
    print("=" * 60)
    print("Starting Exploratory Data Analysis (EDA)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    
    generate_descriptive_stats(df)
    plot_distributions(df)
    plot_correlation_matrix(df)
    plot_time_series_trends(df)
    
    print("\nEDA completed successfully.")

if __name__ == "__main__":
    main()
