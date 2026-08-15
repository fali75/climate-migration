#!/usr/bin/env python3
"""
abm/visualization.py
====================
Visualization module for ABM simulation results.
Generates spatial maps, flow diagrams, and time series plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import os

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "figure.dpi": 150, "savefig.dpi": 300,
})


def plot_simulation_timeseries(results_df, output_path, scenario="SSP2-4.5"):
    """Plot key simulation metrics over time."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Aggregate across repetitions
    agg = results_df.groupby("step").agg({
        "total_displaced": ["mean", "std"],
        "total_trapped": ["mean", "std"],
        "total_migrations": ["mean", "std"],
        "global_climate_risk": "mean",
        "mean_income": ["mean", "std"],
        "income_inequality": ["mean", "std"],
    })
    
    steps = agg.index
    
    # Panel 1: Displaced population
    ax = axes[0, 0]
    mean = agg[("total_displaced", "mean")]
    std = agg[("total_displaced", "std")]
    ax.plot(steps, mean, color="#e74c3c", linewidth=2)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#e74c3c")
    ax.set_ylabel("Displaced Households")
    ax.set_title("(a) Displacement Over Time")
    
    # Panel 2: Trapped population
    ax = axes[0, 1]
    mean = agg[("total_trapped", "mean")]
    std = agg[("total_trapped", "std")]
    ax.plot(steps, mean, color="#9b59b6", linewidth=2)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#9b59b6")
    ax.set_ylabel("Trapped Households")
    ax.set_title("(b) Involuntary Immobility")
    
    # Panel 3: Migrations per step
    ax = axes[1, 0]
    mean = agg[("total_migrations", "mean")]
    std = agg[("total_migrations", "std")]
    ax.plot(steps, mean, color="#3498db", linewidth=2)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#3498db")
    ax.set_ylabel("Migrations")
    ax.set_title("(c) Migration Events Per Step")
    
    # Panel 4: Climate risk trajectory
    ax = axes[1, 1]
    ax.plot(steps, agg[("global_climate_risk", "mean")], color="#e67e22", linewidth=2)
    ax.set_ylabel("Risk Level")
    ax.set_title(f"(d) Climate Risk — {scenario}")
    
    # Panel 5: Mean income
    ax = axes[2, 0]
    mean = agg[("mean_income", "mean")]
    std = agg[("mean_income", "std")]
    ax.plot(steps, mean, color="#27ae60", linewidth=2)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#27ae60")
    ax.set_ylabel("Mean Income")
    ax.set_xlabel("Simulation Step")
    ax.set_title("(e) Economic Impact")
    
    # Panel 6: Inequality
    ax = axes[2, 1]
    mean = agg[("income_inequality", "mean")]
    std = agg[("income_inequality", "std")]
    ax.plot(steps, mean, color="#c0392b", linewidth=2)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#c0392b")
    ax.set_ylabel("Gini Proxy (CV)")
    ax.set_xlabel("Simulation Step")
    ax.set_title("(f) Income Inequality")
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.suptitle(f"ABM Simulation Results — {scenario}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scenario_comparison(all_results, output_path):
    """Compare displacement outcomes across SSP scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Final displacement by scenario
    ax = axes[0]
    scenarios = all_results["scenario"].unique()
    colors = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
    
    for i, scenario in enumerate(sorted(scenarios)):
        data = all_results[all_results["scenario"] == scenario]
        final = data.groupby("step")["total_displaced"].mean()
        ax.plot(final.index, final.values, label=scenario, color=colors[i], linewidth=2)
    
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Displaced Households")
    ax.set_title("(a) Displacement by Climate Scenario")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Policy comparison
    ax = axes[1]
    policies = all_results["policy"].unique()
    policy_colors = {"reactive": "#3498db", "proactive": "#27ae60", "maladaptive": "#e74c3c"}
    
    for policy in sorted(policies):
        data = all_results[all_results["policy"] == policy]
        final = data.groupby("step")["total_displaced"].mean()
        ax.plot(final.index, final.values, label=policy.capitalize(),
                color=policy_colors.get(policy, "gray"), linewidth=2)
    
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Displaced Households")
    ax.set_title("(b) Displacement by Policy Mode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.suptitle("Scenario Comparison Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_spatial_heatmap(model, output_path, metric="climate_risk"):
    """Plot spatial heatmap of the environment grid."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    data = np.zeros((model.width, model.height))
    
    for (x, y), cell in model.grid.items():
        if metric == "climate_risk":
            data[x, y] = cell.climate_risk
        elif metric == "population":
            data[x, y] = cell.current_population
        elif metric == "resources":
            data[x, y] = cell.resource_availability
    
    im = ax.imshow(data.T, cmap="YlOrRd" if metric == "climate_risk" else "YlGn",
                   origin="lower", aspect="equal")
    plt.colorbar(im, ax=ax, label=metric.replace("_", " ").title())
    
    # Overlay agent positions
    for h in model.households:
        if h.pos:
            color = {
                "settled": "#27ae60",
                "migrating": "#e74c3c",
                "trapped": "#8e44ad",
                "considering": "#f39c12",
            }.get(h.status.value, "gray")
            ax.plot(h.pos[0], h.pos[1], "o", color=color, markersize=2, alpha=0.5)
    
    ax.set_title(f"Spatial Distribution — {metric.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
