#!/usr/bin/env python3
"""
analysis/04_abm_experiments.py
==============================
Standalone script for running Agent-Based Model (ABM) experiments.
Evaluates displacement outcomes under multiple SSP scenarios and policy modes.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

METRICS_DIR = BASE_DIR / "results" / "metrics"
FIG_DIR = BASE_DIR / "results" / "figures"

def main():
    print("=" * 60)
    print("Agent-Based Model (ABM) Experiments")
    print("=" * 60)
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import mesa
    except ImportError:
        print("Warning: mesa package is not installed. ABM will run in fallback standalone mode if supported.")

    from abm.model import run_all_experiments
    from abm.visualization import plot_scenario_comparison, plot_simulation_timeseries
    
    # Run the experiments
    # For a quick run, keep n_households and steps small. Adjust for full simulation.
    print("Running ABM simulations...")
    results = run_all_experiments(
        output_dir=str(METRICS_DIR),
        n_households=500,
        steps=100,
        n_reps=5  # Use 50 for publication
    )
    
    # Generate visualization for comparison
    print("Generating ABM visualizations...")
    plot_scenario_comparison(
        results,
        str(FIG_DIR / "abm_scenario_comparison.png")
    )
    
    for scenario in results["scenario"].unique():
        scenario_data = results[results["scenario"] == scenario]
        safe_name = scenario.replace(".", "_").replace("-", "_")
        plot_simulation_timeseries(
            scenario_data,
            str(FIG_DIR / f"abm_timeseries_{safe_name}.png"),
            scenario=scenario
        )
        
    print("\nABM experiments and visualizations complete.")

if __name__ == "__main__":
    main()
