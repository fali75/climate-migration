#!/usr/bin/env python3
"""
abm/model.py
=============
Climate Displacement Simulation Model using Mesa ABM framework.

Simulates household migration decisions under climate stress scenarios.

Features:
  - 50×50 spatial grid with heterogeneous environment cells
  - 1000 household agents with bounded rationality
  - Climate inputs from LSTM-Attention predictions
  - 4 SSP scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5)
  - 3 policy modes (reactive, proactive, maladaptive)
  - Data collectors for displacement metrics
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import json
import os

try:
    from mesa import Model
    from mesa.space import MultiGrid
    from mesa.time import RandomActivation
    from mesa.datacollection import DataCollector
    MESA_AVAILABLE = True
except ImportError:
    MESA_AVAILABLE = False
    print("Mesa not installed. Using standalone simulation mode.")

from abm.agents import HouseholdAgent, GovernmentAgent, EnvironmentCell, MigrationStatus


# ─── SSP Climate Scenarios ──────────────────────────────────────────────────

SSP_SCENARIOS = {
    "SSP1-2.6": {
        "name": "Sustainability",
        "risk_trajectory": lambda t: 0.3 + 0.002 * t,  # Slow, plateauing
        "description": "Low challenges; strong mitigation",
    },
    "SSP2-4.5": {
        "name": "Middle of the Road",
        "risk_trajectory": lambda t: 0.3 + 0.005 * t,  # Moderate increase
        "description": "Medium challenges; moderate mitigation",
    },
    "SSP3-7.0": {
        "name": "Regional Rivalry",
        "risk_trajectory": lambda t: 0.3 + 0.008 * t + 0.0001 * t**2,  # Accelerating
        "description": "High challenges; weak mitigation",
    },
    "SSP5-8.5": {
        "name": "Fossil-fueled Development",
        "risk_trajectory": lambda t: 0.3 + 0.01 * t + 0.0002 * t**2,  # Rapid increase
        "description": "Very high challenges; no mitigation",
    },
}


# ─── Model Class ─────────────────────────────────────────────────────────────

class ClimateDisplacementModel:
    """
    Main ABM simulation engine for climate-induced displacement.
    
    Parameters:
        width, height: Grid dimensions
        n_households: Number of household agents
        ssp_scenario: SSP climate scenario key
        policy_mode: Government policy mode
        climate_data: External climate predictions (optional)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        width=50,
        height=50,
        n_households=1000,
        ssp_scenario="SSP2-4.5",
        policy_mode="reactive",
        climate_data=None,
        seed=42,
    ):
        self.width = width
        self.height = height
        self.n_households = n_households
        self.ssp_scenario = ssp_scenario
        self.policy_mode = policy_mode
        self.climate_data = climate_data
        self.seed = seed
        self.step_count = 0
        self.running = True
        
        # Random generator
        self.random = np.random.RandomState(seed)
        
        # Initialize environment grid
        self.grid = {}
        self._init_environment()
        
        # Initialize agents
        self.households = []
        self.government = None
        self._init_agents()
        
        # Data collection
        self.history = defaultdict(list)
        self.migration_flows = []
        
        # Get scenario function
        self.risk_trajectory = SSP_SCENARIOS[ssp_scenario]["risk_trajectory"]
    
    def _init_environment(self):
        """Initialize the spatial environment with heterogeneous cells."""
        for x in range(self.width):
            for y in range(self.height):
                # Create environmental gradient
                # Higher risk near edges (coastal/border regions)
                edge_dist = min(x, y, self.width - x - 1, self.height - y - 1)
                edge_factor = max(0, 1 - edge_dist / (self.width / 4))
                
                base_risk = 0.2 + edge_factor * 0.3 + self.random.random() * 0.2
                resources = max(0.3, 1.0 - base_risk * 0.5 + self.random.random() * 0.2)
                
                cell = EnvironmentCell(x, y, climate_risk=base_risk, resources=resources)
                cell.elevation = 1 - edge_factor + self.random.random() * 0.3
                cell.coastal_proximity = edge_factor
                self.grid[(x, y)] = cell
    
    def _init_agents(self):
        """Initialize household and government agents."""
        # Create household agents with heterogeneous attributes
        for i in range(self.n_households):
            # Place agents preferentially in safer, resource-rich areas
            attempts = 0
            while attempts < 10:
                x = self.random.randint(0, self.width)
                y = self.random.randint(0, self.height)
                cell = self.grid[(x, y)]
                if cell.current_population < cell.carrying_capacity:
                    break
                attempts += 1
            
            # Create agent with location-influenced attributes
            income = max(0.1, cell.resource_availability * 0.5 + self.random.random() * 0.4)
            vulnerability = max(0.1, cell.climate_risk * 0.6 + self.random.random() * 0.3)
            
            agent = HouseholdAgent(self, income=income, vulnerability=vulnerability)
            agent.pos = (x, y)
            agent.origin = (x, y)
            agent.agricultural_dependency = max(0.1, cell.resource_availability * 0.6)
            
            self.households.append(agent)
            cell.current_population += 1
        
        # Create government agent
        self.government = GovernmentAgent(self, budget=1.0, capacity=0.5)
        self.government.set_policy_mode(self.policy_mode)
    
    def get_neighbors(self, pos, radius=2):
        """Get neighboring household agents within radius."""
        x, y = pos
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                for h in self.households:
                    if h.pos == (nx, ny):
                        neighbors.append(h)
        return neighbors
    
    def get_global_climate_risk(self):
        """Get current global climate risk level from scenario."""
        base_risk = self.risk_trajectory(self.step_count)
        
        # Add stochastic extreme events
        if self.random.random() < 0.1:  # 10% chance of extreme event
            extreme_factor = 1.5 + self.random.random()
        else:
            extreme_factor = 1.0
        
        return min(1.0, base_risk * extreme_factor)
    
    def step(self):
        """Execute one simulation step."""
        self.step_count += 1
        
        # 1. Update climate
        global_risk = self.get_global_climate_risk()
        scenario_mult = 1.0
        if self.ssp_scenario == "SSP5-8.5":
            scenario_mult = 1.3
        elif self.ssp_scenario == "SSP3-7.0":
            scenario_mult = 1.15
        
        for cell in self.grid.values():
            cell.update_climate(global_risk, scenario_mult)
        
        # 2. Climate shocks affect households
        for household in self.households:
            if household.pos is None:
                continue
            cell = self.grid[household.pos]
            
            if cell.climate_risk > 0.6:
                shock_severity = cell.climate_risk * self.random.random()
                household.update_after_climate_shock(shock_severity)
        
        # 3. Update social networks
        for household in self.households:
            if household.pos:
                neighbors = self.get_neighbors(household.pos, radius=2)
                household.update_social_network(neighbors)
        
        # 4. Migration decisions
        migrations_this_step = 0
        trapped_this_step = 0
        
        for household in self.households:
            if household.pos is None:
                continue
            
            cell = self.grid[household.pos]
            
            # Compute economic opportunity gap
            neighbor_incomes = [
                h.income for h in self.get_neighbors(household.pos, radius=3)
                if isinstance(h, HouseholdAgent)
            ]
            avg_neighbor_income = np.mean(neighbor_incomes) if neighbor_incomes else household.income
            economic_gap = max(0, avg_neighbor_income - household.income)
            
            # Decide
            decision = household.decide_migration(
                cell.climate_risk, economic_gap,
                migration_threshold=0.45
            )
            
            if decision == "migrate":
                # Find best destination
                dest = self._find_destination(household)
                if dest:
                    old_pos = household.pos
                    self.grid[old_pos].current_population -= 1
                    household.pos = dest
                    self.grid[dest].current_population += 1
                    household.migration_history += 1
                    household.time_at_location = 0
                    
                    # Government assistance
                    self.government.provide_assistance(household)
                    
                    self.migration_flows.append({
                        "step": self.step_count,
                        "from": old_pos,
                        "to": dest,
                        "agent_income": household.income,
                        "climate_risk_origin": self.grid[old_pos].climate_risk,
                    })
                    
                    migrations_this_step += 1
            
            elif decision == "trapped":
                trapped_this_step += 1
            
            # Agent step
            household.step()
        
        # Government step
        self.government.step()
        
        # 5. Collect data
        self._collect_data(global_risk, migrations_this_step, trapped_this_step)
    
    def _find_destination(self, household):
        """Find the best migration destination for a household."""
        best_score = -np.inf
        best_pos = None
        
        # Search in expanding radius
        for radius in [5, 10, 15]:
            x, y = household.pos
            candidates = []
            
            for dx in range(-radius, radius + 1, 2):
                for dy in range(-radius, radius + 1, 2):
                    nx = (x + dx) % self.width
                    ny = (y + dy) % self.height
                    cell = self.grid[(nx, ny)]
                    
                    if cell.is_habitable() and (nx, ny) != household.pos:
                        # Score based on safety, resources, and current population
                        score = (
                            (1 - cell.climate_risk) * 0.4 +
                            cell.resource_availability * 0.3 +
                            cell.infrastructure_quality * 0.2 +
                            (1 - cell.current_population / max(1, cell.carrying_capacity)) * 0.1
                        )
                        candidates.append(((nx, ny), score))
            
            if candidates:
                # Pick from top candidates with some randomness
                candidates.sort(key=lambda c: c[1], reverse=True)
                top = candidates[:min(5, len(candidates))]
                choice = top[self.random.randint(0, len(top))]
                return choice[0]
        
        return None
    
    def _collect_data(self, global_risk, migrations, trapped):
        """Collect step-level data for analysis."""
        statuses = [h.status for h in self.households]
        incomes = [h.income for h in self.households]
        
        self.history["step"].append(self.step_count)
        self.history["global_climate_risk"].append(global_risk)
        self.history["total_migrations"].append(migrations)
        self.history["total_trapped"].append(trapped)
        self.history["total_displaced"].append(
            sum(1 for s in statuses if s == MigrationStatus.DISPLACED or s == MigrationStatus.MIGRATING)
        )
        self.history["total_settled"].append(
            sum(1 for s in statuses if s == MigrationStatus.SETTLED)
        )
        self.history["total_considering"].append(
            sum(1 for s in statuses if s == MigrationStatus.CONSIDERING)
        )
        self.history["mean_income"].append(np.mean(incomes))
        self.history["income_inequality"].append(np.std(incomes) / max(np.mean(incomes), 0.01))
        self.history["govt_budget"].append(self.government.budget)
        self.history["govt_assisted"].append(self.government.total_assisted)
    
    def run(self, steps=100):
        """Run the simulation for a specified number of steps."""
        for _ in range(steps):
            if self.running:
                self.step()
    
    def get_results(self):
        """Get simulation results as a DataFrame."""
        return pd.DataFrame(self.history)
    
    def get_migration_flows(self):
        """Get migration flow data."""
        return pd.DataFrame(self.migration_flows) if self.migration_flows else pd.DataFrame()
    
    def get_agent_summary(self):
        """Get summary statistics about agents at current state."""
        records = []
        for h in self.households:
            records.append({
                "pos_x": h.pos[0] if h.pos else None,
                "pos_y": h.pos[1] if h.pos else None,
                "income": h.income,
                "assets": h.assets,
                "vulnerability": h.vulnerability,
                "status": h.status.value,
                "migration_history": h.migration_history,
                "cumulative_stress": h.cumulative_climate_stress,
                "agricultural_dependency": h.agricultural_dependency,
            })
        return pd.DataFrame(records)


# ─── Experiment Runner ──────────────────────────────────────────────────────

def run_scenario_experiment(
    ssp_scenario,
    policy_mode,
    n_households=1000,
    steps=100,
    n_repetitions=10,
    seed_base=42,
):
    """
    Run a scenario experiment with Monte Carlo repetitions.
    
    Returns:
        results_df: Aggregated results across repetitions
    """
    print(f"  Scenario: {ssp_scenario} | Policy: {policy_mode}")
    print(f"  Agents: {n_households} | Steps: {steps} | Repetitions: {n_repetitions}")
    
    all_results = []
    
    for rep in range(n_repetitions):
        model = ClimateDisplacementModel(
            n_households=n_households,
            ssp_scenario=ssp_scenario,
            policy_mode=policy_mode,
            seed=seed_base + rep,
        )
        
        model.run(steps=steps)
        results = model.get_results()
        results["repetition"] = rep
        results["scenario"] = ssp_scenario
        results["policy"] = policy_mode
        all_results.append(results)
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Summary statistics
    final_steps = combined[combined["step"] == steps]
    print(f"    Final displaced (mean): {final_steps['total_displaced'].mean():.0f}")
    print(f"    Final trapped (mean): {final_steps['total_trapped'].mean():.0f}")
    
    return combined


def run_all_experiments(output_dir=None, n_households=500, steps=100, n_reps=10):
    """Run all scenario × policy experiments."""
    print("=" * 60)
    print("ABM Experiment Suite")
    print("=" * 60)
    
    all_results = []
    
    scenarios = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
    policies = ["reactive", "proactive", "maladaptive"]
    
    for scenario in scenarios:
        for policy in policies:
            print(f"\n{'─'*40}")
            results = run_scenario_experiment(
                scenario, policy,
                n_households=n_households,
                steps=steps,
                n_repetitions=n_reps,
            )
            all_results.append(results)
    
    combined = pd.concat(all_results, ignore_index=True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined.to_csv(os.path.join(output_dir, "abm_all_experiments.csv"), index=False)
        print(f"\n  Results saved to: {output_dir}")
    
    return combined


if __name__ == "__main__":
    print("Running quick ABM test...")
    
    model = ClimateDisplacementModel(
        width=30, height=30,
        n_households=200,
        ssp_scenario="SSP2-4.5",
        policy_mode="reactive",
        seed=42,
    )
    
    model.run(steps=50)
    results = model.get_results()
    
    print(f"\nSimulation complete: {len(results)} steps")
    print(f"Final displaced: {results['total_displaced'].iloc[-1]}")
    print(f"Final trapped: {results['total_trapped'].iloc[-1]}")
    print(f"Total migrations: {results['total_migrations'].sum()}")
