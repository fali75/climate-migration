#!/usr/bin/env python3
"""
abm/agents.py
=============
Agent definitions for the Climate-Induced Migration ABM.

Agent types:
  1. HouseholdAgent: Core decision-making unit (migration/stay)
  2. GovernmentAgent: Policy response simulator
  3. EnvironmentCell: Grid cell with climate and resource properties

Decision framework: Aspirations-Capabilities model (de Haas, 2021)
  - Migration = f(Aspiration × Capability)
  - Aspiration = f(climate_risk, relative_deprivation, information)
  - Capability = f(income, assets, social_network, health)
"""

import numpy as np
from enum import Enum

class Agent:
    def __init__(self, model):
        self.unique_id = id(self)
        self.model = model
        self.pos = None


class MigrationStatus(Enum):
    SETTLED = "settled"
    CONSIDERING = "considering"
    MIGRATING = "migrating"
    DISPLACED = "displaced"
    RETURNED = "returned"
    TRAPPED = "trapped"


class HouseholdAgent(Agent):
    """
    Household agent representing a family unit making migration decisions.
    
    Attributes:
        income: Annual household income (normalized 0-1)
        assets: Accumulated assets/savings (normalized 0-1)
        vulnerability: Climate vulnerability score (0-1, higher = more vulnerable)
        social_network_strength: Strength of local social ties (0-1)
        migration_network: Number of connections who have migrated (0-1)
        education_level: Education attainment (0-1)
        household_size: Number of people in household
        agricultural_dependency: Fraction of income from agriculture (0-1)
        health_status: Overall health (0-1)
        risk_perception: Subjective risk perception (0-1)
        migration_history: Number of past migrations
        time_at_location: Steps spent at current location
        status: Current migration status
    """
    
    def __init__(self, model, income=None, vulnerability=None):
        super().__init__(model)
        
        # Initialize with heterogeneous attributes
        rng = model.random if hasattr(model, 'random') else np.random
        
        # Economic attributes
        self.income = income or max(0.05, min(0.95, rng.random() * 0.6 + 0.2))
        self.assets = max(0.0, self.income * rng.random() * 1.5)
        self.savings_rate = 0.05 + rng.random() * 0.15
        
        # Vulnerability attributes
        self.vulnerability = vulnerability or max(0.0, min(1.0, rng.random()))
        self.agricultural_dependency = max(0.0, rng.random() * 0.8)
        self.health_status = max(0.3, min(1.0, 0.7 + rng.random() * 0.3))
        
        # Social attributes
        self.social_network_strength = max(0.1, rng.random())
        self.migration_network = 0.0  # Updated based on neighbors
        self.education_level = max(0.1, rng.random() * 0.8)
        self.household_size = max(1, int(rng.random() * 6 + 1))
        
        # Cognitive/behavioral attributes
        self.risk_perception = max(0.1, min(0.9, rng.random()))
        self.risk_tolerance = max(0.1, min(0.9, rng.random()))
        self.information_access = max(0.1, rng.random() * 0.7 + 0.1)
        
        # Status tracking
        self.status = MigrationStatus.SETTLED
        self.migration_history = 0
        self.time_at_location = 0
        self.destination = None
        self.origin = None
        self.cumulative_climate_stress = 0.0
        
        # Economic tracking
        self.income_history = []
        self.consumption = self.income * (1 - self.savings_rate)
    
    def compute_aspiration(self, climate_risk, economic_opportunity_gap):
        """
        Compute migration aspiration based on push/pull factors.
        
        Aspiration increases with:
          - Climate risk exposure
          - Economic deprivation relative to others
          - Migration network information
          - Past exposure to shocks
        
        Aspiration decreases with:
          - Strong local social ties
          - Recent arrival (low time at location)
          - Satisfaction with current conditions
        """
        # Push factors (increase aspiration)
        climate_push = (
            climate_risk * self.vulnerability * 
            self.agricultural_dependency * self.risk_perception
        )
        
        economic_push = max(0, economic_opportunity_gap * (1 - self.income))
        
        network_pull = self.migration_network * self.information_access * 0.5
        
        cumulative_stress = min(1.0, self.cumulative_climate_stress / 5.0)
        
        # Pull-back factors (decrease aspiration)
        social_anchor = self.social_network_strength * 0.3
        inertia = max(0, 0.2 - self.migration_history * 0.05)
        
        # Combined aspiration
        aspiration = (
            climate_push * 0.35 +
            economic_push * 0.25 +
            network_pull * 0.15 +
            cumulative_stress * 0.15 -
            social_anchor * 0.5 -
            inertia * 0.5
        )
        
        return max(0.0, min(1.0, aspiration))
    
    def compute_capability(self):
        """
        Compute migration capability (ability to move).
        
        Capability increases with:
          - Higher income and assets
          - Better education
          - Stronger migration network
          - Better health
        
        Capability decreases with:
          - Large household size
          - Low assets (poverty trap)
          - Poor health
        """
        financial = (self.income * 0.4 + self.assets * 0.6)
        human_capital = (self.education_level * 0.5 + self.health_status * 0.5)
        network_support = self.migration_network * 0.3
        
        # Household size burden (larger families harder to move)
        size_burden = max(0, 1.0 - (self.household_size - 1) * 0.1)
        
        capability = (
            financial * 0.40 +
            human_capital * 0.25 +
            network_support * 0.15 +
            size_burden * 0.20
        )
        
        return max(0.0, min(1.0, capability))
    
    def decide_migration(self, climate_risk, economic_gap, migration_threshold=0.5):
        """
        Core migration decision using Aspirations-Capabilities framework.
        
        Migration occurs when:
          Aspiration > threshold AND Capability > threshold
        
        Trapped populations:
          High aspiration but low capability → "involuntary immobility"
        """
        aspiration = self.compute_aspiration(climate_risk, economic_gap)
        capability = self.compute_capability()
        
        migration_score = aspiration * capability
        
        # Decision logic
        if aspiration > migration_threshold and capability > migration_threshold:
            self.status = MigrationStatus.MIGRATING
            return "migrate"
        elif aspiration > migration_threshold and capability <= migration_threshold * 0.6:
            self.status = MigrationStatus.TRAPPED
            return "trapped"
        elif aspiration > migration_threshold * 0.7:
            self.status = MigrationStatus.CONSIDERING
            return "considering"
        else:
            self.status = MigrationStatus.SETTLED
            return "stay"
    
    def update_after_climate_shock(self, shock_severity):
        """Update agent state after a climate shock event."""
        # Asset loss proportional to vulnerability and shock severity
        asset_loss = shock_severity * self.vulnerability * self.agricultural_dependency
        self.assets = max(0, self.assets - asset_loss * 0.3)
        
        # Income reduction
        income_loss = shock_severity * self.agricultural_dependency * 0.4
        self.income = max(0.05, self.income - income_loss)
        
        # Health impact
        health_loss = shock_severity * (1 - self.health_status) * 0.1
        self.health_status = max(0.1, self.health_status - health_loss)
        
        # Increase risk perception
        self.risk_perception = min(1.0, self.risk_perception + shock_severity * 0.2)
        
        # Accumulate climate stress
        self.cumulative_climate_stress += shock_severity
    
    def update_social_network(self, neighbors):
        """Update social network metrics based on neighboring agents."""
        if not neighbors:
            return
        
        # Count migrated neighbors
        migrated = sum(1 for n in neighbors if isinstance(n, HouseholdAgent) and 
                      n.status in [MigrationStatus.MIGRATING, MigrationStatus.DISPLACED])
        
        self.migration_network = migrated / max(1, len(neighbors))
        
        # Social network weakens as neighbors migrate
        remaining = sum(1 for n in neighbors if isinstance(n, HouseholdAgent) and
                       n.status == MigrationStatus.SETTLED)
        self.social_network_strength = remaining / max(1, len(neighbors))
    
    def step(self):
        """Execute one time step for the household."""
        self.time_at_location += 1
        
        # Save income to history
        self.income_history.append(self.income)
        
        # Slow asset accumulation
        self.assets = min(2.0, self.assets + self.income * self.savings_rate)
        
        # Update consumption
        self.consumption = self.income * (1 - self.savings_rate)


class GovernmentAgent(Agent):
    """
    Government agent that implements policy responses to climate displacement.
    
    Policies:
      - Evacuation orders (reactive)
      - Relocation assistance (proactive)
      - Agricultural subsidies (adaptation)
      - Infrastructure investment (resilience)
    """
    
    def __init__(self, model, budget=1.0, capacity=0.5):
        super().__init__(model)
        self.budget = budget
        self.capacity = capacity  # Institutional capacity (0-1)
        self.policy_mode = "reactive"  # reactive, proactive, maladaptive
        
        # Policy levers
        self.evacuation_threshold = 0.7
        self.assistance_per_household = 0.1
        self.infrastructure_investment = 0.0
        self.early_warning_coverage = 0.3
        
        # Tracking
        self.total_assisted = 0
        self.total_spent = 0
    
    def set_policy_mode(self, mode):
        """Set government policy mode for scenario analysis."""
        self.policy_mode = mode
        
        if mode == "proactive":
            self.evacuation_threshold = 0.5
            self.assistance_per_household = 0.15
            self.infrastructure_investment = 0.3
            self.early_warning_coverage = 0.7
        elif mode == "reactive":
            self.evacuation_threshold = 0.7
            self.assistance_per_household = 0.1
            self.infrastructure_investment = 0.1
            self.early_warning_coverage = 0.3
        elif mode == "maladaptive":
            self.evacuation_threshold = 0.9
            self.assistance_per_household = 0.05
            self.infrastructure_investment = 0.0
            self.early_warning_coverage = 0.1
    
    def provide_assistance(self, household):
        """Provide assistance to a displacing household."""
        if self.budget > self.assistance_per_household:
            household.assets += self.assistance_per_household * self.capacity
            self.budget -= self.assistance_per_household
            self.total_assisted += 1
            self.total_spent += self.assistance_per_household
            return True
        return False
    
    def step(self):
        """Government step: replenish budget, adjust policies."""
        # Partial budget replenishment per step
        self.budget = min(1.0, self.budget + 0.05)


class EnvironmentCell:
    """
    Represents a grid cell in the spatial environment.
    
    Properties:
        climate_risk: Current climate risk level (0-1)
        resource_availability: Available resources for livelihoods (0-1)
        infrastructure_quality: Quality of local infrastructure (0-1)
        carrying_capacity: Maximum population the cell can support
        elevation: Relative elevation (affects flood risk)
        coastal_proximity: Distance to coast (affects storm/flood risk)
    """
    
    def __init__(self, x, y, climate_risk=0.3, resources=0.7):
        self.x = x
        self.y = y
        self.climate_risk = climate_risk
        self.resource_availability = resources
        self.infrastructure_quality = 0.5
        self.carrying_capacity = 10
        self.current_population = 0
        self.elevation = np.random.random()
        self.coastal_proximity = np.random.random()
        
        # Historical tracking
        self.risk_history = []
        self.population_history = []
    
    def update_climate(self, global_risk_factor, scenario_multiplier=1.0):
        """Update cell climate risk based on global conditions."""
        # Base risk influenced by elevation and coastal proximity
        base_vulnerability = (1 - self.elevation) * 0.3 + self.coastal_proximity * 0.3 + 0.4
        
        # Apply global risk with local modulation
        noise = np.random.normal(0, 0.05)
        self.climate_risk = max(0, min(1,
            global_risk_factor * base_vulnerability * scenario_multiplier + noise
        ))
        
        # Resources decline with climate risk
        self.resource_availability = max(0.1, min(1.0,
            self.resource_availability - self.climate_risk * 0.05 + 0.02
        ))
        
        # Track history
        self.risk_history.append(self.climate_risk)
        self.population_history.append(self.current_population)
    
    def is_habitable(self):
        """Check if cell is still habitable."""
        return (
            self.climate_risk < 0.9 and
            self.resource_availability > 0.1 and
            self.current_population < self.carrying_capacity
        )
