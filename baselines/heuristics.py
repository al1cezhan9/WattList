"""
Baseline heuristic policies for EV charging.

These baselines provide comparison points for the learned RL policy.
"""

import numpy as np
from typing import Dict, Optional


class SolarFirstGreedy:
    """
    Greedy baseline: Always use solar when available, grid otherwise.
    No planning, just immediate optimization.
    """
    
    def __init__(self, max_charging_power_kw: float = 7.4):
        self.max_charging_power_kw = max_charging_power_kw
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action from observation.
        
        Observation: [soc, time_elapsed, pv_current, pv_forecast_mean (4), pv_forecast_std (4), grid_price]
        """
        pv_current = observation[2]
        
        # If solar available, use solar only; otherwise use grid only
        if pv_current > 0.1:  # Threshold for solar availability
            return 0  # Solar only
        else:
            return 1  # Grid only


class ConservativeDeadline:
    """
    Conservative baseline: Always charge at max power to ensure target SoC.
    Assumes worst-case departure time.
    """
    
    def __init__(self, target_soc: float = 0.9, max_charging_power_kw: float = 7.4):
        self.target_soc = target_soc
        self.max_charging_power_kw = max_charging_power_kw
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Always charge at max power to meet deadline.
        Never stops charging aggressively, even after reaching target.
        """
        # Always charge at max power (grid only) to ensure target SoC
        # Don't stop even after reaching target - maintain buffer
        return 1  # Grid only


