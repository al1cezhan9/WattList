"""
Baseline heuristic policies for EV charging.

These baselines provide comparison points for the learned RL policy.
"""

import numpy as np
from typing import Dict, Optional
from collections import deque


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
        
        # If solar available, use solar only; otherwise use grid
        if pv_current > 0.1:  # Threshold for solar availability
            return 0  # Solar only
        else:
            return 1  # Solar + Grid


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
        """
        current_soc = observation[0]
        time_elapsed_norm = observation[1]
        
        # If already at target, use solar only
        if current_soc >= self.target_soc:
            pv_current = observation[2]
            return 0 if pv_current > 0.1 else 1
        
        # Otherwise, always charge at max
        return 1  # Solar + Grid


class EmpiricalSurvival:
    """
    Empirical survival-based heuristic.
    Tracks historical departure times and estimates survival probability.
    Uses this to decide when to charge aggressively.
    """
    
    def __init__(self, target_soc: float = 0.9, max_charging_power_kw: float = 7.4):
        self.target_soc = target_soc
        self.max_charging_power_kw = max_charging_power_kw
        self.departure_times = deque(maxlen=1000)  # Track last 1000 departures
        self.episode_length = 0
    
    def update_departure(self, episode_length: int):
        """Record departure time."""
        self.departure_times.append(episode_length)
    
    def estimate_survival_probability(self, current_time: int) -> float:
        """
        Estimate probability of surviving beyond current time based on history.
        """
        if len(self.departure_times) == 0:
            return 0.5  # Default if no history
        
        # Count how many episodes survived beyond current time
        survived = sum(1 for t in self.departure_times if t > current_time)
        return survived / len(self.departure_times)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Use survival probability to decide charging strategy.
        """
        current_soc = observation[0]
        time_elapsed_norm = observation[1]
        pv_current = observation[2]
        grid_price = observation[-1]
        
        # Estimate current time in steps
        current_time = int(time_elapsed_norm * 96)  # Assuming max 96 steps
        
        # Estimate survival probability
        survival_prob = self.estimate_survival_probability(current_time)
        
        # If low survival probability and low SoC, charge aggressively
        if survival_prob < 0.3 and current_soc < self.target_soc:
            return 1  # Solar + Grid
        
        # If high survival probability, can be more conservative
        if survival_prob > 0.7 and current_soc > 0.7:
            return 0 if pv_current > 0.1 else 1
        
        # Default: use solar when available
        return 0 if pv_current > 0.1 else 1
