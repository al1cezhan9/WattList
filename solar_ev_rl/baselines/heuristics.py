"""
Baseline Heuristic Policies for EV Charging
============================================

These baselines provide comparison points for the RL agent.
They don't learn - they follow fixed rules.

1. SolarFirstGreedy: Always prefer solar, only use grid if SoC critically low
2. ConservativeDeadline: Estimate deadline and charge aggressively early
3. EmpiricalSurvival: Track historical departure times, adapt charging rate

These baselines highlight what the RL agent needs to learn:
- When is it worth paying for grid power?
- How to handle unknown departure times?
"""

import numpy as np
from typing import Dict, Optional, List
from collections import deque


class BaseHeuristic:
    """Base class for heuristic policies."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        env_cfg = self.config.get('environment', {})
        
        # Common parameters
        self.target_soc = env_cfg.get('target_soc', 0.90)
        self.max_charging_power_kw = env_cfg.get('max_charging_power_kw', 7.4)
        self.battery_capacity_kwh = env_cfg.get('battery_capacity_kwh', 75.0)
        self.charging_efficiency = env_cfg.get('charging_efficiency', 0.90)
        self.timestep_hours = env_cfg.get('timestep_minutes', 15) / 60.0
        
    def select_action(self, observation: np.ndarray) -> int:
        """Select action given observation. Override in subclass."""
        raise NotImplementedError
    
    def reset(self):
        """Reset any internal state. Override if needed."""
        pass
    
    def _parse_observation(self, obs: np.ndarray) -> Dict:
        """Parse observation vector into named fields."""
        return {
            'soc': obs[0],
            'time_normalized': obs[1],
            'current_pv': obs[2],
            'pv_forecast_mean': obs[3:7],
            'pv_forecast_std': obs[7:11],
            'price_normalized': obs[11],
        }


class SolarFirstGreedy(BaseHeuristic):
    """
    Solar-First Greedy Heuristic
    
    Strategy: Always use solar-only unless SoC is critically low.
    
    This baseline is naive about departure uncertainty.
    It will often undercharge because it doesn't anticipate early departures.
    """
    
    def __init__(self, config: Optional[Dict] = None, critical_soc: float = 0.3):
        super().__init__(config)
        self.critical_soc = critical_soc  # Below this, use grid
        self.name = "SolarFirstGreedy"
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Action selection:
        - If SoC < critical threshold: use solar + grid
        - Otherwise: use solar only
        """
        state = self._parse_observation(observation)
        
        if state['soc'] < self.critical_soc:
            return 1  # Solar + Grid (desperate)
        else:
            return 0  # Solar only (save money)


class ConservativeDeadline(BaseHeuristic):
    """
    Conservative Deadline Heuristic
    
    Strategy: Assume a conservative (early) deadline and charge 
    aggressively at the start to ensure target SoC is reached.
    
    This baseline over-estimates urgency and tends to use too much
    grid power early, missing opportunities to wait for solar.
    """
    
    def __init__(
        self, 
        config: Optional[Dict] = None, 
        assumed_deadline_hours: float = 4.0,
        safety_margin: float = 0.1
    ):
        super().__init__(config)
        self.assumed_deadline_hours = assumed_deadline_hours
        self.safety_margin = safety_margin  # Extra SoC buffer
        self.name = "ConservativeDeadline"
        self.steps_elapsed = 0
    
    def reset(self):
        self.steps_elapsed = 0
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Action selection based on charging schedule to meet deadline.
        
        Calculates required charging rate to reach target SoC by
        assumed deadline, then decides if grid is needed.
        """
        state = self._parse_observation(observation)
        self.steps_elapsed += 1
        
        current_soc = state['soc']
        current_pv = state['current_pv']
        
        # Calculate SoC deficit
        target_with_margin = min(1.0, self.target_soc + self.safety_margin)
        soc_needed = max(0, target_with_margin - current_soc)
        
        # Energy needed (kWh)
        energy_needed = soc_needed * self.battery_capacity_kwh / self.charging_efficiency
        
        # Time remaining until assumed deadline
        hours_elapsed = self.steps_elapsed * self.timestep_hours
        time_remaining = max(0.5, self.assumed_deadline_hours - hours_elapsed)
        
        # Required charging rate to meet deadline
        required_rate = energy_needed / time_remaining
        
        # If solar alone is insufficient, use grid
        if current_pv < required_rate:
            return 1  # Solar + Grid
        else:
            return 0  # Solar only


class EmpiricalSurvival(BaseHeuristic):
    """
    Empirical Survival-Based Heuristic
    
    Strategy: Track historical departure times and estimate survival
    probability. Charge more aggressively when survival probability
    drops below threshold.
    
    This is the most sophisticated baseline - it learns a simple
    empirical model of departure times (but not as flexible as RL).
    
    Key limitation: Uses only marginal survival, doesn't condition
    on other observations or adapt to non-stationarity.
    """
    
    def __init__(
        self, 
        config: Optional[Dict] = None,
        history_size: int = 100,
        urgency_threshold: float = 0.5
    ):
        super().__init__(config)
        self.history_size = history_size
        self.urgency_threshold = urgency_threshold
        self.name = "EmpiricalSurvival"
        
        # Track departure times from past episodes
        self.departure_history: deque = deque(maxlen=history_size)
        
        # Current episode tracking
        self.current_episode_length = 0
        self.episode_active = True
        
        # Bootstrap with some initial data (assume moderate departure times)
        for _ in range(10):
            self.departure_history.append(np.random.randint(8, 40))
    
    def reset(self):
        """Reset for new episode."""
        # Record previous episode length if we have one
        if hasattr(self, 'current_episode_length') and self.current_episode_length > 0:
            self.departure_history.append(self.current_episode_length)
        
        self.current_episode_length = 0
        self.episode_active = True
    
    def _estimate_survival_probability(self, step: int) -> float:
        """
        Estimate P(still connected at step t | connected until now).
        Uses Kaplan-Meier-like survival estimate.
        """
        if len(self.departure_history) == 0:
            return 0.5
        
        # Count episodes that lasted at least this long
        survived = sum(1 for t in self.departure_history if t > step)
        total = len(self.departure_history)
        
        return survived / total
    
    def _get_urgency(self, state: Dict) -> float:
        """
        Calculate urgency based on:
        1. Current SoC deficit
        2. Estimated survival probability
        """
        soc_deficit = max(0, self.target_soc - state['soc'])
        survival_prob = self._estimate_survival_probability(self.current_episode_length)
        
        # Urgency increases as:
        # - SoC deficit is larger
        # - Survival probability drops
        urgency = soc_deficit * (1 - survival_prob)
        
        return urgency
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Action selection based on empirical survival estimates.
        
        Uses grid power when urgency (combining SoC deficit and
        estimated departure risk) exceeds threshold.
        """
        state = self._parse_observation(observation)
        self.current_episode_length += 1
        
        current_pv = state['current_pv']
        urgency = self._get_urgency(state)
        survival = self._estimate_survival_probability(self.current_episode_length)
        
        # Decision logic:
        # 1. If survival probability is high and SoC not critical: wait for solar
        # 2. If urgency is high: use grid
        # 3. If solar is abundant: use it regardless
        
        # Always fully charged - just use solar
        if state['soc'] >= self.target_soc:
            return 0
        
        # Solar is abundant - use it
        if current_pv > self.max_charging_power_kw * 0.8:
            return 0
        
        # Check urgency against threshold
        if urgency > self.urgency_threshold:
            return 1  # Grid needed
        
        # Low survival probability - be more aggressive
        if survival < 0.3:
            return 1  # Grid needed
        
        # Default: wait for solar
        return 0


def evaluate_baseline(
    baseline: BaseHeuristic,
    env,
    n_episodes: int = 100,
    seed: int = 42
) -> Dict:
    """
    Evaluate a baseline policy on the environment.
    
    Returns:
        Dictionary with metrics: mean_cost, mean_soc, undercharge_rate
    """
    np.random.seed(seed)
    
    episode_costs = []
    final_socs = []
    undercharge_count = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        baseline.reset()
        done = False
        episode_cost = 0
        
        while not done:
            action = baseline.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_cost += info.get('grid_cost', 0)
            done = terminated or truncated
        
        episode_costs.append(episode_cost)
        final_socs.append(info['soc'])
        
        if info['soc'] < env.target_soc:
            undercharge_count += 1
    
    return {
        'name': baseline.name,
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_soc': np.mean(final_socs),
        'std_soc': np.std(final_socs),
        'undercharge_rate': undercharge_count / n_episodes,
    }


def compare_baselines(env, n_episodes: int = 100, seed: int = 42):
    """Compare all baseline policies."""
    print("=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    
    baselines = [
        SolarFirstGreedy(),
        ConservativeDeadline(),
        EmpiricalSurvival(),
    ]
    
    results = []
    for baseline in baselines:
        result = evaluate_baseline(baseline, env, n_episodes, seed)
        results.append(result)
        
        print(f"\n{result['name']}:")
        print(f"  Mean Cost: ${result['mean_cost']:.2f} ± ${result['std_cost']:.2f}")
        print(f"  Mean SoC: {result['mean_soc']:.1%} ± {result['std_soc']:.1%}")
        print(f"  Undercharge Rate: {result['undercharge_rate']:.1%}")
    
    return results


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '..')
    from envs.ev_charging_env import EVChargingEnv
    
    env = EVChargingEnv()
    compare_baselines(env, n_episodes=50)
