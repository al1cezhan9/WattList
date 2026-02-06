"""
EV Charging Environment with Recurrent State and Hidden Departure Model.

Key features:
- 15-minute timesteps
- PV forecast with AR(1) correlated noise
- Hidden, non-stationary departure behavior
- Time-of-use grid pricing
- Recurrent observation space (no departure probability exposed)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import math


class EVChargingEnv(gym.Env):
    """
    Gymnasium-compatible environment for EV charging under solar uncertainty
    and unknown, non-stationary departure behavior.
    
    Observation space (7D):
    - current SoC (0-1)
    - time elapsed (normalized 0-1)
    - current PV output (kW)
    - PV forecast mean for next 4 steps (kW)
    - PV forecast std for next 4 steps (kW)
    - current grid price ($/kWh)
    
    Action space: Discrete(2)
    - 0: solar only charging
    - 1: solar + grid charging
    
    Episode ends when EV departs (hidden hazard process).
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}  # 4 steps per hour = 15 min steps
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        if config is None:
            config = {}
        
        # Environment parameters
        self.ev_capacity_kwh = config.get('ev_capacity_kwh', 75.0)
        self.charging_efficiency = config.get('charging_efficiency', 0.90)
        self.max_charging_power_kw = config.get('max_charging_power_kw', 7.4)
        self.target_soc = config.get('target_soc', 0.90)
        self.lambda_penalty = config.get('lambda_penalty', 100.0)  # Terminal penalty weight
        
        # Solar/PV parameters
        self.max_pv_output_kw = config.get('max_pv_output_kw', 10.0)
        self.pv_noise_std = config.get('pv_noise_std', 0.5)
        self.pv_ar_coef = config.get('pv_ar_coef', 0.7)  # AR(1) correlation
        self.pv_forecast_bias = config.get('pv_forecast_bias', 0.1)  # Forecast bias
        
        # Grid pricing
        self.tou_prices = config.get('tou_prices', {
            'off_peak': 0.10,  # 11 PM - 9 AM
            'mid': 0.20,       # 9 AM - 4 PM, 9 PM - 11 PM
            'peak': 0.30       # 4 PM - 9 PM
        })
        
        # Departure model parameters
        self.departure_drift_rate = config.get('departure_drift_rate', 0.01)
        self.num_user_types = config.get('num_user_types', 3)
        self.base_hazard_rates = config.get('base_hazard_rates', [0.05, 0.10, 0.15])  # Per timestep
        
        # Timestep: 15 minutes
        self.timestep_minutes = 15
        self.max_episode_steps = config.get('max_episode_steps', 96)  # 24 hours max
        
        # Observation space: [soc, time_elapsed, pv_current, pv_forecast_mean (4), pv_forecast_std (4), grid_price]
        # Total: 1 + 1 + 1 + 4 + 4 + 1 = 12D
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0] + [0.0] * 4 + [0.0] * 4 + [0.0]),
            high=np.array([1.0, 1.0, self.max_pv_output_kw] + [self.max_pv_output_kw] * 4 + [self.max_pv_output_kw] * 4 + [0.5]),
            dtype=np.float32
        )
        self.obs_dim = 12  # Store observation dimension
        
        # Action space: 2 discrete actions
        self.action_space = spaces.Discrete(2)
        
        # Internal state
        self.current_soc = 0.0
        self.time_elapsed = 0
        self.current_hour = 0.0  # Hour of day (0-24)
        self.pv_ar_state = 0.0  # AR(1) state for PV noise
        self.user_type = 0  # Hidden user type
        self.user_type_weights = np.ones(self.num_user_types) / self.num_user_types  # Non-stationary weights
        
        # Episode tracking
        self.episode_cost = 0.0
        self.total_grid_energy = 0.0
        
    def _get_pv_output(self, hour: float) -> float:
        """
        Generate PV output using bell-shaped daily curve (sinusoidal).
        Peak at noon (hour 12).
        """
        hour_normalized = hour % 24
        
        if 6 <= hour_normalized <= 18:  # Daylight hours
            angle = (hour_normalized - 12) * np.pi / 12
            pv_base = self.max_pv_output_kw * np.cos(angle)
        else:
            pv_base = 0.0
        
        # AR(1) correlated noise
        self.pv_ar_state = self.pv_ar_coef * self.pv_ar_state + np.random.normal(0, self.pv_noise_std)
        pv_output = np.clip(pv_base + self.pv_ar_state, 0.0, self.max_pv_output_kw)
        
        return pv_output
    
    def _get_pv_forecast(self, current_hour: float, horizon: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate imperfect PV forecast for next horizon steps.
        Returns (mean, std) arrays.
        """
        forecast_mean = []
        forecast_std = []
        
        for i in range(horizon):
            future_hour = (current_hour + (i + 1) * self.timestep_minutes / 60.0) % 24
            
            if 6 <= future_hour <= 18:
                angle = (future_hour - 12) * np.pi / 12
                base_forecast = self.max_pv_output_kw * np.cos(angle)
            else:
                base_forecast = 0.0
            
            # Add bias and uncertainty
            forecast_mean.append(base_forecast * (1 + self.pv_forecast_bias))
            forecast_std.append(self.pv_noise_std * (1 + abs(self.pv_ar_state)))
        
        return np.array(forecast_mean, dtype=np.float32), np.array(forecast_std, dtype=np.float32)
    
    def _get_grid_price(self, hour: float) -> float:
        """Get Time-of-Use grid price based on hour of day."""
        hour_normalized = hour % 24
        
        if 16 <= hour_normalized < 21:  # 4 PM - 9 PM
            return self.tou_prices['peak']
        elif (9 <= hour_normalized < 16) or (21 <= hour_normalized < 23):
            return self.tou_prices['mid']
        else:  # 11 PM - 9 AM
            return self.tou_prices['off_peak']
    
    def _sample_departure(self) -> bool:
        """
        Sample departure using hidden hazard-based model.
        Returns True if EV departs this timestep.
        """
        # Sample user type from non-stationary distribution
        self.user_type = np.random.choice(
            self.num_user_types,
            p=self.user_type_weights
        )
        
        # Get hazard rate for current user type
        hazard_rate = self.base_hazard_rates[self.user_type]
        
        # Increase hazard with time elapsed (more likely to leave later)
        time_factor = 1.0 + (self.time_elapsed / self.max_episode_steps) * 2.0
        adjusted_hazard = hazard_rate * time_factor
        
        # Sample departure
        return np.random.random() < adjusted_hazard
    
    def _update_user_type_distribution(self):
        """Update user type distribution (non-stationary drift)."""
        # Simple drift: shift weights slightly
        drift = np.random.normal(0, self.departure_drift_rate, size=self.num_user_types)
        self.user_type_weights += drift
        self.user_type_weights = np.clip(self.user_type_weights, 0.01, 1.0)
        self.user_type_weights /= self.user_type_weights.sum()  # Renormalize
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random initial SoC
        self.current_soc = np.random.uniform(0.2, 0.5)
        
        # Random start hour
        self.current_hour = np.random.uniform(0, 24)
        
        # Reset episode state
        self.time_elapsed = 0
        self.pv_ar_state = np.random.normal(0, self.pv_noise_std)
        self.episode_cost = 0.0
        self.total_grid_energy = 0.0
        
        # Update user type distribution (non-stationary)
        self._update_user_type_distribution()
        
        # Sample initial user type
        self.user_type = np.random.choice(
            self.num_user_types,
            p=self.user_type_weights
        )
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        pv_current = self._get_pv_output(self.current_hour)
        pv_forecast_mean, pv_forecast_std = self._get_pv_forecast(self.current_hour, horizon=4)
        grid_price = self._get_grid_price(self.current_hour)
        time_elapsed_norm = self.time_elapsed / self.max_episode_steps
        
        observation = np.array([
            self.current_soc,
            time_elapsed_norm,
            pv_current,
            *pv_forecast_mean,
            *pv_forecast_std,
            grid_price
        ], dtype=np.float32)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep (15 minutes).
        
        Args:
            action: 0 (solar only) or 1 (solar + grid)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current PV output
        pv_output = self._get_pv_output(self.current_hour)
        grid_price = self._get_grid_price(self.current_hour)
        
        # Calculate charging power based on action
        if action == 0:  # Solar only
            charging_power_kw = np.minimum(pv_output, self.max_charging_power_kw)
            grid_power_kw = 0.0
        else:  # Action 1: Solar + Grid
            charging_power_kw = self.max_charging_power_kw
            grid_power_kw = np.maximum(0.0, self.max_charging_power_kw - pv_output)
        
        # Update SoC (15 minutes = 0.25 hours)
        energy_added_kwh = charging_power_kw * 0.25 * self.charging_efficiency
        self.current_soc = np.clip(
            self.current_soc + (energy_added_kwh / self.ev_capacity_kwh),
            0.0, 1.0
        )
        
        # Calculate step reward (negative cost)
        grid_energy_kwh = grid_power_kw * 0.25
        step_cost = grid_energy_kwh * grid_price
        reward = -step_cost  # Negative cost as reward
        
        # Track costs
        self.episode_cost += step_cost
        self.total_grid_energy += grid_energy_kwh
        
        # Update time
        self.time_elapsed += 1
        self.current_hour = (self.current_hour + self.timestep_minutes / 60.0) % 24
        
        # Check for departure (hidden process)
        terminated = self._sample_departure()
        
        # Apply terminal penalty if SoC < target
        if terminated:
            soc_deficit = max(0.0, self.target_soc - self.current_soc)
            terminal_penalty = -self.lambda_penalty * soc_deficit
            reward += terminal_penalty
        
        # Check truncation (max episode length)
        truncated = (self.time_elapsed >= self.max_episode_steps)
        
        # Get next observation
        next_observation = self._get_observation()
        
        info = {
            'soc': self.current_soc,
            'grid_energy_kwh': grid_energy_kwh,
            'step_cost': step_cost,
            'episode_cost': self.episode_cost,
            'time_elapsed': self.time_elapsed,
            'departed': terminated
        }
        
        return next_observation, reward, terminated, truncated, info
