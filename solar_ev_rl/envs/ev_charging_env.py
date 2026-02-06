"""
EV Charging Environment with Solar Uncertainty and Unknown Departure
====================================================================

This environment simulates EV charging decisions under:
1. Uncertain solar PV generation (correlated forecast errors)
2. Unknown, non-stationary departure behavior (hidden hazard model)
3. Time-of-use grid pricing

The agent must learn to balance:
- Using free solar energy when available
- Supplementing with grid power to meet charging goals
- Handling uncertainty about when the EV will depart

CRITICAL DESIGN CHOICE:
The departure probability is NOT included in the observation.
The agent must learn the departure distribution implicitly through experience.
This tests the agent's ability to handle epistemic uncertainty.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class EVChargingEnv(gym.Env):
    """
    Gymnasium environment for EV charging under solar uncertainty.
    
    Observation Space (7-dimensional float vector):
        [0] current_soc: Battery state of charge (0-1)
        [1] time_elapsed_normalized: Fraction of max episode time (0-1)
        [2] current_pv_output: Current solar generation (kW, 0-10)
        [3-6] pv_forecast_mean: Mean PV forecast for next 4 steps
        [7-10] pv_forecast_std: Std of PV forecast for next 4 steps
        [11] current_grid_price: Current electricity price ($/kWh)
    
    Action Space (Discrete, 2 actions):
        0: Solar-only charging (use only available PV)
        1: Solar + Grid charging (charge at max power, grid fills gap)
    
    Episode Dynamics:
        - Starts when EV plugs in with random initial SoC
        - Ends when EV departs (stochastic, unknown to agent)
        - Each timestep = 15 minutes
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None):
        """
        Initialize the EV charging environment.
        
        Args:
            config: Configuration dictionary (from YAML)
            render_mode: Rendering mode (optional)
        """
        super().__init__()
        
        # Load configuration with defaults
        self.config = config or {}
        env_cfg = self.config.get('environment', {})
        dep_cfg = self.config.get('departure', {})
        reward_cfg = self.config.get('reward', {})
        
        # =====================================================================
        # TIME PARAMETERS
        # =====================================================================
        self.timestep_minutes = env_cfg.get('timestep_minutes', 15)
        self.max_episode_steps = env_cfg.get('max_episode_steps', 96)  # 24 hours
        self.timestep_hours = self.timestep_minutes / 60.0
        
        # =====================================================================
        # EV BATTERY PARAMETERS
        # =====================================================================
        self.battery_capacity_kwh = env_cfg.get('battery_capacity_kwh', 75.0)
        self.max_charging_power_kw = env_cfg.get('max_charging_power_kw', 7.4)
        self.charging_efficiency = env_cfg.get('charging_efficiency', 0.90)
        self.initial_soc_range = env_cfg.get('initial_soc_range', [0.2, 0.5])
        self.target_soc = env_cfg.get('target_soc', 0.90)
        
        # =====================================================================
        # SOLAR PV PARAMETERS
        # =====================================================================
        solar_cfg = env_cfg.get('solar', {})
        self.peak_solar_kw = solar_cfg.get('peak_output_kw', 10.0)
        self.sunrise_hour = solar_cfg.get('sunrise_hour', 6.0)
        self.sunset_hour = solar_cfg.get('sunset_hour', 20.0)
        self.solar_noise_std = solar_cfg.get('noise_std', 0.5)
        self.ar1_coef = solar_cfg.get('ar1_coefficient', 0.7)
        self.forecast_bias = solar_cfg.get('forecast_bias', 0.1)
        self.forecast_horizon = solar_cfg.get('forecast_horizon', 4)
        
        # =====================================================================
        # GRID PRICING PARAMETERS
        # =====================================================================
        grid_cfg = env_cfg.get('grid', {})
        self.off_peak_price = grid_cfg.get('off_peak_price', 0.10)
        self.mid_price = grid_cfg.get('mid_price', 0.20)
        self.peak_price = grid_cfg.get('peak_price', 0.30)
        
        # =====================================================================
        # DEPARTURE MODEL PARAMETERS (Hidden from agent!)
        # =====================================================================
        self.num_user_types = dep_cfg.get('num_user_types', 3)
        self.base_hazard_rates = np.array(dep_cfg.get('base_hazard_rates', [0.02, 0.01, 0.005]))
        self.time_hazard_scale = dep_cfg.get('time_hazard_scale', 0.001)
        self.drift_rate = dep_cfg.get('drift_rate', 0.0001)
        self.drift_period = dep_cfg.get('drift_period_episodes', 1000)
        self.min_stay_steps = dep_cfg.get('min_stay_steps', 4)
        
        # =====================================================================
        # REWARD PARAMETERS
        # =====================================================================
        self.grid_cost_weight = reward_cfg.get('grid_cost_weight', 1.0)
        self.undercharge_penalty = reward_cfg.get('undercharge_penalty_lambda', 50.0)
        self.solar_bonus_weight = reward_cfg.get('solar_bonus_weight', 0.1)
        
        # =====================================================================
        # OBSERVATION & ACTION SPACES
        # =====================================================================
        # Observation: [soc, time_norm, pv_now, pv_forecast_mean(4), pv_forecast_std(4), price]
        obs_dim = 1 + 1 + 1 + self.forecast_horizon + self.forecast_horizon + 1
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.array(
                [1.0, 1.0, self.peak_solar_kw] +  # soc, time, pv_now
                [self.peak_solar_kw] * self.forecast_horizon +  # forecast mean
                [self.peak_solar_kw] * self.forecast_horizon +  # forecast std
                [1.0],  # price (normalized)
                dtype=np.float32
            ),
            dtype=np.float32
        )
        
        # Action: 0 = solar only, 1 = solar + grid
        self.action_space = spaces.Discrete(2)
        
        # =====================================================================
        # INTERNAL STATE VARIABLES
        # =====================================================================
        self.current_soc = 0.0
        self.current_step = 0
        self.start_hour = 0.0  # Hour of day when episode starts
        self.ar1_noise = 0.0   # AR(1) noise state for solar forecast error
        self.user_type = 0     # Hidden user type for this episode
        self.episode_count = 0  # Track episodes for non-stationarity
        
        # User type distribution (drifts over training)
        self.user_type_probs = np.ones(self.num_user_types) / self.num_user_types
        
        # Episode tracking for metrics
        self.episode_solar_used = 0.0
        self.episode_grid_used = 0.0
        self.episode_cost = 0.0
        
        self.render_mode = render_mode
        self._np_random = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        The departure model evolves over episodes (non-stationary).
        """
        super().reset(seed=seed)
        
        # Update episode counter for non-stationarity
        self.episode_count += 1
        
        # =====================================================================
        # NON-STATIONARY DEPARTURE: Drift user type distribution
        # =====================================================================
        # User type probabilities drift sinusoidally over training
        # This simulates seasonal changes in user behavior
        drift_phase = 2 * np.pi * self.episode_count / self.drift_period
        drift = self.drift_rate * np.sin(drift_phase)
        
        # Shift probability mass between user types
        self.user_type_probs[0] += drift
        self.user_type_probs[2] -= drift
        self.user_type_probs = np.clip(self.user_type_probs, 0.1, 0.6)
        self.user_type_probs /= self.user_type_probs.sum()  # Normalize
        
        # Sample hidden user type for this episode
        self.user_type = self.np_random.choice(
            self.num_user_types, 
            p=self.user_type_probs
        )
        
        # =====================================================================
        # INITIALIZE EPISODE STATE
        # =====================================================================
        # Random initial SoC
        self.current_soc = self.np_random.uniform(*self.initial_soc_range)
        
        # Random start time (hour of day)
        self.start_hour = self.np_random.uniform(0, 24)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset AR(1) noise for solar forecast
        self.ar1_noise = 0.0
        
        # Reset episode metrics
        self.episode_solar_used = 0.0
        self.episode_grid_used = 0.0
        self.episode_cost = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        info = {"user_type": self.user_type}  # Hidden, for debugging only
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one charging decision.
        
        Args:
            action: 0 = solar only, 1 = solar + grid
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # =====================================================================
        # GET CURRENT SOLAR AND PRICE
        # =====================================================================
        current_hour = self._get_current_hour()
        solar_available = self._get_actual_solar(current_hour)
        grid_price = self._get_grid_price(current_hour)
        
        # =====================================================================
        # EXECUTE CHARGING ACTION
        # =====================================================================
        if action == 0:
            # Solar only: charge at rate of available solar
            charging_power = min(solar_available, self.max_charging_power_kw)
            solar_used = charging_power
            grid_used = 0.0
        else:
            # Solar + Grid: charge at max rate, grid fills the gap
            charging_power = self.max_charging_power_kw
            solar_used = min(solar_available, charging_power)
            grid_used = charging_power - solar_used
        
        # Calculate energy delivered to battery (accounting for efficiency)
        energy_delivered_kwh = charging_power * self.timestep_hours * self.charging_efficiency
        
        # Update SoC
        soc_increase = energy_delivered_kwh / self.battery_capacity_kwh
        self.current_soc = min(1.0, self.current_soc + soc_increase)
        
        # Track energy usage
        self.episode_solar_used += solar_used * self.timestep_hours
        self.episode_grid_used += grid_used * self.timestep_hours
        
        # =====================================================================
        # CALCULATE STEP REWARD
        # =====================================================================
        # Cost of grid energy (negative reward)
        grid_cost = grid_used * self.timestep_hours * grid_price
        self.episode_cost += grid_cost
        
        # Small bonus for using solar
        solar_bonus = solar_used * self.timestep_hours * self.solar_bonus_weight
        
        step_reward = -self.grid_cost_weight * grid_cost + solar_bonus
        
        # =====================================================================
        # CHECK FOR DEPARTURE (Stochastic, hidden from agent)
        # =====================================================================
        self.current_step += 1
        departed = self._check_departure()
        truncated = self.current_step >= self.max_episode_steps
        terminated = departed
        
        # =====================================================================
        # TERMINAL REWARD (Undercharge penalty)
        # =====================================================================
        if terminated or truncated:
            undercharge = max(0.0, self.target_soc - self.current_soc)
            terminal_penalty = self.undercharge_penalty * undercharge
            step_reward -= terminal_penalty
        
        # =====================================================================
        # GET NEW OBSERVATION
        # =====================================================================
        obs = self._get_observation()
        
        info = {
            "soc": self.current_soc,
            "solar_used": solar_used,
            "grid_used": grid_used,
            "grid_cost": grid_cost,
            "departed": departed,
            "user_type": self.user_type,  # Hidden, for analysis only
            "episode_cost": self.episode_cost,
            "episode_solar": self.episode_solar_used,
            "episode_grid": self.episode_grid_used,
        }
        
        return obs, step_reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector (does NOT include departure info)."""
        current_hour = self._get_current_hour()
        
        # Current solar (with noise)
        current_pv = self._get_actual_solar(current_hour)
        
        # PV forecast for next N steps (biased, with uncertainty)
        forecast_mean, forecast_std = self._get_solar_forecast(current_hour)
        
        # Current grid price (normalized to 0-1)
        current_price = self._get_grid_price(current_hour) / self.peak_price
        
        # Time elapsed (normalized)
        time_normalized = self.current_step / self.max_episode_steps
        
        # Assemble observation
        obs = np.array(
            [self.current_soc, time_normalized, current_pv] +
            list(forecast_mean) +
            list(forecast_std) +
            [current_price],
            dtype=np.float32
        )
        
        return obs
    
    def _get_current_hour(self) -> float:
        """Get current hour of day (0-24)."""
        hours_elapsed = self.current_step * self.timestep_hours
        return (self.start_hour + hours_elapsed) % 24.0
    
    def _get_base_solar(self, hour: float) -> float:
        """
        Get base solar output (no noise) using bell curve.
        Peak at solar noon (midpoint between sunrise and sunset).
        """
        if hour < self.sunrise_hour or hour > self.sunset_hour:
            return 0.0
        
        # Sinusoidal bell curve
        solar_noon = (self.sunrise_hour + self.sunset_hour) / 2
        day_length = self.sunset_hour - self.sunrise_hour
        
        # Map hour to angle (0 at sunrise, π at sunset)
        angle = np.pi * (hour - self.sunrise_hour) / day_length
        
        return self.peak_solar_kw * np.sin(angle)
    
    def _get_actual_solar(self, hour: float) -> float:
        """Get actual solar output with AR(1) correlated noise."""
        base_solar = self._get_base_solar(hour)
        
        # Update AR(1) noise process
        innovation = self.np_random.normal(0, self.solar_noise_std)
        self.ar1_noise = self.ar1_coef * self.ar1_noise + np.sqrt(1 - self.ar1_coef**2) * innovation
        
        # Apply noise to solar output
        actual = base_solar + self.ar1_noise
        return np.clip(actual, 0.0, self.peak_solar_kw)
    
    def _get_solar_forecast(self, current_hour: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get imperfect solar forecast for next N timesteps.
        Forecast has systematic bias and increasing uncertainty with horizon.
        """
        forecast_mean = np.zeros(self.forecast_horizon, dtype=np.float32)
        forecast_std = np.zeros(self.forecast_horizon, dtype=np.float32)
        
        for i in range(self.forecast_horizon):
            future_hour = (current_hour + (i + 1) * self.timestep_hours) % 24.0
            base_solar = self._get_base_solar(future_hour)
            
            # Optimistic bias in forecast
            forecast_mean[i] = base_solar * (1 + self.forecast_bias)
            forecast_mean[i] = np.clip(forecast_mean[i], 0.0, self.peak_solar_kw)
            
            # Uncertainty increases with forecast horizon
            forecast_std[i] = self.solar_noise_std * (1 + 0.2 * i)
        
        return forecast_mean, forecast_std
    
    def _get_grid_price(self, hour: float) -> float:
        """Get time-of-use grid price for given hour."""
        hour_int = int(hour) % 24
        
        # Peak: 4 PM - 9 PM (hours 16-21)
        if 16 <= hour_int < 21:
            return self.peak_price
        # Off-peak: 11 PM - 9 AM (hours 23-9)
        elif hour_int >= 23 or hour_int < 9:
            return self.off_peak_price
        # Mid: 9 AM - 4 PM, 9 PM - 11 PM
        else:
            return self.mid_price
    
    def _check_departure(self) -> bool:
        """
        Check if EV departs this timestep.
        Uses hidden hazard model - NOT observable by agent!
        
        The hazard rate depends on:
        1. User type (sampled at episode start)
        2. Time elapsed (departure becomes more likely over time)
        """
        # Minimum stay period
        if self.current_step < self.min_stay_steps:
            return False
        
        # Base hazard rate from user type
        base_hazard = self.base_hazard_rates[self.user_type]
        
        # Time-dependent hazard increase
        time_hazard = self.time_hazard_scale * self.current_step
        
        # Combined hazard rate (probability of leaving this step)
        hazard = base_hazard + time_hazard
        
        # Clamp hazard to valid probability range
        hazard = np.clip(hazard, 0.0, 0.5)
        
        # Bernoulli trial for departure
        return self.np_random.random() < hazard
    
    def render(self):
        """Optional rendering for debugging."""
        if self.render_mode == "human":
            hour = self._get_current_hour()
            solar = self._get_actual_solar(hour)
            price = self._get_grid_price(hour)
            print(f"Step {self.current_step:3d} | "
                  f"Hour {hour:5.1f} | "
                  f"SoC {self.current_soc:.2%} | "
                  f"Solar {solar:5.2f} kW | "
                  f"Price ${price:.2f}")
    
    @property
    def np_random(self):
        """Lazy initialization of numpy random generator."""
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random


def test_environment():
    """Quick test to verify environment works correctly."""
    print("=" * 60)
    print("Testing EVChargingEnv")
    print("=" * 60)
    
    env = EVChargingEnv()
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")
    
    # Run a few steps
    total_reward = 0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i < 5:
            print(f"Step {i+1}: action={action}, reward={reward:.3f}, soc={info['soc']:.2%}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("Environment test PASSED!")
    return True


if __name__ == "__main__":
    test_environment()
