import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces


class SolarArbitrageEnv(gym.Env):
    """
    Solar-Arbitrage EV Controller Environment.
    
    State Space (Box, 4D):
    - solar_output: 0-10 kW (mocked diurnal sine wave + Gaussian noise)
    - ev_soc: 0.0-1.0 (current EV battery level)
    - time_remaining: 0-24 hours (hours until departure)
    - grid_price: $/kWh (3-tier Time-of-Use tariff)
    
    Action Space (Discrete, 2):
    - Action 0 (Solar Only): P = min(solar_output, 7.4 kW), Cost = $0
    - Action 1 (Solar + Grid): P = 7.4 kW, Grid provides difference
    
    Constraints:
    - EV battery capacity: 75 kWh
    - Charging efficiency: η = 0.90
    - NO house battery
    - Each step = 1 hour
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = {'environment': {}}
        
        env_config = config.get('environment', {})
        
        # EV battery constraints
        self.ev_capacity_kwh = env_config.get('ev_capacity_kwh', 75.0)  # 75 kWh
        self.charging_efficiency = env_config.get('charging_efficiency', 0.90)  # η = 0.90
        self.max_charging_power_kw = env_config.get('max_charging_power_kw', 7.4)  # 7.4 kW
        
        # Solar generation parameters
        self.max_solar_output_kw = env_config.get('max_solar_output_kw', 10.0)  # 0-10 kW
        self.solar_noise_std = env_config.get('solar_noise_std', 0.5)  # Gaussian noise std
        
        # Time-of-Use pricing (3-tier)
        self.tou_prices = env_config.get('tou_prices', {
            'peak': 0.30,      # Peak hours: 4 PM - 9 PM
            'mid': 0.20,       # Mid hours: 9 AM - 4 PM, 9 PM - 11 PM
            'off_peak': 0.10   # Off-peak: 11 PM - 9 AM
        })
        
        # Episode parameters
        self.max_hours = env_config.get('max_hours', 24)  # 24-hour episode
        self.target_soc = env_config.get('target_soc', 0.9)  # Target SoC at departure
        self.departure_penalty_scale = env_config.get('departure_penalty_scale', 100.0)
        
        # Reward weights
        self.solar_reward_weight = env_config.get('solar_reward_weight', 1.5)
        self.grid_cost_weight = env_config.get('grid_cost_weight', 1.0)
        
        # Define observation space: [solar_output, ev_soc, time_remaining, grid_price]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0, 24.0, 0.5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space: 2 discrete actions
        self.action_space = spaces.Discrete(2)
        
        # Internal state
        self.current_hour = 0
        self.ev_soc = 0.0
        self.time_remaining = 0.0
        
        # Track episode history for visualization
        self.episode_history = {
            'solar_output': [],
            'ev_soc': [],
            'grid_price': [],
            'action': [],
            'solar_power': [],
            'grid_power': [],
            'reward': []
        }
        
        self.reset()
    
    def _get_solar_output(self, hour):
        """
        Generate solar output using diurnal sine wave + Gaussian noise.
        Peak at noon (hour 12), zero at night.
        """
        # Normalize hour to 0-24 range
        hour_normalized = hour % 24
        
        # Diurnal sine wave: peak at hour 12 (noon)
        # Using sine from -π/2 to 3π/2 to cover full day
        if 6 <= hour_normalized <= 18:  # Daylight hours
            # Map 6-18 hours to -π/2 to π/2
            angle = (hour_normalized - 12) * np.pi / 12
            solar_base = self.max_solar_output_kw * np.cos(angle)
        else:
            solar_base = 0.0
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.solar_noise_std)
        solar_output = np.clip(solar_base + noise, 0.0, self.max_solar_output_kw)
        
        return solar_output
    
    def _get_grid_price(self, hour):
        """
        Get Time-of-Use grid price based on hour of day.
        3-tier pricing: Peak (4 PM - 9 PM), Mid (9 AM - 4 PM, 9 PM - 11 PM), Off-peak (11 PM - 9 AM)
        """
        hour_normalized = hour % 24
        
        if 16 <= hour_normalized < 21:  # 4 PM - 9 PM
            return self.tou_prices['peak']
        elif (9 <= hour_normalized < 16) or (21 <= hour_normalized < 23):  # 9 AM - 4 PM, 9 PM - 11 PM
            return self.tou_prices['mid']
        else:  # 11 PM - 9 AM
            return self.tou_prices['off_peak']
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random initial SoC between 0.2 and 0.5
        self.ev_soc = np.random.uniform(0.2, 0.5)
        
        # Random time remaining (hours until departure) between 4 and 24 hours
        self.time_remaining = np.random.uniform(4.0, self.max_hours)
        
        # Start at random hour of day
        self.current_hour = np.random.randint(0, 24)
        
        # Clear episode history
        self.episode_history = {
            'solar_output': [],
            'ev_soc': [],
            'grid_price': [],
            'action': [],
            'solar_power': [],
            'grid_power': [],
            'reward': []
        }
        
        # Get initial state
        solar_output = self._get_solar_output(self.current_hour)
        grid_price = self._get_grid_price(self.current_hour)
        
        state = np.array([
            solar_output,
            self.ev_soc,
            self.time_remaining,
            grid_price
        ], dtype=np.float32)
        
        return state, {}
    
    def step(self, action):
        """
        Execute one step (1 hour) in the environment.
        
        Args:
            action: 0 (Solar Only) or 1 (Solar + Grid)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current solar output and grid price
        solar_output = self._get_solar_output(self.current_hour)
        grid_price = self._get_grid_price(self.current_hour)
        
        # Calculate charging power based on action
        if action == 0:  # Solar Only
            charging_power_kw = np.minimum(solar_output, self.max_charging_power_kw)
            grid_power_kw = 0.0
            solar_power_kw = charging_power_kw
        else:  # Action 1: Solar + Grid (Max Charge)
            charging_power_kw = self.max_charging_power_kw
            grid_power_kw = np.maximum(0.0, self.max_charging_power_kw - solar_output)
            solar_power_kw = charging_power_kw - grid_power_kw
        
        # Update EV SoC (accounting for charging efficiency)
        energy_added_kwh = charging_power_kw * 1.0 * self.charging_efficiency  # 1 hour step
        self.ev_soc = np.clip(
            self.ev_soc + (energy_added_kwh / self.ev_capacity_kwh),
            0.0, 1.0
        )
        
        # Update time
        self.time_remaining = max(0.0, self.time_remaining - 1.0)
        self.current_hour = (self.current_hour + 1) % 24
        
        # Calculate reward
        # Sustainability: Reward for using free solar energy
        solar_reward = self.solar_reward_weight * solar_power_kw
        
        # Cost: Negative penalty based on grid power cost
        grid_cost = grid_power_kw * grid_price
        
        # Base reward: R = (1.5 * P_solar) - (P_grid * price)
        reward = solar_reward - grid_cost
        
        # Check if episode is done (departure time reached)
        terminated = (self.time_remaining <= 0.0)
        
        # Apply terminal penalty if SoC < target at departure
        # Penalty_deadline = -100 * (0.9 - final_soc) if time_remaining == 0 and ev_soc < 0.9
        if terminated and self.ev_soc < self.target_soc:
            soc_deficit = self.target_soc - self.ev_soc
            terminal_penalty = -self.departure_penalty_scale * soc_deficit
            reward += terminal_penalty
        
        # Truncated flag (not used in this environment)
        truncated = False
        
        # Get next state
        next_solar_output = self._get_solar_output(self.current_hour)
        next_grid_price = self._get_grid_price(self.current_hour)
        
        next_state = np.array([
            next_solar_output,
            self.ev_soc,
            self.time_remaining,
            next_grid_price
        ], dtype=np.float32)
        
        # Store history for visualization
        self.episode_history['solar_output'].append(solar_output)
        self.episode_history['ev_soc'].append(self.ev_soc)
        self.episode_history['grid_price'].append(grid_price)
        self.episode_history['action'].append(action)
        self.episode_history['solar_power'].append(solar_power_kw)
        self.episode_history['grid_power'].append(grid_power_kw)
        self.episode_history['reward'].append(reward)
        
        info = {
            'solar_power_kw': solar_power_kw,
            'grid_power_kw': grid_power_kw,
            'charging_power_kw': charging_power_kw,
            'energy_added_kwh': energy_added_kwh,
            'grid_cost': grid_power_kw * grid_price
        }
        
        return next_state, reward, terminated, truncated, info
