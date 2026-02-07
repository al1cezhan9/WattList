"""
Tests for the EV Charging Environment.
"""
import numpy as np
import pytest
from envs import EVChargingEnv


@pytest.fixture
def env_config():
    """Default environment configuration for testing."""
    return {
        'ev_capacity_kwh': 75.0,
        'charging_efficiency': 0.90,
        'max_charging_power_kw': 7.4,
        'target_soc': 0.70,
        'lambda_penalty': 25.0,
        'max_pv_output_kw': 10.0,
        'pv_noise_std': 0.5,
        'pv_ar_coef': 0.7,
        'pv_forecast_bias': 0.1,
        'tou_prices': {
            'off_peak': 0.10,
            'mid': 0.20,
            'peak': 0.30
        },
        'base_hazard_rate': 0.02,
        'max_episode_steps': 96
    }


@pytest.fixture
def env(env_config):
    """Create environment instance for testing."""
    return EVChargingEnv(env_config)


class TestEVChargingEnv:
    """Test suite for EVChargingEnv."""
    
    def test_initialization(self, env, env_config):
        """Test environment initialization."""
        assert env.ev_capacity_kwh == env_config['ev_capacity_kwh']
        assert env.charging_efficiency == env_config['charging_efficiency']
        assert env.max_charging_power_kw == env_config['max_charging_power_kw']
        assert env.target_soc == env_config['target_soc']
        assert env.lambda_penalty == env_config['lambda_penalty']
        assert env.base_hazard_rate == env_config['base_hazard_rate']
    
    def test_observation_space(self, env):
        """Test observation space shape and bounds."""
        assert env.observation_space.shape == (12,)
        assert env.obs_dim == 12
        
        # Test bounds
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        
        assert obs_low[0] == 0.0  # SoC lower bound
        assert obs_high[0] == 1.0  # SoC upper bound
        assert obs_low[1] == 0.0  # Time lower bound
        assert obs_high[1] == 1.0  # Time upper bound
    
    def test_action_space(self, env):
        """Test action space."""
        assert env.action_space.n == 2  # Two actions: solar only, grid only
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        
        # Check observation shape
        assert obs.shape == (12,)
        assert isinstance(obs, np.ndarray)
        
        # Check initial state
        assert 0.2 <= env.current_soc <= 0.5  # Random initial SoC
        assert 0.0 <= env.current_hour < 24.0  # Random start hour
        assert env.time_elapsed == 0
        assert env.episode_cost == 0.0
    
    def test_step_solar_only(self, env):
        """Test step with solar only action (action 0)."""
        obs, _ = env.reset()
        env.current_hour = 12.0  # Noon (should have solar)
        
        # Get PV output
        pv_output = env._get_pv_output(env.current_hour)
        
        # Step with solar only
        obs_next, reward, terminated, truncated, info = env.step(0)
        
        # Check that no grid power was used
        assert info['grid_energy_kwh'] == 0.0
        assert info['step_cost'] == 0.0
        assert reward == 0.0  # No cost for solar
        
        # Check SoC increased if solar available
        if pv_output > 0:
            assert obs_next[0] > obs[0]  # SoC increased
    
    def test_step_grid_only(self, env):
        """Test step with grid only action (action 1)."""
        obs, _ = env.reset()
        grid_price = env._get_grid_price(env.current_hour)
        
        # Step with grid only
        obs_next, reward, terminated, truncated, info = env.step(1)
        
        # Check that grid power was used
        assert info['grid_energy_kwh'] > 0.0
        assert info['step_cost'] > 0.0
        assert reward < 0.0  # Negative reward (cost)
        
        # Check SoC increased
        assert obs_next[0] > obs[0]
        
        # Check cost calculation
        expected_cost = info['grid_energy_kwh'] * grid_price
        assert abs(info['step_cost'] - expected_cost) < 1e-6
    
    def test_charging_calculation(self, env):
        """Test charging power and SoC update calculation."""
        obs, _ = env.reset()
        initial_soc = env.current_soc
        
        # Step with grid only (max power)
        obs_next, _, _, _, info = env.step(1)
        
        # Calculate expected energy added
        charging_power = env.max_charging_power_kw
        energy_input = charging_power * 0.25  # 15 minutes
        energy_stored = energy_input * env.charging_efficiency
        expected_soc_increase = energy_stored / env.ev_capacity_kwh
        
        # Check SoC increase matches expected
        actual_soc_increase = obs_next[0] - initial_soc
        assert abs(actual_soc_increase - expected_soc_increase) < 1e-6
    
    def test_soc_clipping(self, env):
        """Test that SoC is clipped to [0, 1]."""
        obs, _ = env.reset()
        env.current_soc = 0.99  # Start near max
        
        # Charge multiple steps
        for _ in range(10):
            obs, _, _, _, _ = env.step(1)
            assert 0.0 <= obs[0] <= 1.0  # SoC stays in bounds
    
    def test_reward_function(self, env):
        """Test reward function calculation."""
        obs, _ = env.reset()
        
        # Test solar only (should be 0)
        _, reward_solar, _, _, _ = env.step(0)
        assert reward_solar == 0.0
        
        # Test grid only (should be negative)
        _, reward_grid, _, _, _ = env.step(1)
        assert reward_grid < 0.0
    
    def test_terminal_penalty(self, env):
        """Test terminal penalty on departure."""
        obs, _ = env.reset()
        env.current_soc = 0.50  # Below target (0.70)
        
        # Force departure by setting high hazard
        env.base_hazard_rate = 1.0  # 100% departure chance
        
        # Step until departure
        total_reward = 0.0
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            total_reward += reward
            if terminated:
                break
        
        # Check that penalty was applied
        if terminated:
            soc_deficit = max(0.0, env.target_soc - env.current_soc)
            expected_penalty = -env.lambda_penalty * soc_deficit
            assert reward < 0.0  # Should have penalty
    
    def test_pv_output(self, env):
        """Test PV output generation."""
        # Test at noon (should have high output)
        pv_noon = env._get_pv_output(12.0)
        assert 0.0 <= pv_noon <= env.max_pv_output_kw
        
        # Test at midnight (should have no output)
        pv_midnight = env._get_pv_output(0.0)
        assert pv_midnight == 0.0 or pv_midnight < 1.0  # May have small noise
    
    def test_grid_price(self, env):
        """Test grid price schedule."""
        # Test off-peak (2 AM)
        price_offpeak = env._get_grid_price(2.0)
        assert price_offpeak == env.tou_prices['off_peak']
        
        # Test mid (10 AM)
        price_mid = env._get_grid_price(10.0)
        assert price_mid == env.tou_prices['mid']
        
        # Test peak (6 PM)
        price_peak = env._get_grid_price(18.0)
        assert price_peak == env.tou_prices['peak']
    
    def test_departure_sampling(self, env):
        """Test departure sampling."""
        obs, _ = env.reset()
        
        # Set low hazard rate
        env.base_hazard_rate = 0.001
        
        # Run many steps - should rarely depart
        departures = 0
        for _ in range(100):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                departures += 1
                obs, _ = env.reset()
        
        # With low hazard, should have few departures
        assert departures < 10  # Should be rare
    
    def test_episode_length(self, env):
        """Test episode length limits."""
        obs, _ = env.reset()
        
        # Run until truncation
        steps = 0
        while steps < env.max_episode_steps + 10:
            obs, _, terminated, truncated, _ = env.step(0)
            steps += 1
            if terminated or truncated:
                break
        
        # Should truncate at max_episode_steps
        assert truncated or steps <= env.max_episode_steps
    
    def test_observation_components(self, env):
        """Test observation components."""
        obs, _ = env.reset()
        
        # Check each component
        assert 0.0 <= obs[0] <= 1.0  # SoC
        assert 0.0 <= obs[1] <= 1.0  # Time elapsed
        assert 0.0 <= obs[2] <= env.max_pv_output_kw  # PV current
        assert len(obs) == 12  # Total dimension
    
    def test_info_dict(self, env):
        """Test info dictionary contents."""
        obs, _ = env.reset()
        _, _, _, _, info = env.step(1)
        
        # Check info keys
        assert 'soc' in info
        assert 'grid_energy_kwh' in info
        assert 'step_cost' in info
        assert 'episode_cost' in info
        assert 'time_elapsed' in info
        assert 'departed' in info
        
        # Check values
        assert 0.0 <= info['soc'] <= 1.0
        assert info['grid_energy_kwh'] >= 0.0
        assert info['step_cost'] >= 0.0
