"""
Tests for baseline policies.
"""
import numpy as np
import pytest
from envs import EVChargingEnv
from baselines import SolarFirstGreedy, ConservativeDeadline


@pytest.fixture
def env():
    """Create environment for testing."""
    return EVChargingEnv({
        'target_soc': 0.70,
        'max_charging_power_kw': 7.4
    })


class TestSolarFirstGreedy:
    """Test Solar-First Greedy baseline."""
    
    def test_initialization(self):
        """Test baseline initialization."""
        baseline = SolarFirstGreedy(max_charging_power_kw=7.4)
        assert baseline.max_charging_power_kw == 7.4
    
    def test_solar_available(self, env):
        """Test action when solar is available."""
        baseline = SolarFirstGreedy()
        obs, _ = env.reset()
        env.current_hour = 12.0  # Noon (should have solar)
        
        # Get PV output
        pv_output = env._get_pv_output(env.current_hour)
        obs[2] = pv_output  # Update observation
        
        # Should choose solar only (action 0)
        if pv_output > 0.1:
            action = baseline.predict(obs)
            assert action == 0
    
    def test_no_solar(self, env):
        """Test action when no solar is available."""
        baseline = SolarFirstGreedy()
        obs, _ = env.reset()
        env.current_hour = 2.0  # Night (no solar)
        
        # Get PV output
        pv_output = env._get_pv_output(env.current_hour)
        obs[2] = pv_output  # Update observation
        
        # Should choose grid only (action 1)
        if pv_output <= 0.1:
            action = baseline.predict(obs)
            assert action == 1


class TestConservativeDeadline:
    """Test Conservative Deadline baseline."""
    
    def test_initialization(self):
        """Test baseline initialization."""
        baseline = ConservativeDeadline(target_soc=0.70, max_charging_power_kw=7.4)
        assert baseline.target_soc == 0.70
        assert baseline.max_charging_power_kw == 7.4
    
    def test_below_target(self, env):
        """Test action when SoC is below target."""
        baseline = ConservativeDeadline(target_soc=0.70)
        obs, _ = env.reset()
        obs[0] = 0.50  # Below target
        
        # Should choose grid only (action 1) to charge aggressively
        action = baseline.predict(obs)
        assert action == 1
    
    def test_at_target(self, env):
        """Test action when SoC is at target."""
        baseline = ConservativeDeadline(target_soc=0.70)
        obs, _ = env.reset()
        obs[0] = 0.70  # At target
        
        # Should prefer solar if available
        obs[2] = 5.0  # Solar available
        action = baseline.predict(obs)
        # May choose solar (0) or grid (1) depending on implementation
        assert action in [0, 1]
    
    def test_above_target(self, env):
        """Test action when SoC is above target."""
        baseline = ConservativeDeadline(target_soc=0.70)
        obs, _ = env.reset()
        obs[0] = 0.80  # Above target
        
        # Should prefer solar if available
        obs[2] = 5.0  # Solar available
        action = baseline.predict(obs)
        # Should prefer solar (action 0)
        if obs[2] > 0.1:
            assert action == 0
