"""
Tests for training script and model creation.
"""
import os
import tempfile
import pytest
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import EVChargingEnv


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'environment': {
            'target_soc': 0.70,
            'base_hazard_rate': 0.02,
            'max_episode_steps': 96
        },
        'recurrent_ppo': {
            'policy': 'MlpLstmPolicy',
            'learning_rate': 3e-4,
            'n_steps': 128,  # Small for testing
            'batch_size': 32,
            'n_epochs': 2,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'lstm_hidden_size': 32,
            'n_lstm_layers': 1
        },
        'training': {
            'total_timesteps': 1000,  # Very small for testing
            'num_envs': 1,
            'model_save_path': 'test_model'
        }
    }


def make_test_env(config):
    """Create test environment."""
    def _init():
        return EVChargingEnv(config['environment'])
    return _init


class TestModelCreation:
    """Test model creation and basic training."""
    
    def test_model_initialization(self, config):
        """Test that model can be created."""
        env = DummyVecEnv([make_test_env(config)])
        
        model = RecurrentPPO(
            config['recurrent_ppo']['policy'],
            env,
            learning_rate=config['recurrent_ppo']['learning_rate'],
            n_steps=config['recurrent_ppo']['n_steps'],
            batch_size=config['recurrent_ppo']['batch_size'],
            n_epochs=config['recurrent_ppo']['n_epochs'],
            gamma=config['recurrent_ppo']['gamma'],
            gae_lambda=config['recurrent_ppo']['gae_lambda'],
            clip_range=config['recurrent_ppo']['clip_range'],
            ent_coef=config['recurrent_ppo']['ent_coef'],
            vf_coef=config['recurrent_ppo']['vf_coef'],
            policy_kwargs=dict(
                lstm_hidden_size=config['recurrent_ppo']['lstm_hidden_size'],
                n_lstm_layers=config['recurrent_ppo']['n_lstm_layers'],
            ),
            verbose=0,
            device='cpu'
        )
        
        assert model is not None
        assert model.policy is not None
    
    def test_model_training_step(self, config):
        """Test that model can perform a training step."""
        env = DummyVecEnv([make_test_env(config)])
        
        model = RecurrentPPO(
            config['recurrent_ppo']['policy'],
            env,
            learning_rate=config['recurrent_ppo']['learning_rate'],
            n_steps=config['recurrent_ppo']['n_steps'],
            batch_size=config['recurrent_ppo']['batch_size'],
            n_epochs=config['recurrent_ppo']['n_epochs'],
            policy_kwargs=dict(
                lstm_hidden_size=config['recurrent_ppo']['lstm_hidden_size'],
                n_lstm_layers=config['recurrent_ppo']['n_lstm_layers'],
            ),
            verbose=0,
            device='cpu'
        )
        
        # Train for a few steps
        model.learn(total_timesteps=config['recurrent_ppo']['n_steps'] * 2)
        
        # Model should still be valid
        assert model is not None
    
    def test_model_save_load(self, config):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = DummyVecEnv([make_test_env(config)])
            
            model = RecurrentPPO(
                config['recurrent_ppo']['policy'],
                env,
                learning_rate=config['recurrent_ppo']['learning_rate'],
                n_steps=config['recurrent_ppo']['n_steps'],
                policy_kwargs=dict(
                    lstm_hidden_size=config['recurrent_ppo']['lstm_hidden_size'],
                    n_lstm_layers=config['recurrent_ppo']['n_lstm_layers'],
                ),
                verbose=0,
                device='cpu'
            )
            
            # Train briefly
            model.learn(total_timesteps=config['recurrent_ppo']['n_steps'])
            
            # Save model
            save_path = os.path.join(tmpdir, 'test_model.zip')
            model.save(save_path)
            assert os.path.exists(save_path)
            
            # Load model
            loaded_model = RecurrentPPO.load(save_path, device='cpu')
            assert loaded_model is not None
            assert loaded_model.policy is not None
    
    def test_model_prediction(self, config):
        """Test model prediction."""
        env = DummyVecEnv([make_test_env(config)])
        
        model = RecurrentPPO(
            config['recurrent_ppo']['policy'],
            env,
            learning_rate=config['recurrent_ppo']['learning_rate'],
            n_steps=config['recurrent_ppo']['n_steps'],
            policy_kwargs=dict(
                lstm_hidden_size=config['recurrent_ppo']['lstm_hidden_size'],
                n_lstm_layers=config['recurrent_ppo']['n_lstm_layers'],
            ),
            verbose=0,
            device='cpu'
        )
        
        # Get observation
        obs = env.reset()
        
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Check action is valid
        assert action in [0, 1]
        assert action.shape == (1,)
