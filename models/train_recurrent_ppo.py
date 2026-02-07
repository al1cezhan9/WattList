"""
Train Recurrent PPO (LSTM) policy for EV charging.

This script trains a recurrent policy that can handle partial observability
and learn from the hidden departure process.
"""

import os
import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
import gymnasium as gym

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import EVChargingEnv


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config: dict, rank: int = 0):
    """Create environment function for vectorization."""
    def _init():
        env = EVChargingEnv(config['environment'])
        env = Monitor(env)
        return env
    return _init


def train_recurrent_ppo(config_path: str = "configs/default.yaml"):
    """
    Train Recurrent PPO agent for EV charging.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    config = load_config(config_path)
    env_config = config['environment']
    ppo_config = config['recurrent_ppo']
    train_config = config['training']
    
    print("="*60)
    print("🚀 Training Recurrent PPO for EV Charging")
    print("="*60)
    print(f"Configuration: {config_path}")
    print(f"Total timesteps: {train_config['total_timesteps']}")
    print(f"LSTM hidden size: {ppo_config['lstm_hidden_size']}")
    print("="*60)
    
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create vectorized environment
    # Use parallel environments for faster training (adjust num_envs as needed)
    num_envs = train_config.get('num_envs', 1)  # Default: 1, increase for speed
    env = DummyVecEnv([make_env(config, i) for i in range(num_envs)])
    eval_env = DummyVecEnv([make_env(config, i) for i in range(1)])
    
    if num_envs > 1:
        print(f"✅ Using {num_envs} parallel environments for faster training")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create RecurrentPPO agent
    model = RecurrentPPO(
        ppo_config['policy'],
        env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        policy_kwargs=dict(
            lstm_hidden_size=ppo_config['lstm_hidden_size'],
            n_lstm_layers=ppo_config['n_lstm_layers'],
        ),
        verbose=1,
        tensorboard_log=train_config['tensorboard_log'],
        device='cpu',  # CPU only for ARM64
    )
    
    print("✅ RecurrentPPO agent created")
    print(f"   Policy: {ppo_config['policy']}")
    print(f"   LSTM hidden size: {ppo_config['lstm_hidden_size']}")
    print(f"   LSTM layers: {ppo_config['n_lstm_layers']}")
    
    # Setup callbacks
    os.makedirs(os.path.dirname(train_config['model_save_path']), exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=train_config['save_freq'],
        save_path=os.path.dirname(train_config['model_save_path']),
        name_prefix='recurrent_ppo_ev_charging'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(train_config['model_save_path']),
        log_path='./logs/',
        eval_freq=train_config['eval_freq'],
        n_eval_episodes=train_config['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    # Train the agent
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    model.learn(
        total_timesteps=train_config['total_timesteps'],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        log_interval=train_config['log_interval']
    )
    
    # Save final model
    final_model_path = train_config['model_save_path']
    print(f"\n💾 Saving final model to {final_model_path}...")
    model.save(final_model_path)
    print("✅ Training complete!")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Recurrent PPO for EV Charging')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    train_recurrent_ppo(args.config)
