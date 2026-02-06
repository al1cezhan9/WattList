"""
Recurrent PPO Training Script for Solar EV Charging
====================================================

This script trains a recurrent (LSTM-based) PPO agent to optimize
EV charging decisions under solar uncertainty and unknown departures.

Key Features:
- Uses RecurrentPPO from sb3-contrib (LSTM policy)
- CPU-only training (designed for ARM64 Snapdragon)
- Vectorized environments for sample efficiency
- Comprehensive logging and checkpointing
- Evaluation callbacks

Why Recurrent Policy?
--------------------
The departure time is NOT observable - the agent only sees current state.
A recurrent policy can learn to infer latent state (like user type)
from the history of observations and actions.

This is crucial for:
1. Estimating departure risk from implicit patterns
2. Adapting to non-stationary user behavior
3. Making decisions under epistemic uncertainty

Usage:
    python train_recurrent_ppo.py --config configs/default.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional

# Stable Baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.utils import set_random_seed

# RecurrentPPO from sb3-contrib
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.ev_charging_env import EVChargingEnv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_env(config: Dict, seed: int = 0):
    """
    Factory function to create environment instances.
    Used by DummyVecEnv for vectorized training.
    """
    def _init():
        env = EVChargingEnv(config)
        env.reset(seed=seed)
        return env
    return _init


def setup_directories(config: Dict) -> Dict[str, str]:
    """Create directories for models and logs."""
    paths = config.get('paths', {})
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    dirs = {
        'model_dir': os.path.join(paths.get('model_dir', 'trained_models'), run_name),
        'log_dir': os.path.join(paths.get('log_dir', 'logs'), run_name),
        'export_dir': paths.get('export_dir', 'exported_models'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def create_recurrent_ppo(
    env,
    config: Dict,
    tensorboard_log: str
) -> RecurrentPPO:
    """
    Create RecurrentPPO agent with LSTM policy.
    
    The LSTM allows the agent to maintain hidden state across timesteps,
    enabling it to learn temporal patterns in the environment.
    
    This is critical because:
    1. Departure probability depends on hidden user type
    2. User type can be inferred from observation patterns
    3. Non-stationarity requires adaptation over time
    """
    train_cfg = config.get('training', {})
    
    # Policy architecture
    # The LSTM receives observations and previous hidden state,
    # outputting actions and value estimates
    policy_kwargs = dict(
        lstm_hidden_size=train_cfg.get('lstm_hidden_size', 64),
        n_lstm_layers=train_cfg.get('n_lstm_layers', 1),
        # Shared feature extractor for actor and critic
        shared_lstm=True,
        # Enable orthogonal initialization for stability
        enable_critic_lstm=True,
    )
    
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        # PPO hyperparameters
        learning_rate=train_cfg.get('learning_rate', 3e-4),
        n_steps=train_cfg.get('n_steps', 2048),
        batch_size=train_cfg.get('batch_size', 64),
        n_epochs=train_cfg.get('n_epochs', 10),
        gamma=train_cfg.get('gamma', 0.99),
        gae_lambda=train_cfg.get('gae_lambda', 0.95),
        clip_range=train_cfg.get('clip_range', 0.2),
        ent_coef=train_cfg.get('ent_coef', 0.01),
        vf_coef=train_cfg.get('vf_coef', 0.5),
        max_grad_norm=train_cfg.get('max_grad_norm', 0.5),
        # Logging
        tensorboard_log=tensorboard_log,
        verbose=train_cfg.get('verbose', 1),
        # CPU only (no CUDA)
        device='cpu',
        # Policy architecture
        policy_kwargs=policy_kwargs,
    )
    
    return model


def train(config_path: str, resume_from: Optional[str] = None):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML configuration file
        resume_from: Path to checkpoint to resume from (optional)
    """
    # =========================================================================
    # SETUP
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Solar EV RL Training - Recurrent PPO")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Setup directories
    dirs = setup_directories(config)
    logger.info(f"Model directory: {dirs['model_dir']}")
    logger.info(f"Log directory: {dirs['log_dir']}")
    
    # Save config to run directory
    config_save_path = os.path.join(dirs['model_dir'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    train_cfg = config.get('training', {})
    n_envs = train_cfg.get('n_envs', 4)
    
    logger.info(f"Creating {n_envs} vectorized environments...")
    
    # Create vectorized environment
    # DummyVecEnv runs environments sequentially (safe for CPU)
    env = DummyVecEnv([make_env(config, seed + i) for i in range(n_envs)])
    
    # Wrap with monitor for episode statistics
    env = VecMonitor(env)
    
    # Create separate evaluation environment
    eval_env = DummyVecEnv([make_env(config, seed + 1000)])
    eval_env = VecMonitor(eval_env)
    
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # =========================================================================
    # MODEL CREATION
    # =========================================================================
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = RecurrentPPO.load(resume_from, env=env, device='cpu')
    else:
        logger.info("Creating new RecurrentPPO model...")
        model = create_recurrent_ppo(env, config, dirs['log_dir'])
    
    # Log model architecture
    logger.info(f"Policy: {model.policy.__class__.__name__}")
    logger.info(f"LSTM hidden size: {train_cfg.get('lstm_hidden_size', 64)}")
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg.get('save_freq', 10000) // n_envs,
        save_path=dirs['model_dir'],
        name_prefix="ppo_lstm",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirs['model_dir'],
        log_path=dirs['log_dir'],
        eval_freq=train_cfg.get('eval_freq', 5000) // n_envs,
        n_eval_episodes=train_cfg.get('n_eval_episodes', 10),
        deterministic=True,
        render=False,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    total_timesteps = train_cfg.get('total_timesteps', 500000)
    
    logger.info("=" * 60)
    logger.info(f"Starting training for {total_timesteps:,} timesteps")
    logger.info("=" * 60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=train_cfg.get('log_interval', 10),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # =========================================================================
    # SAVE FINAL MODEL
    # =========================================================================
    final_model_path = os.path.join(dirs['model_dir'], "final_model")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    env.close()
    eval_env.close()
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    
    return model, dirs


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train RecurrentPPO agent for EV charging optimization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    # Run training
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
