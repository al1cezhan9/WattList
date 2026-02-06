"""
Stable Baselines 3 training script for Solar-Arbitrage EV Controller.
Includes ONNX export functionality for Snapdragon 8 Elite NPU deployment.
"""

import os
import logging
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

try:
    from .environment import SolarArbitrageEnv
    from .config import load_config
except ImportError:
    from environment import SolarArbitrageEnv
    from config import load_config

logger = logging.getLogger(__name__)


class OnnxablePolicy(nn.Module):
    """
    ONNX-compatible policy wrapper for SB3 PPO.
    Removes broadcast layers and ensures fixed input shape (1, 4) for NPU inference.
    Based on SB3 documentation for ONNX export.
    """
    
    def __init__(self, policy):
        super(OnnxablePolicy, self).__init__()
        
        # Extract the policy network components
        # For MlpPolicy, the structure is: features_extractor -> mlp_extractor -> action_net
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
    
    def forward(self, observation):
        """
        Forward pass compatible with ONNX export.
        Input: (batch_size, 4) - [solar_output, ev_soc, time_remaining, grid_price]
        Output: action probabilities (batch_size, 2)
        """
        # Extract features (for MlpPolicy, this is usually identity or FlattenExtractor)
        features = self.features_extractor(observation)
        
        # Pass through MLP extractor to get actor latent representation
        # forward_actor returns the latent_pi (actor features)
        latent_pi = self.mlp_extractor.forward_actor(features)
        
        # Get action logits from action net
        action_logits = self.action_net(latent_pi)
        
        # Apply softmax to get probabilities
        action_probs = torch.softmax(action_logits, dim=-1)
        
        return action_probs


def create_onnxable_policy(model):
    """
    Create an ONNX-compatible policy from a trained SB3 PPO model.
    
    Args:
        model: Trained PPO model from stable_baselines3
    
    Returns:
        OnnxablePolicy instance
    """
    # Get the policy network
    policy_net = model.policy
    
    # Create ONNX-compatible wrapper
    onnxable_policy = OnnxablePolicy(policy_net)
    onnxable_policy.eval()
    
    return onnxable_policy


def export_to_onnx(model, output_path='solar_agent.onnx', input_shape=(1, 4)):
    """
    Export trained SB3 PPO model to ONNX format for Snapdragon 8 Elite NPU.
    
    Args:
        model: Trained PPO model from stable_baselines3
        output_path: Path to save the ONNX model
        input_shape: Fixed input shape (batch_size, state_dim) for NPU inference
    """
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    # Create ONNX-compatible policy
    onnxable_policy = create_onnxable_policy(model)
    
    # Create dummy input with fixed shape (1, 4) for batch-1 inference
    dummy_input = torch.randn(input_shape, dtype=torch.float32)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            onnxable_policy,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,  # Compatible with onnxruntime-qnn
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action_probs'],
            dynamic_axes=None,  # Fixed shape for NPU compatibility
            verbose=False
        )
        
        logger.info(f"✅ ONNX model exported successfully to {output_path}")
        logger.info(f"   Input shape: {input_shape}")
        logger.info(f"   Output: action probabilities (batch_size, 2)")
        
        # Verify the exported model
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
            
            # Test inference
            test_input = np.random.randn(*input_shape).astype(np.float32)
            outputs = session.run(None, {'observation': test_input})
            
            logger.info(f"✅ ONNX model verification successful")
            logger.info(f"   Output shape: {outputs[0].shape}")
            
        except ImportError:
            logger.warning("onnxruntime not available for verification")
        except Exception as e:
            logger.warning(f"ONNX verification failed: {e}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        raise


def train_solar_arbitrage_agent(config=None, total_timesteps=100000, 
                                model_save_path='models/solar_agent', 
                                onnx_export_path='solar_agent.onnx'):
    """
    Train a PPO agent for Solar-Arbitrage EV control using Stable Baselines 3.
    
    Args:
        config: Configuration dictionary (optional)
        total_timesteps: Total training timesteps
        model_save_path: Path to save the trained model
        onnx_export_path: Path to save the ONNX model
    """
    if config is None:
        config = load_config()
    
    logger.info("="*60)
    logger.info("🚀 Solar-Arbitrage EV Controller Training")
    logger.info("="*60)
    
    # Create environment
    def make_env():
        return SolarArbitrageEnv(config)
    
    # Create vectorized environment (single env for now)
    env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    logger.info("Environment created:")
    logger.info(f"  Observation space: {env.observation_space}")
    logger.info(f"  Action space: {env.action_space}")
    
    # PPO hyperparameters from config
    ppo_config = config.get('ppo', {})
    
    # Create PPO agent with MlpPolicy
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        verbose=1,
        tensorboard_log='./tensorboard_logs/'
    )
    
    logger.info("PPO agent created with MlpPolicy")
    logger.info(f"Training for {total_timesteps} timesteps...")
    
    # Setup callbacks
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.',
        name_prefix='solar_agent'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save the final model
    logger.info(f"Saving final model to {model_save_path}...")
    model.save(model_save_path)
    logger.info("✅ Model saved successfully")
    
    # Export to ONNX
    logger.info("Exporting model to ONNX format...")
    export_to_onnx(model, onnx_export_path, input_shape=(1, 4))
    
    logger.info("="*60)
    logger.info("✅ Training complete!")
    logger.info(f"   Model saved: {model_save_path}")
    logger.info(f"   ONNX model: {onnx_export_path}")
    logger.info("="*60)
    
    return model


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config()
    
    # Training parameters
    total_timesteps = config.get('training', {}).get('total_timesteps', 100000)
    model_save_path = config.get('training', {}).get('model_save_path', 'models/solar_agent')
    onnx_export_path = config.get('training', {}).get('onnx_export_path', 'solar_agent.onnx')
    
    # Train
    train_solar_arbitrage_agent(
        config=config,
        total_timesteps=total_timesteps,
        model_save_path=model_save_path,
        onnx_export_path=onnx_export_path
    )
