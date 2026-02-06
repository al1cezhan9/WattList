"""
ONNX Inference Verification Script for Solar-Arbitrage EV Controller.
Mimics how the Galaxy S25 (Snapdragon 8 Elite NPU) will execute the agent.
"""

import numpy as np
import onnxruntime as ort
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

try:
    from .environment import SolarArbitrageEnv
    from .config import load_config
except ImportError:
    from environment import SolarArbitrageEnv
    from config import load_config

logger = logging.getLogger(__name__)


class ONNXInferenceAgent:
    """
    ONNX-based inference agent for Solar-Arbitrage EV control.
    Compatible with Snapdragon 8 Elite NPU via onnxruntime-qnn.
    """
    
    def __init__(self, onnx_model_path: str):
        """
        Initialize ONNX inference agent.
        
        Args:
            onnx_model_path: Path to the ONNX model file
        """
        logger.info(f"Loading ONNX model from: {onnx_model_path}")
        
        # Configure providers for NPU acceleration
        providers = []
        
        # Try QNN provider first (for Snapdragon NPU)
        available_providers = ort.get_available_providers()
        if 'QNNExecutionProvider' in available_providers:
            try:
                providers.append('QNNExecutionProvider')
                logger.info("✅ QNN provider available for NPU acceleration")
            except Exception as e:
                logger.warning(f"Failed to initialize QNN provider: {e}")
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        # Create inference session
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Check which provider is actually being used
        actual_providers = self.session.get_providers()
        logger.info(f"Session created with providers: {actual_providers}")
        
        if 'QNNExecutionProvider' in actual_providers:
            logger.info("🚀 Using QNN (NPU) acceleration!")
        else:
            logger.info("⚠️ Using CPU execution (QNN not available)")
        
        # Verify input shape
        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Model output shape: {self.session.get_outputs()[0].shape}")
    
    def predict(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict action from observation.
        
        Args:
            observation: State array [solar_output, ev_soc, time_remaining, grid_price]
                        Shape: (4,) or (1, 4)
        
        Returns:
            action: Selected action (0 or 1)
            action_probs: Action probabilities array
        """
        # Ensure correct shape: (1, 4) for batch-1 inference
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        # Ensure float32 dtype
        observation = observation.astype(np.float32)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: observation})
        action_probs = outputs[0][0]  # Remove batch dimension
        
        # Select action with highest probability (deterministic)
        action = int(np.argmax(action_probs))
        
        return action, action_probs


def evaluate_onnx_agent(onnx_model_path: str, config=None, num_episodes: int = 1, 
                       render: bool = True) -> Dict:
    """
    Evaluate ONNX agent on Solar-Arbitrage environment.
    
    Args:
        onnx_model_path: Path to ONNX model
        config: Configuration dictionary
        num_episodes: Number of episodes to run
        render: Whether to plot results
    
    Returns:
        Dictionary with evaluation metrics and episode history
    """
    if config is None:
        config = load_config()
    
    logger.info("="*60)
    logger.info("🔍 ONNX Agent Evaluation")
    logger.info("="*60)
    
    # Create environment
    env = SolarArbitrageEnv(config)
    
    # Load ONNX agent
    agent = ONNXInferenceAgent(onnx_model_path)
    
    # Run episodes
    all_episodes = []
    
    for episode in range(num_episodes):
        logger.info(f"\nEpisode {episode + 1}/{num_episodes}")
        
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'solar_output': [],
            'ev_soc': [],
            'grid_price': [],
            'solar_power': [],
            'grid_power': [],
            'time_remaining': []
        }
        
        while not done:
            # Predict action
            action, action_probs = agent.predict(state)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store data
            episode_data['states'].append(state.copy())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['solar_output'].append(state[0])
            episode_data['ev_soc'].append(state[1])
            episode_data['time_remaining'].append(state[2])
            episode_data['grid_price'].append(state[3])
            episode_data['solar_power'].append(info.get('solar_power_kw', 0.0))
            episode_data['grid_power'].append(info.get('grid_power_kw', 0.0))
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        logger.info(f"Episode {episode + 1} completed:")
        logger.info(f"  Total reward: {episode_reward:.2f}")
        logger.info(f"  Episode length: {episode_length} hours")
        logger.info(f"  Final SoC: {episode_data['ev_soc'][-1]:.3f}")
        logger.info(f"  Total grid cost: ${sum([p * p for p, g in zip(episode_data['grid_price'], episode_data['grid_power'])]):.2f}")
        
        all_episodes.append(episode_data)
    
    # Plot results if requested
    if render and len(all_episodes) > 0:
        plot_evaluation_results(all_episodes[0], config)
    
    # Calculate summary statistics
    summary = {
        'num_episodes': num_episodes,
        'episodes': all_episodes,
        'avg_reward': np.mean([sum(ep['rewards']) for ep in all_episodes]),
        'avg_length': np.mean([len(ep['rewards']) for ep in all_episodes]),
        'avg_final_soc': np.mean([ep['ev_soc'][-1] for ep in all_episodes])
    }
    
    return summary


def plot_evaluation_results(episode_data: Dict, config=None):
    """
    Plot evaluation results showing Solar vs Grid Power Ratio and SoC over time.
    
    Args:
        episode_data: Dictionary with episode history data
        config: Configuration dictionary
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    hours = np.arange(len(episode_data['rewards']))
    
    # Plot 1: Solar vs Grid Power Ratio
    ax1 = axes[0]
    solar_power = np.array(episode_data['solar_power'])
    grid_power = np.array(episode_data['grid_power'])
    total_power = solar_power + grid_power
    
    # Calculate ratio (avoid division by zero)
    solar_ratio = np.where(total_power > 0, solar_power / total_power, 0.0)
    grid_ratio = np.where(total_power > 0, grid_power / total_power, 0.0)
    
    ax1.fill_between(hours, 0, solar_ratio, alpha=0.6, label='Solar Power Ratio', color='gold')
    ax1.fill_between(hours, solar_ratio, 1.0, alpha=0.6, label='Grid Power Ratio', color='steelblue')
    ax1.set_ylabel('Power Ratio', fontsize=12)
    ax1.set_title('Solar vs Grid Power Ratio Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SoC over time
    ax2 = axes[1]
    ax2.plot(hours, episode_data['ev_soc'], 'g-', linewidth=2, label='EV SoC')
    ax2.axhline(y=0.9, color='r', linestyle='--', linewidth=1.5, label='Target SoC (0.9)')
    ax2.set_ylabel('State of Charge', fontsize=12)
    ax2.set_title('EV Battery SoC Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Solar Output and Grid Price
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(hours, episode_data['solar_output'], 'orange', linewidth=2, label='Solar Output (kW)')
    line2 = ax3_twin.plot(hours, episode_data['grid_price'], 'purple', linewidth=2, linestyle='--', label='Grid Price ($/kWh)')
    
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_ylabel('Solar Output (kW)', fontsize=12, color='orange')
    ax3_twin.set_ylabel('Grid Price ($/kWh)', fontsize=12, color='purple')
    ax3.set_title('Solar Output and Grid Price Over Time', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('solar_arbitrage_evaluation.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Evaluation plot saved to: solar_arbitrage_evaluation.png")
    plt.show()


def single_prediction_test(onnx_model_path: str):
    """
    Run a single prediction test, mimicking Galaxy S25 execution.
    
    Args:
        onnx_model_path: Path to ONNX model
    """
    logger.info("="*60)
    logger.info("🧪 Single Prediction Test (Galaxy S25 Simulation)")
    logger.info("="*60)
    
    # Load agent
    agent = ONNXInferenceAgent(onnx_model_path)
    
    # Create a sample observation: [solar_output, ev_soc, time_remaining, grid_price]
    sample_observation = np.array([5.0, 0.5, 12.0, 0.20], dtype=np.float32)
    
    logger.info(f"Input observation: {sample_observation}")
    logger.info(f"  - Solar output: {sample_observation[0]:.2f} kW")
    logger.info(f"  - EV SoC: {sample_observation[1]:.3f}")
    logger.info(f"  - Time remaining: {sample_observation[2]:.1f} hours")
    logger.info(f"  - Grid price: ${sample_observation[3]:.2f}/kWh")
    
    # Run prediction
    action, action_probs = agent.predict(sample_observation)
    
    logger.info(f"\nPrediction results:")
    logger.info(f"  Action probabilities: {action_probs}")
    logger.info(f"  Selected action: {action} ({'Solar Only' if action == 0 else 'Solar + Grid'})")
    logger.info(f"  Confidence: {action_probs[action]:.3f}")
    
    logger.info("="*60)
    logger.info("✅ Single prediction test completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python inference_onnx.py <onnx_model_path> [--single-test] [--episodes N]")
        print("Example: python inference_onnx.py solar_agent.onnx --single-test")
        print("Example: python inference_onnx.py solar_agent.onnx --episodes 3")
        sys.exit(1)
    
    onnx_model_path = sys.argv[1]
    
    # Check for single test flag
    if '--single-test' in sys.argv:
        single_prediction_test(onnx_model_path)
    else:
        # Run evaluation
        num_episodes = 1
        if '--episodes' in sys.argv:
            idx = sys.argv.index('--episodes')
            if idx + 1 < len(sys.argv):
                num_episodes = int(sys.argv[idx + 1])
        
        config = load_config()
        results = evaluate_onnx_agent(onnx_model_path, config, num_episodes=num_episodes, render=True)
        
        logger.info("\n" + "="*60)
        logger.info("📊 Evaluation Summary")
        logger.info("="*60)
        logger.info(f"Average reward: {results['avg_reward']:.2f}")
        logger.info(f"Average episode length: {results['avg_length']:.1f} hours")
        logger.info(f"Average final SoC: {results['avg_final_soc']:.3f}")
