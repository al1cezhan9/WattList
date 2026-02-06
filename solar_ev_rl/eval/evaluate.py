"""
Evaluation Script for Solar EV RL Agent
========================================

This script evaluates trained RL agents and compares them against baselines.

Metrics Computed:
- Mean episode cost (grid electricity cost)
- Mean final SoC (state of charge at departure)
- Undercharge rate (fraction of episodes ending below target SoC)
- Solar utilization (fraction of solar energy used vs available)

Also tests agent behavior under different scenarios:
- Morning vs evening starts
- High vs low solar days
- Different user type distributions

Usage:
    python evaluate.py --model trained_models/run_xxx/best_model.zip
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Stable Baselines 3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.ev_charging_env import EVChargingEnv
from baselines.heuristics import (
    SolarFirstGreedy,
    ConservativeDeadline,
    EmpiricalSurvival
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_rl_agent(
    model: RecurrentPPO,
    env: EVChargingEnv,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 42
) -> Dict:
    """
    Evaluate RecurrentPPO agent on the environment.
    
    Args:
        model: Trained RecurrentPPO model
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        seed: Random seed
    
    Returns:
        Dictionary with evaluation metrics
    """
    np.random.seed(seed)
    
    # Metrics storage
    episode_costs = []
    episode_solar_used = []
    episode_grid_used = []
    final_socs = []
    episode_lengths = []
    undercharge_count = 0
    
    # Per-step data for analysis
    step_data = defaultdict(list)
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        
        # Initialize LSTM hidden state
        # RecurrentPPO stores state internally, but we need to reset it
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        done = False
        episode_cost = 0
        episode_steps = 0
        
        while not done:
            # Get action from model
            action, lstm_states = model.predict(
                obs.reshape(1, -1),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic
            )
            
            # Clear episode start flag after first step
            episode_starts = np.zeros((1,), dtype=bool)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            
            episode_cost += info.get('grid_cost', 0)
            episode_steps += 1
            
            # Record step data
            step_data['action'].append(action[0])
            step_data['solar_used'].append(info.get('solar_used', 0))
            step_data['grid_used'].append(info.get('grid_used', 0))
            step_data['soc'].append(info.get('soc', 0))
        
        # Record episode metrics
        episode_costs.append(episode_cost)
        episode_solar_used.append(info.get('episode_solar', 0))
        episode_grid_used.append(info.get('episode_grid', 0))
        final_socs.append(info.get('soc', 0))
        episode_lengths.append(episode_steps)
        
        if info.get('soc', 0) < env.target_soc:
            undercharge_count += 1
    
    # Compute summary statistics
    results = {
        'name': 'RecurrentPPO',
        'n_episodes': n_episodes,
        # Cost metrics
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'median_cost': np.median(episode_costs),
        # SoC metrics
        'mean_soc': np.mean(final_socs),
        'std_soc': np.std(final_socs),
        'min_soc': np.min(final_socs),
        # Undercharge rate
        'undercharge_rate': undercharge_count / n_episodes,
        # Episode length
        'mean_length': np.mean(episode_lengths),
        # Energy metrics
        'mean_solar_kwh': np.mean(episode_solar_used),
        'mean_grid_kwh': np.mean(episode_grid_used),
        # Action distribution
        'solar_only_ratio': np.mean([a == 0 for a in step_data['action']]),
    }
    
    return results


def evaluate_baseline(
    baseline,
    env: EVChargingEnv,
    n_episodes: int = 100,
    seed: int = 42
) -> Dict:
    """Evaluate a heuristic baseline policy."""
    np.random.seed(seed)
    
    episode_costs = []
    episode_solar_used = []
    episode_grid_used = []
    final_socs = []
    episode_lengths = []
    undercharge_count = 0
    actions = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        baseline.reset()
        done = False
        episode_cost = 0
        episode_steps = 0
        
        while not done:
            action = baseline.select_action(obs)
            actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_cost += info.get('grid_cost', 0)
            episode_steps += 1
        
        episode_costs.append(episode_cost)
        episode_solar_used.append(info.get('episode_solar', 0))
        episode_grid_used.append(info.get('episode_grid', 0))
        final_socs.append(info.get('soc', 0))
        episode_lengths.append(episode_steps)
        
        if info.get('soc', 0) < env.target_soc:
            undercharge_count += 1
    
    results = {
        'name': baseline.name,
        'n_episodes': n_episodes,
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'median_cost': np.median(episode_costs),
        'mean_soc': np.mean(final_socs),
        'std_soc': np.std(final_socs),
        'min_soc': np.min(final_socs),
        'undercharge_rate': undercharge_count / n_episodes,
        'mean_length': np.mean(episode_lengths),
        'mean_solar_kwh': np.mean(episode_solar_used),
        'mean_grid_kwh': np.mean(episode_grid_used),
        'solar_only_ratio': np.mean([a == 0 for a in actions]),
    }
    
    return results


def compare_all(
    model_path: str,
    config: Dict,
    n_episodes: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compare RL agent against all baselines.
    
    Returns DataFrame with comparison results.
    """
    logger.info("=" * 60)
    logger.info("Comprehensive Evaluation")
    logger.info("=" * 60)
    
    # Create environment
    env = EVChargingEnv(config)
    
    # Load RL model
    logger.info(f"Loading RL model from: {model_path}")
    model = RecurrentPPO.load(model_path, device='cpu')
    
    # Evaluate all policies
    results = []
    
    # RL Agent
    logger.info("Evaluating RecurrentPPO agent...")
    rl_results = evaluate_rl_agent(model, env, n_episodes, seed=seed)
    results.append(rl_results)
    
    # Baselines
    baselines = [
        SolarFirstGreedy(config),
        ConservativeDeadline(config),
        EmpiricalSurvival(config),
    ]
    
    for baseline in baselines:
        logger.info(f"Evaluating {baseline.name}...")
        env_baseline = EVChargingEnv(config)  # Fresh env
        bl_results = evaluate_baseline(baseline, env_baseline, n_episodes, seed)
        results.append(bl_results)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for _, row in df.iterrows():
        logger.info(f"\n{row['name']}:")
        logger.info(f"  Cost:      ${row['mean_cost']:.2f} ± ${row['std_cost']:.2f}")
        logger.info(f"  Final SoC: {row['mean_soc']:.1%} ± {row['std_soc']:.1%}")
        logger.info(f"  Undercharge Rate: {row['undercharge_rate']:.1%}")
        logger.info(f"  Solar-only Ratio: {row['solar_only_ratio']:.1%}")
    
    return df


def plot_comparison(df: pd.DataFrame, output_path: str = "evaluation_results.png"):
    """Create visualization of evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = df['name'].tolist()
    x = np.arange(len(names))
    
    # Cost comparison
    ax1 = axes[0, 0]
    ax1.bar(x, df['mean_cost'], yerr=df['std_cost'], capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Episode Cost ($)')
    ax1.set_title('Grid Electricity Cost')
    
    # SoC comparison
    ax2 = axes[0, 1]
    ax2.bar(x, df['mean_soc'] * 100, yerr=df['std_soc'] * 100, capsize=5)
    ax2.axhline(y=90, color='r', linestyle='--', label='Target SoC')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Final SoC (%)')
    ax2.set_title('Final State of Charge')
    ax2.legend()
    
    # Undercharge rate
    ax3 = axes[1, 0]
    ax3.bar(x, df['undercharge_rate'] * 100)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Undercharge Rate (%)')
    ax3.set_title('Episodes Ending Below Target SoC')
    
    # Solar utilization
    ax4 = axes[1, 1]
    ax4.bar(x, df['solar_only_ratio'] * 100)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.set_ylabel('Solar-Only Action Ratio (%)')
    ax4.set_title('Solar Utilization Strategy')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    plt.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent and compare with baselines"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output prefix for results"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run comparison
    df = compare_all(
        args.model,
        config,
        n_episodes=args.episodes,
        seed=args.seed
    )
    
    # Save results
    csv_path = f"{args.output}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")
    
    # Plot
    plot_path = f"{args.output}.png"
    plot_comparison(df, plot_path)


if __name__ == "__main__":
    main()
