"""
Evaluate trained Recurrent PPO policy and compare with baselines.
"""

import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import EVChargingEnv
from baselines import SolarFirstGreedy, ConservativeDeadline, EmpiricalSurvival


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_policy(env, model, n_episodes: int = 100, deterministic: bool = True):
    """
    Evaluate a policy (RL model or baseline).
    
    Returns:
        Dictionary with metrics: avg_cost, avg_final_soc, undercharge_rate, etc.
    """
    costs = []
    final_socs = []
    episode_lengths = []
    undercharged = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_cost = 0.0
        episode_length = 0
        
        # For recurrent policies, need to track LSTM state
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        while not done:
            # Get action
            if isinstance(model, (PPO, RecurrentPPO)):
                if isinstance(model, RecurrentPPO):
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=deterministic
                    )
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # Baseline heuristic
                action = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_cost -= reward  # Reward is negative cost
            episode_length += 1
            episode_starts[0] = done
            
            # Update baseline if needed
            if isinstance(model, EmpiricalSurvival) and done:
                model.update_departure(episode_length)
        
        costs.append(episode_cost)
        final_socs.append(info['soc'])
        episode_lengths.append(episode_length)
        
        if info['soc'] < env.target_soc:
            undercharged += 1
    
    return {
        'avg_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'avg_final_soc': np.mean(final_socs),
        'std_final_soc': np.std(final_socs),
        'undercharge_rate': undercharged / n_episodes,
        'avg_episode_length': np.mean(episode_lengths),
    }


def main():
    """Run evaluation comparing RL policy with baselines."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate EV Charging Policies')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained RL model (optional)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    env_config = config['environment']
    eval_config = config.get('evaluation', {})
    n_episodes = args.episodes or eval_config.get('n_episodes', 100)
    
    print("="*60)
    print("📊 Evaluating EV Charging Policies")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print("="*60)
    
    # Create environment
    env = EVChargingEnv(env_config)
    
    results = {}
    
    # Evaluate baselines
    print("\n1. Evaluating Solar-First Greedy Baseline...")
    baseline1 = SolarFirstGreedy(env_config['max_charging_power_kw'])
    results['Solar-First Greedy'] = evaluate_policy(env, baseline1, n_episodes)
    
    print("\n2. Evaluating Conservative Deadline Baseline...")
    baseline2 = ConservativeDeadline(
        env_config['target_soc'],
        env_config['max_charging_power_kw']
    )
    results['Conservative Deadline'] = evaluate_policy(env, baseline2, n_episodes)
    
    print("\n3. Evaluating Empirical Survival Baseline...")
    baseline3 = EmpiricalSurvival(
        env_config['target_soc'],
        env_config['max_charging_power_kw']
    )
    results['Empirical Survival'] = evaluate_policy(env, baseline3, n_episodes)
    
    # Evaluate RL policy if provided
    if args.model and os.path.exists(args.model):
        print(f"\n4. Evaluating Recurrent PPO Policy ({args.model})...")
        try:
            rl_model = RecurrentPPO.load(args.model, device='cpu')
            results['Recurrent PPO'] = evaluate_policy(env, rl_model, n_episodes, deterministic=True)
        except Exception as e:
            print(f"⚠️ Failed to load RL model: {e}")
    
    # Print results
    print("\n" + "="*60)
    print("📊 Evaluation Results")
    print("="*60)
    
    for policy_name, metrics in results.items():
        print(f"\n{policy_name}:")
        print(f"  Average Cost: ${metrics['avg_cost']:.2f} ± ${metrics['std_cost']:.2f}")
        print(f"  Average Final SoC: {metrics['avg_final_soc']:.3f} ± {metrics['std_final_soc']:.3f}")
        print(f"  Undercharge Rate: {metrics['undercharge_rate']:.1%}")
        print(f"  Avg Episode Length: {metrics['avg_episode_length']:.1f} steps")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
