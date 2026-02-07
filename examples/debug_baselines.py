"""
Debug script to understand why Conservative Deadline has worse undercharge rate.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import EVChargingEnv
from baselines import SolarFirstGreedy, ConservativeDeadline


def compare_baselines(n_episodes=100):
    """Compare baseline behaviors."""
    env = EVChargingEnv({
        'target_soc': 0.70,
        'base_hazard_rate': 0.02
    })
    
    print("="*70)
    print("🔍 Baseline Comparison Analysis")
    print("="*70)
    
    print(f"\nEnvironment Settings:")
    print(f"  Target SoC: {env.target_soc}")
    print(f"  Max Charging Power: {env.max_charging_power_kw} kW")
    
    # Test both baselines
    baselines = {
        'Solar-First Greedy': SolarFirstGreedy(),
        'Conservative Deadline': ConservativeDeadline(target_soc=env.target_soc)
    }
    
    results = {}
    
    for name, baseline in baselines.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        costs = []
        final_socs = []
        undercharged = 0
        episode_lengths = []
        actions_taken = {0: 0, 1: 0}  # Track action distribution
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_cost = 0.0
            episode_length = 0
            episode_actions = []
            
            while True:
                action = baseline.predict(obs)
                episode_actions.append(action)
                actions_taken[action] += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_cost -= reward  # Reward is negative cost
                episode_length += 1
                
                if terminated or truncated:
                    final_soc = obs[0]
                    final_socs.append(final_soc)
                    if final_soc < env.target_soc:
                        undercharged += 1
                    break
            
            costs.append(episode_cost)
            episode_lengths.append(episode_length)
        
        results[name] = {
            'avg_cost': np.mean(costs),
            'avg_soc': np.mean(final_socs),
            'undercharge_rate': undercharged / n_episodes,
            'avg_length': np.mean(episode_lengths),
            'action_dist': actions_taken.copy()
        }
        
        print(f"\nResults:")
        print(f"  Average Cost: ${np.mean(costs):.2f}")
        print(f"  Average Final SoC: {np.mean(final_socs):.3f}")
        print(f"  Undercharge Rate: {undercharged/n_episodes:.1%}")
        print(f"  Average Episode Length: {np.mean(episode_lengths):.1f} steps")
        print(f"  Action Distribution:")
        print(f"    Action 0 (Solar): {actions_taken[0]} ({actions_taken[0]/sum(actions_taken.values())*100:.1f}%)")
        print(f"    Action 1 (Grid): {actions_taken[1]} ({actions_taken[1]/sum(actions_taken.values())*100:.1f}%)")
    
    # Compare
    print(f"\n{'='*70}")
    print("📊 Comparison")
    print(f"{'='*70}")
    
    print(f"\n{'Baseline':<25} {'Avg SoC':<12} {'Undercharge':<15} {'Avg Cost':<12}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['avg_soc']:.3f}{'':<8} {metrics['undercharge_rate']:.1%}{'':<10} ${metrics['avg_cost']:.2f}")
    
    # Analyze the issue
    print(f"\n{'='*70}")
    print("🔍 Analysis")
    print(f"{'='*70}")
    
    solar_first = results['Solar-First Greedy']
    conservative = results['Conservative Deadline']
    
    if conservative['undercharge_rate'] > solar_first['undercharge_rate']:
        print("\n⚠️ ISSUE: Conservative Deadline has HIGHER undercharge rate!")
        print(f"  Solar-First: {solar_first['undercharge_rate']:.1%}")
        print(f"  Conservative: {conservative['undercharge_rate']:.1%}")
        
        print("\n💡 Possible Reasons:")
        print("  1. Conservative uses grid only, missing free solar during day")
        print("  2. Conservative may stop charging early if it reaches target")
        print("  3. Solar-First uses solar when available (free charging)")
        
        print(f"\n📈 Action Distribution:")
        print(f"  Solar-First:")
        print(f"    Solar: {solar_first['action_dist'][0]/sum(solar_first['action_dist'].values())*100:.1f}%")
        print(f"    Grid: {solar_first['action_dist'][1]/sum(solar_first['action_dist'].values())*100:.1f}%")
        print(f"  Conservative:")
        print(f"    Solar: {conservative['action_dist'][0]/sum(conservative['action_dist'].values())*100:.1f}%")
        print(f"    Grid: {conservative['action_dist'][1]/sum(conservative['action_dist'].values())*100:.1f}%")
        
        if conservative['action_dist'][0] < solar_first['action_dist'][0]:
            print("\n✅ CONFIRMED: Conservative uses less solar!")
            print("   This means it's paying for grid power even when solar is available.")
            print("   However, this should make it charge FASTER, not slower...")
        
        if conservative['avg_soc'] < solar_first['avg_soc']:
            print("\n⚠️ Conservative reaches LOWER final SoC!")
            print("   This suggests it's not charging fast enough.")
            print("   Possible bug: Conservative may be stopping too early?")


def trace_single_episode():
    """Trace a single episode to see what's happening."""
    print(f"\n{'='*70}")
    print("🔬 Single Episode Trace")
    print(f"{'='*70}")
    
    env = EVChargingEnv({
        'target_soc': 0.70,
        'base_hazard_rate': 0.02
    })
    
    baseline = ConservativeDeadline(target_soc=env.target_soc)
    
    obs, _ = env.reset()
    print(f"\nEpisode Start:")
    print(f"  Initial SoC: {env.current_soc:.3f}")
    print(f"  Target SoC: {env.target_soc:.3f}")
    print(f"  Baseline Target: {baseline.target_soc:.3f}")
    
    step = 0
    while step < 20:  # First 20 steps
        pv_current = obs[2]
        current_soc = obs[0]
        
        action = baseline.predict(obs)
        action_name = "Solar" if action == 0 else "Grid"
        
        print(f"\nStep {step}:")
        print(f"  SoC: {current_soc:.3f}, PV: {pv_current:.2f} kW")
        print(f"  Action: {action} ({action_name})")
        print(f"  Condition: SoC >= target? {current_soc >= baseline.target_soc}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  Final SoC: {obs[0]:.3f}")
            break
        
        step += 1


if __name__ == "__main__":
    compare_baselines(n_episodes=50)
    trace_single_episode()
