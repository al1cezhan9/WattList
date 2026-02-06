#!/usr/bin/env python
"""
Quick Test Script for Solar EV RL
=================================

Run this to verify everything is set up correctly before training.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_environment():
    """Test environment creation and rollout."""
    print("=" * 60)
    print("Testing EVChargingEnv")
    print("=" * 60)
    
    from envs.ev_charging_env import EVChargingEnv
    
    env = EVChargingEnv()
    obs, info = env.reset(seed=42)
    
    print(f"✓ Environment created")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Run a few steps
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break
    
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final SoC: {info['soc']:.1%}")
    print(f"✓ Environment rollout successful\n")
    return True


def test_baselines():
    """Test baseline heuristics."""
    print("=" * 60)
    print("Testing Baselines")
    print("=" * 60)
    
    from envs.ev_charging_env import EVChargingEnv
    from baselines.heuristics import SolarFirstGreedy, ConservativeDeadline, EmpiricalSurvival
    
    env = EVChargingEnv()
    
    baselines = [
        SolarFirstGreedy(),
        ConservativeDeadline(),
        EmpiricalSurvival(),
    ]
    
    for baseline in baselines:
        obs, _ = env.reset(seed=42)
        baseline.reset()
        
        total_reward = 0
        for _ in range(50):
            action = baseline.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"✓ {baseline.name}: reward={total_reward:.2f}, soc={info['soc']:.1%}")
    
    print("✓ All baselines work correctly\n")
    return True


def test_vectorized_env():
    """Test vectorized environment for training."""
    print("=" * 60)
    print("Testing Vectorized Environment")
    print("=" * 60)
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    from envs.ev_charging_env import EVChargingEnv
    
    def make_env(seed):
        def _init():
            env = EVChargingEnv()
            env.reset(seed=seed)
            return env
        return _init
    
    n_envs = 4
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    print(f"✓ Created {n_envs} vectorized environments")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test step
    obs = env.reset()
    print(f"  Reset observation shape: {obs.shape}")
    
    actions = [env.action_space.sample() for _ in range(n_envs)]
    obs, rewards, dones, infos = env.step(actions)
    
    print(f"  Step observation shape: {obs.shape}")
    print(f"  Rewards: {rewards}")
    
    env.close()
    print("✓ Vectorized environment works correctly\n")
    return True


def test_recurrent_ppo_creation():
    """Test RecurrentPPO model creation."""
    print("=" * 60)
    print("Testing RecurrentPPO Creation")
    print("=" * 60)
    
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import RecurrentPPO
    from envs.ev_charging_env import EVChargingEnv
    
    # Set CPU
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: CPU (CUDA not required)")
    
    def make_env(seed):
        def _init():
            env = EVChargingEnv()
            env.reset(seed=seed)
            return env
        return _init
    
    env = DummyVecEnv([make_env(0)])
    
    # Create minimal model for testing
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,  # Small for quick test
        batch_size=32,
        n_epochs=2,
        verbose=0,
        device='cpu',
    )
    
    print(f"✓ RecurrentPPO model created")
    print(f"  Policy: {model.policy.__class__.__name__}")
    
    # Test prediction
    obs = env.reset()
    lstm_states = None
    episode_starts = [True]
    
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
    print(f"  Test prediction: action={action}")
    
    env.close()
    print("✓ RecurrentPPO works correctly\n")
    return True


def test_config_loading():
    """Test configuration loading."""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    import yaml
    
    config_path = "configs/default.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded config from: {config_path}")
    print(f"  Environment settings:")
    print(f"    - Battery capacity: {config['environment']['battery_capacity_kwh']} kWh")
    print(f"    - Target SoC: {config['environment']['target_soc']}")
    print(f"  Training settings:")
    print(f"    - Total timesteps: {config['training']['total_timesteps']}")
    print(f"    - LSTM hidden size: {config['training']['lstm_hidden_size']}")
    print("✓ Configuration valid\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Solar EV RL - Quick Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Environment", test_environment),
        ("Baselines", test_baselines),
        ("Vectorized Env", test_vectorized_env),
        ("Configuration", test_config_loading),
        ("RecurrentPPO", test_recurrent_ppo_creation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} FAILED: {e}\n")
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if not success:
            all_passed = False
            if error:
                print(f"         Error: {error}")
    
    print()
    if all_passed:
        print("All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. python -m models.train_recurrent_ppo --config configs/default.yaml")
        print("  2. python -m models.export_onnx --model trained_models/run_xxx/best_model.zip")
        print("  3. python -m android.inference_example --model exported_models/solar_ev_policy.onnx")
    else:
        print("Some tests failed. Please fix the issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
