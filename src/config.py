import yaml
import os


def load_config():
    """Load configuration from YAML file or return defaults."""
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration if file doesn't exist (Solar-Arbitrage EV Controller)
        return {
            'environment': {
                'ev_capacity_kwh': 75.0,
                'charging_efficiency': 0.90,
                'max_charging_power_kw': 7.4,
                'max_solar_output_kw': 10.0,
                'solar_noise_std': 0.5,
                'tou_prices': {'peak': 0.30, 'mid': 0.20, 'off_peak': 0.10},
                'max_hours': 24,
                'target_soc': 0.9,
                'departure_penalty_scale': 100.0,
                'solar_reward_weight': 1.5,
                'grid_cost_weight': 1.0
            },
            'network': {'input_dim': 4, 'hidden_dim': 128, 'output_dim': 2},
            'ppo': {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01, 'vf_coef': 0.5},
            'training': {'total_timesteps': 100000, 'model_save_path': 'models/solar_agent', 'onnx_export_path': 'solar_agent.onnx', 'simulation_speed': 0.05, 'reward_history_length': 1000, 'episode_history_length': 100},
            'server': {'host': '0.0.0.0', 'port': 8080, 'debug': False},
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(message)s', 'episode_summary_frequency': 10, 'log_file': 'training.log'}
        }
