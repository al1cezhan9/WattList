# EV Charging RL with Recurrent Policies

A reinforcement learning system for optimizing EV charging under solar uncertainty and unknown departure behavior. Trained on ARM64 Snapdragon CPU.

## 🎯 Overview

This project trains a recurrent LSTM policy (PPO) to optimize EV charging decisions:
- **Action 0**: Solar only charging (uses available solar power)
- **Action 1**: Grid only charging (uses grid power at max rate)

The agent learns to minimize charging costs while ensuring the EV reaches target SoC (70%) before departure.

## 📁 Project Structure

```
WattList/
├── envs/
│   └── ev_charging_env.py          # Gymnasium environment
├── baselines/
│   └── heuristics.py                # Baseline policies (Solar-First, Conservative)
├── models/
│   ├── train_recurrent_ppo.py      # Training script
│   └── export_onnx.py              # ONNX export
├── eval/
│   └── evaluate.py                 # Evaluation script
├── configs/
│   └── default.yaml                # Configuration (see comments for fast training options)
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train Model

```bash
# Standard training
python models/train_recurrent_ppo.py --config configs/default.yaml

# Fast training (for testing) - adjust parameters in default.yaml:
# - n_steps: 1024, n_epochs: 5, lstm_hidden_size: 32
# - total_timesteps: 200000, num_envs: 4
```

### 3. Evaluate

```bash
python eval/evaluate.py --model models/recurrent_ppo_ev_charging --episodes 100
```

### 4. Export to ONNX

```bash
python models/export_onnx.py --model models/recurrent_ppo_ev_charging --output models/ev_charging_policy.onnx
```

## 📊 Environment Details

### Observation Space (12D)

```
[soc, time_elapsed_norm, pv_current, 
 pv_forecast_mean (4), pv_forecast_std (4), 
 grid_price]
```

- **soc**: Current State of Charge (0.0 - 1.0)
- **time_elapsed_norm**: Normalized time elapsed (0.0 - 1.0)
- **pv_current**: Current PV output (0 - 10 kW)
- **pv_forecast_mean**: PV forecast mean for next 4 steps (0 - 10 kW each)
- **pv_forecast_std**: PV forecast uncertainty (0 - 10 kW each)
- **grid_price**: Current grid price ($/kWh, 0.10 - 0.30)

### Action Space

- **Action 0**: Solar only charging
  - Uses available solar power (up to 7.4 kW max)
  - No grid power used (free)
  - If no solar available, charging = 0

- **Action 1**: Grid only charging
  - Charges at maximum power (7.4 kW)
  - All power from grid (costs money)
  - Works day and night

### Charging Simulation

**Charging Power**:
- Solar only: `min(PV_output, 7.4 kW)`
- Grid only: `7.4 kW` (always max)

**SoC Update**:
```
energy_added = charging_power × 0.25 hours × 0.90 efficiency
soc_new = soc_old + (energy_added / 75.0 kWh)
```

**Key Parameters**:
- Battery capacity: 75 kWh
- Max charging power: 7.4 kW
- Charging efficiency: 90% (10% loss)
- Timestep: 15 minutes (0.25 hours)

### Reward Function

**Step Reward** (every timestep):
```
reward = -grid_energy × grid_price
```
- Solar only: `reward = $0.00` (no grid usage)
- Grid only: `reward = -grid_energy × price` (negative cost)

**Terminal Penalty** (on departure):
```
penalty = -lambda_penalty × max(0, target_soc - final_soc)
```
- Default: `lambda_penalty = 25.0`, `target_soc = 0.70`
- If SoC < 0.70 at departure: penalty applied
- If SoC >= 0.70: no penalty

**Total Reward**:
```
total_reward = step_reward + terminal_penalty
```

### Departure Model

**Single User Type**:
- Hazard rate: 0.02 per timestep (2% chance)
- Increases with time: `hazard × (1 + 2 × time_elapsed / max_steps)`
- Departure is stochastic and hidden (not observable)

**Why Single User Type?**
- Simplifies the problem
- RL learns adaptive policy based on observable signals (time, SoC)
- Doesn't need to classify user types

### PV (Solar) Model

**Generation**:
- Bell-shaped daily curve (peak at noon)
- AR(1) correlated noise (realistic forecast errors)
- Max output: 10 kW

**Forecast**:
- Provides mean and std for next 4 steps (1 hour ahead)
- Forecasts have 10% bias (overestimate)
- Uncertainty increases with current noise level

### Grid Pricing

**Time-of-Use (ToU) Schedule**:
- **Off-peak** (11 PM - 9 AM): $0.10/kWh
- **Mid** (9 AM - 4 PM, 9 PM - 11 PM): $0.20/kWh
- **Peak** (4 PM - 9 PM): $0.30/kWh

## 🧠 RL Policy Architecture

### RecurrentPPO with LSTM

- **Policy**: `MlpLstmPolicy` (LSTM + MLP)
- **Memory**: LSTM hidden state (default: 64 units)
- **Training**: CPU-only (ARM64 compatible)
- **Framework**: Stable Baselines 3 (sb3-contrib)

### What RL Learns

RL learns an **adaptive policy** that:
- Responds to observable signals (time elapsed, SoC, price)
- Balances cost minimization vs. urgency
- Works across different departure patterns
- Doesn't classify user types (responds to time pressure instead)

**Key Strategies Learned**:
- Avoid peak hours (4-9 PM) when possible
- Prefer off-peak hours (11 PM - 9 AM)
- Use solar when available (free)
- Charge aggressively when time is running out

## 📈 Baselines

Two baseline policies for comparison:

### 1. Solar-First Greedy
- Uses solar when available (action 0)
- Uses grid when no solar (action 1)
- No planning, just immediate optimization

### 2. Conservative Deadline
- Always charges at max power (action 1) until target SoC
- Assumes worst-case departure time
- Ensures target SoC but wastes money

## ⚙️ Configuration

### Key Parameters (`configs/default.yaml`)

**Environment**:
```yaml
target_soc: 0.70              # Target SoC (reduced from 0.90)
lambda_penalty: 25.0          # Terminal penalty weight (reduced)
base_hazard_rate: 0.02        # Departure hazard rate (reduced)
```

**Training**:
```yaml
total_timesteps: 500000       # Total training steps
num_envs: 1                   # Parallel environments
lstm_hidden_size: 64          # LSTM capacity
```

**Fast Training** (adjust parameters in `default.yaml`):
- Reduced timesteps: 200,000
- Smaller LSTM: 32 units
- Parallel environments: 4
- Faster updates: n_steps=1024, n_epochs=5

## 🔧 Training

### Standard Training

```bash
python models/train_recurrent_ppo.py --config configs/default.yaml
```

**Outputs**:
- Model checkpoints: `models/recurrent_ppo_ev_charging_*`
- TensorBoard logs: `./tensorboard_logs/`
- Final model: `models/recurrent_ppo_ev_charging.zip`

### Monitoring

```bash
tensorboard --logdir ./tensorboard_logs/
```

**Key Metrics**:
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `train/value_loss`: Value function loss (should decrease)
- `train/policy_loss`: Policy loss (should decrease)

## 📤 ONNX Export (Optional)

Export trained model to ONNX format for deployment:

```bash
python models/export_onnx.py \
    --model models/recurrent_ppo_ev_charging \
    --output models/ev_charging_policy.onnx
```

## 🐛 Troubleshooting

### High Undercharge Rate

**Problem**: EVs departing before reaching target SoC

**Solution**: Already fixed in configs:
- Reduced hazard rate: `0.02` (was `0.05-0.15`)
- Reduced target SoC: `0.70` (was `0.90`)
- Reduced penalty: `25.0` (was `100.0`)

### Slow Training

**Solutions**:
- Adjust parameters in `default.yaml` (see comments for fast training options)
- Increase `num_envs` for parallel training
- Reduce `total_timesteps` for testing

### Poor RL Performance

**Check**:
1. Training long enough? (500k+ timesteps)
2. Reward scaling balanced? (step vs terminal)
3. Exploration sufficient? (check `ent_coef`)

## 📚 Technical Details

### Why Recurrent Policy?

- **Temporal dependencies**: Current decision affects future
- **Partial observability**: Departure time is hidden
- **LSTM memory**: Tracks patterns over time

### Reward Design

- **Step reward**: Negative cost (minimize grid usage)
- **Terminal penalty**: Encourages reaching target SoC
- **Balanced**: Reduced penalty (25.0) balances cost vs. urgency

### Departure Model

- **Single user type**: Simplified from 3 types
- **Hazard-based**: Probability increases with time
- **Hidden**: Not observable, agent learns from outcomes

### PV Forecasts

- **AR(1) noise**: Correlated errors (realistic)
- **Bias**: Forecasts overestimate by 10%
- **Uncertainty**: Provided as forecast std

## 📝 Requirements

- Python 3.10+
- PyTorch (CPU)
- stable-baselines3 >= 2.0.0
- sb3-contrib >= 2.0.0
- gymnasium >= 0.29.0
- onnxruntime >= 1.17.0

See `requirements.txt` for complete list.

## 🎓 Usage Examples

### Custom Training

```bash
python models/train_recurrent_ppo.py --config my_config.yaml
```

### Evaluation

```bash
# Compare all policies
python eval/evaluate.py --episodes 200

# Evaluate specific model
python eval/evaluate.py --model models/recurrent_ppo_ev_charging --episodes 100
```

### ONNX Export (Optional)

```bash
python models/export_onnx.py \
    --model models/recurrent_ppo_ev_charging \
    --output models/ev_charging_policy.onnx \
    --config configs/default.yaml
```

## 🧪 Testing

Run tests to verify the environment and baselines:

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/test_environment.py tests/test_baselines.py tests/test_training.py

# Run with coverage
pytest tests/ --cov=envs --cov=baselines --cov-report=html
```

See `TESTING_GUIDE.md` for detailed testing instructions.

## 📄 License

See LICENSE file.
