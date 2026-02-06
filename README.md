# EV Charging RL with Recurrent Policies

A reinforcement learning system for optimizing EV charging under solar uncertainty and unknown, non-stationary departure behavior. Trained on ARM64 Snapdragon CPU and deployed to Android devices via ONNX.

## 🎯 Key Features

- **Recurrent LSTM Policy**: Handles temporal dependencies and partial observability
- **Hidden Departure Model**: Learns from non-stationary, unknown departure patterns
- **PV Forecasts**: Uses imperfect forecasts for better planning
- **CPU-Only Training**: Optimized for ARM64 Snapdragon processors
- **Mobile Deployment**: ONNX export with LSTM state management for Android

## 📁 Project Structure

```
solar_ev_rl/
├── envs/
│   └── ev_charging_env.py          # Recurrent environment with hidden departure
├── baselines/
│   └── heuristics.py                # Baseline policies for comparison
├── models/
│   ├── train_recurrent_ppo.py      # RecurrentPPO training
│   └── export_onnx.py              # ONNX export for LSTM
├── eval/
│   └── evaluate.py                 # Evaluation and comparison
├── android/
│   └── inference_example.py       # Android ONNX inference
├── configs/
│   └── default.yaml                # Configuration
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Recurrent PPO

```bash
python models/train_recurrent_ppo.py --config configs/default.yaml
```

This will:
- Train a RecurrentPPO agent with LSTM policy
- Save checkpoints every 50k timesteps
- Evaluate periodically
- Save final model to `models/recurrent_ppo_ev_charging`

### 3. Evaluate Policy

```bash
python eval/evaluate.py --model models/recurrent_ppo_ev_charging --episodes 100
```

Compares RL policy with baselines:
- Solar-First Greedy
- Conservative Deadline
- Empirical Survival

### 4. Export to ONNX

```bash
python models/export_onnx.py --model models/recurrent_ppo_ev_charging --output models/ev_charging_policy.onnx
```

### 5. Test Android Inference

```bash
python android/inference_example.py
```

## 🔧 Configuration

Edit `configs/default.yaml` to customize:

- **Environment**: PV parameters, departure model, pricing
- **Training**: Hyperparameters, timesteps, logging
- **LSTM**: Hidden size, layers
- **ONNX**: Export settings

## 📊 Environment Details

### Observation Space (12D)

```
[soc, time_elapsed_norm, pv_current, 
 pv_forecast_mean (4), pv_forecast_std (4), 
 grid_price]
```

### Action Space

- **0**: Solar only charging
- **1**: Solar + grid charging (max power)

### Key Features

- **15-minute timesteps** (vs 1-hour in simple version)
- **PV forecasts** with AR(1) correlated noise
- **Hidden departure** (hazard-based, non-stationary)
- **Time-of-use pricing** (peak/mid/off-peak)

## 🧠 Policy Architecture

- **Type**: RecurrentPPO with LSTM
- **Policy**: MlpLstmPolicy
- **Memory**: LSTM hidden state (configurable, default 64)
- **Training**: CPU-only (ARM64 compatible)

## 📱 Android Deployment

See `android/inference_example.py` for complete example:

```python
from android.inference_example import EVChargingInference

# Initialize
inference = EVChargingInference("models/ev_charging_policy.onnx")

# Reset for new episode
inference.reset()

# Run inference
action, probs = inference.predict(observation)
```

**Key Points:**
- LSTM state must be reset at episode start
- State persists across timesteps within episode
- NNAPI provider used for NPU acceleration (if available)

## 📈 Baselines

Three baseline policies included:

1. **SolarFirstGreedy**: Always use solar when available
2. **ConservativeDeadline**: Always charge at max power
3. **EmpiricalSurvival**: Tracks historical departures

Compare these with the learned RL policy using `eval/evaluate.py`.

## 🔬 Technical Details

### Hidden Departure Model

- **User types**: 3 types with different hazard rates
- **Non-stationary**: Distribution drifts during training
- **Hazard-based**: Probability increases with time
- **Not observable**: Agent learns from interaction only

### PV Forecast Model

- **AR(1) noise**: Correlated forecast errors
- **Bias**: Forecasts systematically overestimate
- **Horizon**: 4 steps (1 hour) ahead
- **Uncertainty**: Forecast std provided

## 📝 Requirements

- Python 3.10+
- PyTorch (CPU)
- stable-baselines3 >= 2.0.0
- sb3-contrib >= 2.0.0
- gymnasium >= 0.29.0
- onnxruntime >= 1.17.0

## 🎓 Usage Examples

### Training with Custom Config

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

### ONNX Export

```bash
python models/export_onnx.py \
    --model models/recurrent_ppo_ev_charging \
    --output models/ev_charging_policy.onnx \
    --config configs/default.yaml
```

## 🐛 Troubleshooting

### Training Issues

- **Memory**: Reduce `n_steps` or `batch_size` in config
- **Slow training**: Normal on CPU, consider reducing `total_timesteps` for testing

### ONNX Export Issues

- **LSTM state**: Ensure `lstm_hidden_size` matches training config
- **Shape mismatch**: Check observation dimension (should be 12)

### Android Inference Issues

- **State management**: Always reset LSTM state at episode start
- **NNAPI**: Falls back to CPU if NPU not available

## 📚 Documentation

- `TRANSFORMATION_SUMMARY.md`: Detailed changes from old codebase
- `configs/default.yaml`: Comprehensive configuration reference
- Code comments: Inline documentation throughout

## 🔄 Migration from Old Codebase

⚠️ **Breaking Changes**: This is a complete rewrite. Old code/models not compatible.

See `TRANSFORMATION_SUMMARY.md` for migration guide.

## 📄 License

See LICENSE file.
