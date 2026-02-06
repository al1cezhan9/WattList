# Solar EV RL - Reinforcement Learning for Smart EV Charging

A complete reinforcement learning system for optimizing EV charging under solar uncertainty and unknown departure behavior. Designed to train on ARM-based Qualcomm Snapdragon computers and run inference on Samsung Android phones.

## 🎯 Problem Overview

The agent learns to decide when to use:
- **Solar-only charging** (Action 0): Free but limited by PV generation
- **Solar + Grid charging** (Action 1): Reliable but costs money

### Key Challenge: Unknown Departures
The EV departure time follows an **unknown, non-stationary process**. The agent must:
1. Learn departure patterns implicitly from experience
2. Adapt to drifting user behavior over time
3. Balance cost minimization with charging reliability

### Why Recurrent Policy (LSTM)?
The departure probability is NOT in the observation. The LSTM allows the agent to:
- Maintain memory of the episode history
- Infer hidden user type from patterns
- Make decisions under epistemic uncertainty

## 📁 Project Structure

```
solar_ev_rl/
├── envs/
│   └── ev_charging_env.py      # Gymnasium environment
├── baselines/
│   └── heuristics.py           # Baseline comparison policies
├── models/
│   ├── train_recurrent_ppo.py  # Training script (LSTM-PPO)
│   └── export_onnx.py          # ONNX export for mobile
├── eval/
│   └── evaluate.py             # Evaluation and comparison
├── android/
│   └── inference_example.py    # Mobile inference demo
├── configs/
│   └── default.yaml            # Configuration file
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd solar_ev_rl
pip install -r requirements.txt
```

### 2. Test Environment

```bash
python -m envs.ev_charging_env
```

### 3. Train Agent

```bash
python -m models.train_recurrent_ppo --config configs/default.yaml
```

Training will:
- Create vectorized environments
- Train RecurrentPPO with LSTM policy
- Save checkpoints and best model
- Log to TensorBoard

### 4. Export to ONNX

```bash
python -m models.export_onnx \
    --model trained_models/run_xxx/best_model.zip \
    --output exported_models/solar_ev_policy.onnx
```

### 5. Evaluate

```bash
python -m eval.evaluate \
    --model trained_models/run_xxx/best_model.zip \
    --episodes 100
```

### 6. Android Inference Demo

```bash
# Simulate inference
python -m android.inference_example --model exported_models/solar_ev_policy.onnx

# Show Android/Kotlin integration code
python -m android.inference_example --show-android
```

## 🔧 Environment Details

### Observation Space (12-dimensional)

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | current_soc | 0-1 | Battery state of charge |
| 1 | time_normalized | 0-1 | Elapsed time fraction |
| 2 | current_pv | 0-10 kW | Current solar generation |
| 3-6 | pv_forecast_mean | 0-10 kW | Mean PV forecast (4 steps) |
| 7-10 | pv_forecast_std | 0-10 kW | Forecast uncertainty |
| 11 | grid_price | 0-1 | Normalized electricity price |

**Note:** Departure probability is NOT included - the agent must learn it!

### Action Space (Discrete, 2 actions)

- **Action 0 (Solar Only)**: Charge at `min(solar_available, 7.4 kW)`, cost = $0
- **Action 1 (Solar + Grid)**: Charge at 7.4 kW, grid fills the gap

### Reward Function

```
R_step = -grid_cost_weight * (grid_energy * price) + solar_bonus * solar_energy
R_terminal = -λ * max(0, target_soc - final_soc)
```

### Departure Model (Hidden from agent!)

- Episodes have a hidden "user type" sampled from a drifting distribution
- Each user type has different departure hazard rates
- Hazard increases with time elapsed
- Distribution shifts sinusoidally over training (non-stationary)

## 📊 Baselines

Three heuristic baselines for comparison:

1. **SolarFirstGreedy**: Always use solar unless SoC is critical
2. **ConservativeDeadline**: Assume early deadline, charge aggressively
3. **EmpiricalSurvival**: Track historical departures, adapt strategy

## 📱 Android Deployment

The exported ONNX model can be deployed on Samsung Android using ONNX Runtime Mobile.

Key integration points:
1. Reset LSTM state when EV plugs in
2. Pass observation + state to get action
3. Carry LSTM state to next timestep
4. Use 15-minute decision intervals

See `android/inference_example.py --show-android` for complete Kotlin code.

## ⚙️ Configuration

All parameters are in `configs/default.yaml`:

```yaml
environment:
  battery_capacity_kwh: 75.0
  max_charging_power_kw: 7.4
  target_soc: 0.90

departure:
  num_user_types: 3
  drift_rate: 0.0001

training:
  total_timesteps: 500000
  lstm_hidden_size: 64
```

## 🔬 Technical Design Decisions

### Why RecurrentPPO over PPO?
Standard PPO assumes Markovian observations. Our environment has:
- Hidden user type (not observable)
- Time-dependent hazard (must be inferred)
- Non-stationary dynamics

The LSTM enables the agent to build an internal model of these latent factors.

### Why CPU-only training?
- Designed for ARM64 Snapdragon laptops (no NVIDIA GPU)
- RecurrentPPO with LSTM is not highly parallelizable
- Inference on mobile is CPU anyway

### Why fixed ONNX input shapes?
- ONNX Runtime Mobile works best with fixed shapes
- Batch size = 1 for real-time inference
- Explicit LSTM state handling for stateful inference

## 📈 Expected Results

After training, the RL agent should:
- Achieve lower cost than solar-first baseline
- Have lower undercharge rate than conservative baseline
- Adapt to changing user behavior over time

Typical metrics:
- Mean cost: $0.30-0.50 per episode
- Final SoC: 88-92%
- Undercharge rate: <15%

## 🐛 Troubleshooting

**Training is slow:**
- Reduce `n_envs` in config
- Reduce `n_steps` for faster updates

**ONNX export fails:**
- Ensure model was trained with latest sb3-contrib
- Check opset version compatibility

**Android inference crashes:**
- Verify input tensor shapes match exactly
- Ensure LSTM state is properly initialized

## 📚 References

- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [sb3-contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
