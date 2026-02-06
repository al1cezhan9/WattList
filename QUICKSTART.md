# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python -m src.sb3_training
```

This will:
- Train a PPO agent for ~100k timesteps
- Save model to `models/solar_agent.zip`
- Export ONNX model to `solar_agent.onnx`

**Time**: ~30-60 minutes (depending on hardware)

### 3. Test the Model

```bash
# Quick test
python -m src.inference_onnx solar_agent.onnx --single-test

# Full evaluation with plots
python -m src.inference_onnx solar_agent.onnx --episodes 3
```

## 📋 What You Need to Know

### The Problem
An EV needs to be charged before departure. You have:
- **Solar panels** (free energy, variable output)
- **Grid power** (costs money, Time-of-Use pricing)
- **Limited time** (hours until departure)

### The Solution
An RL agent learns to:
- ✅ Use free solar when available
- ✅ Avoid expensive peak-hour grid power
- ✅ Ensure 90% charge before departure

### The Environment

**State (4 numbers):**
- Solar output (0-10 kW)
- EV battery level (0.0-1.0)
- Time remaining (0-24 hours)
- Grid price ($/kWh)

**Actions (2 choices):**
- **0**: Charge with solar only (free)
- **1**: Charge at max power (solar + grid)

**Reward:**
- Positive for using solar
- Negative for grid costs
- Big penalty if not charged enough at departure

## 🎯 Example Usage

### Training with Custom Config

```bash
# Edit config.yaml first, then:
python -m src.sb3_training
```

### Monitor Training

```bash
# In another terminal:
tensorboard --logdir=./tensorboard_logs
# Open http://localhost:6006
```

### Deploy to Device

```python
# On Snapdragon device:
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    'solar_agent.onnx',
    providers=['QNNExecutionProvider']
)

# Observation: [solar, soc, time, price]
obs = np.array([[5.0, 0.5, 12.0, 0.20]], dtype=np.float32)
probs = session.run(None, {'observation': obs})[0]
action = np.argmax(probs[0])  # 0 or 1
```

## ⚙️ Configuration

Edit `config.yaml` to change:

**Training duration:**
```yaml
training:
  total_timesteps: 100000  # Increase for better performance
```

**Environment:**
```yaml
environment:
  ev_capacity_kwh: 75.0     # EV battery size
  target_soc: 0.9           # Target charge level
```

**PPO settings:**
```yaml
ppo:
  learning_rate: 3e-4       # Lower = more stable, slower
  n_steps: 2048             # More = better samples, slower
```

## 🐛 Troubleshooting

**Problem**: Import errors
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem**: Training too slow
```bash
# Solution: Reduce timesteps in config.yaml
# Or use GPU (SB3 auto-detects)
```

**Problem**: ONNX export fails
```bash
# Solution: Ensure training completed successfully
# Check PyTorch version: torch >= 1.9.0
```

## 📊 Understanding Output

### Training Output
```
Episode 1: reward = 45.2, length = 18 hours
Episode 2: reward = 52.1, length = 20 hours
...
Average reward increasing → Agent learning ✅
```

### Inference Output
```
Input: [5.0 kW solar, 0.5 SoC, 12.0 hours, $0.20/kWh]
Action: 1 (Solar + Grid)
Confidence: 0.85
```

### Evaluation Plot
Shows 3 graphs:
1. **Solar vs Grid ratio** - How much free vs paid power
2. **SoC over time** - Battery charge progress
3. **Solar & Price** - Available solar and grid cost

## 🎓 Next Steps

1. ✅ Train agent → `python -m src.sb3_training`
2. ✅ Test model → `python -m src.inference_onnx solar_agent.onnx`
3. ✅ Deploy ONNX → Transfer to Snapdragon device
4. ✅ Integrate → Connect to real EV charging system

For detailed documentation, see [README.md](README.md).
