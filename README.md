# Solar-Arbitrage EV Controller

A reinforcement learning system for optimizing EV charging using solar power arbitrage. Trained on Snapdragon X PC and deployed to Snapdragon 8 Elite NPU.

## 🎯 Project Overview

This project implements a **single-agent reinforcement learning controller** that optimizes EV charging by:
- Maximizing use of free solar energy
- Minimizing grid electricity costs (Time-of-Use pricing)
- Ensuring the EV reaches target charge (90% SoC) before departure

The agent learns when to charge using only solar power vs. when to supplement with grid power, balancing cost, sustainability, and user satisfaction.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │ Environment  │──────▶│  SB3 PPO     │──────▶│  ONNX    │ │
│  │ (Gymnasium)  │◀──────│  Agent       │      │  Export   │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│         │                      │                             │
│         └──────────────────────┘                             │
│                    Training Loop                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Deployment Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   ONNX       │──────▶│  ONNX       │──────▶│  NPU     │ │
│  │   Model      │      │  Runtime    │      │  (S8E)    │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
WattList/
├── src/
│   ├── environment.py      # SolarArbitrageEnv (Gymnasium environment)
│   ├── sb3_training.py     # Training script with ONNX export
│   ├── inference_onnx.py   # ONNX inference verification
│   ├── config.py           # Configuration loader
│   └── utils.py            # Utility functions
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── models/                 # Trained models (created during training)
    └── solar_agent.zip     # SB3 model checkpoint
```

## 🔧 Components Explained

### 1. Environment (`src/environment.py`)

**SolarArbitrageEnv** - A Gymnasium-compatible RL environment:

**State Space (4D):**
- `solar_output` (0-10 kW): Current solar generation (diurnal sine wave + noise)
- `ev_soc` (0.0-1.0): EV battery state of charge
- `time_remaining` (0-24 hours): Hours until departure
- `grid_price` ($/kWh): Current Time-of-Use electricity price

**Action Space (2 discrete actions):**
- **Action 0 (Solar Only)**: Charge at `min(solar_output, 7.4 kW)`, cost = $0
- **Action 1 (Solar + Grid)**: Charge at max 7.4 kW, grid provides difference

**Reward Function:**
```
R = (1.5 × P_solar) - (P_grid × price) - Penalty_deadline
```

Where:
- `P_solar`: Solar power used (kW)
- `P_grid`: Grid power used (kW)
- `Penalty_deadline`: `-100 × (0.9 - final_soc)` if SoC < 90% at departure

**Constraints:**
- EV battery: 75 kWh capacity
- Charging efficiency: 90% (η = 0.90)
- Max charging power: 7.4 kW
- Each step = 1 hour

### 2. Training (`src/sb3_training.py`)

Uses **Stable Baselines 3 (SB3)** PPO algorithm:

1. **Creates Environment**: Wraps `SolarArbitrageEnv` in vectorized environment
2. **Initializes PPO Agent**: Uses `MlpPolicy` (multi-layer perceptron)
3. **Trains**: Runs for specified timesteps, saves checkpoints
4. **Exports to ONNX**: Converts trained policy to ONNX format for NPU deployment

**Key Features:**
- Checkpoint callbacks (saves every 10k steps)
- Evaluation callbacks (tests performance periodically)
- ONNX export with fixed input shape `(1, 4)` for batch-1 NPU inference

### 3. ONNX Export (`src/sb3_training.py`)

**OnnxablePolicy** wrapper:
- Extracts actor network from SB3 policy
- Removes broadcast layers for NPU compatibility
- Ensures fixed input shape: `(batch_size=1, state_dim=4)`
- Output: Action probabilities `(batch_size=1, num_actions=2)`

### 4. Inference (`src/inference_onnx.py`)

**ONNXInferenceAgent**:
- Loads ONNX model
- Supports QNN provider for Snapdragon NPU acceleration
- Falls back to CPU if QNN unavailable
- Provides evaluation and visualization functions

## 🚀 How to Run

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `stable-baselines3` - PPO implementation
- `gymnasium` - RL environment interface
- `torch` - PyTorch for training
- `onnxruntime` - ONNX inference
- `matplotlib` - Visualization

### Step 1: Configure (Optional)

Edit `config.yaml` to customize:
- Environment parameters (EV capacity, charging efficiency, etc.)
- PPO hyperparameters (learning rate, batch size, etc.)
- Training duration (`total_timesteps`)
- Model save paths

### Step 2: Train the Agent

```bash
# Train the agent (will take time depending on total_timesteps)
python -m src.sb3_training
```

**What happens:**
1. Loads configuration from `config.yaml`
2. Creates `SolarArbitrageEnv` environment
3. Initializes PPO agent with `MlpPolicy`
4. Trains for specified timesteps (default: 100,000)
5. Saves checkpoints to `models/solar_agent_*`
6. Exports final model to `solar_agent.onnx`

**Output:**
- `models/solar_agent.zip` - SB3 model checkpoint
- `models/solar_agent_*` - Periodic checkpoints
- `solar_agent.onnx` - ONNX model for deployment
- `tensorboard_logs/` - Training metrics (view with TensorBoard)

**Monitor Training:**
```bash
# View training progress with TensorBoard
tensorboard --logdir=./tensorboard_logs
```

### Step 3: Verify ONNX Model

```bash
# Single prediction test (mimics Galaxy S25 execution)
python -m src.inference_onnx solar_agent.onnx --single-test

# Full evaluation with plots
python -m src.inference_onnx solar_agent.onnx --episodes 3
```

**What happens:**
1. Loads ONNX model
2. Creates environment
3. Runs episodes using ONNX agent
4. Generates plots:
   - Solar vs Grid Power Ratio over time
   - EV SoC over time (with target line)
   - Solar output and grid price over time
5. Saves plot to `solar_arbitrage_evaluation.png`

### Step 4: Deploy to NPU (Snapdragon 8 Elite)

The ONNX model (`solar_agent.onnx`) is ready for deployment:

1. **Transfer to device**: Copy `solar_agent.onnx` to Galaxy S25
2. **Use onnxruntime-qnn**: Load model with QNN provider
3. **Run inference**: Pass observations `(1, 4)` → get action probabilities

**Example NPU inference code:**
```python
import onnxruntime as ort
import numpy as np

# Load model with QNN provider
session = ort.InferenceSession(
    'solar_agent.onnx',
    providers=['QNNExecutionProvider', 'CPUExecutionProvider']
)

# Prepare observation: [solar_output, ev_soc, time_remaining, grid_price]
observation = np.array([[5.0, 0.5, 12.0, 0.20]], dtype=np.float32)

# Run inference
outputs = session.run(None, {'observation': observation})
action_probs = outputs[0][0]
action = np.argmax(action_probs)  # 0 or 1
```

## 📊 Understanding the Environment

### Solar Generation Model

Solar output follows a **diurnal sine wave**:
- **Peak**: Noon (hour 12) → 10 kW max
- **Night**: Hours 0-6 and 18-24 → 0 kW
- **Noise**: Gaussian noise (std=0.5) added for realism

### Time-of-Use Pricing

**3-tier pricing system:**
- **Peak** (4 PM - 9 PM): $0.30/kWh
- **Mid** (9 AM - 4 PM, 9 PM - 11 PM): $0.20/kWh
- **Off-peak** (11 PM - 9 AM): $0.10/kWh

### Episode Structure

- **Initialization**: Random SoC (0.2-0.5), random time remaining (4-24 hours)
- **Each step**: 1 hour passes
- **Termination**: When `time_remaining <= 0`
- **Success**: SoC ≥ 90% at departure
- **Failure**: SoC < 90% at departure → penalty applied

## 🎓 Training Process

### What the Agent Learns

The PPO agent learns to:
1. **Prioritize solar**: Use free solar energy when available
2. **Avoid peak pricing**: Minimize grid usage during expensive hours
3. **Plan ahead**: Balance immediate costs with future needs
4. **Meet deadline**: Ensure sufficient charge before departure

### Training Metrics

Monitor with TensorBoard:
- **Episode reward**: Total reward per episode
- **Episode length**: Hours per episode
- **Value loss**: Critic network loss
- **Policy loss**: Actor network loss

### Typical Training Time

- **100k timesteps**: ~30-60 minutes (depends on hardware)
- **500k timesteps**: ~2-4 hours (recommended for better performance)
- **1M timesteps**: ~4-8 hours (for production-quality agent)

## 🔍 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the project root
cd /path/to/WattList

# Install dependencies
pip install -r requirements.txt
```

**2. ONNX Export Fails**
- Check PyTorch version: `torch >= 1.9.0`
- Ensure model is trained first
- Check ONNX opset version compatibility

**3. QNN Provider Not Available**
- Normal on non-Snapdragon devices
- Falls back to CPU automatically
- For NPU testing, use Snapdragon device or emulator

**4. Training Too Slow**
- Reduce `total_timesteps` in config.yaml
- Use GPU if available (SB3 will auto-detect)
- Reduce `n_steps` or `batch_size` for faster updates

## 📈 Expected Results

After training, you should see:
- **Average reward**: Increasing over time
- **Final SoC**: Consistently ≥ 90% at departure
- **Grid usage**: Minimized during peak hours
- **Solar utilization**: Maximized when available

## 🔗 Key Files Reference

| File | Purpose |
|------|---------|
| `src/environment.py` | RL environment definition |
| `src/sb3_training.py` | Training script + ONNX export |
| `src/inference_onnx.py` | ONNX inference verification |
| `config.yaml` | Configuration parameters |
| `requirements.txt` | Python dependencies |

## 📝 Configuration Guide

### Environment Parameters (`config.yaml`)

```yaml
environment:
  ev_capacity_kwh: 75.0          # EV battery capacity
  charging_efficiency: 0.90      # Charging efficiency (90%)
  max_charging_power_kw: 7.4     # Max charging power
  max_solar_output_kw: 10.0      # Max solar generation
  target_soc: 0.9                # Target SoC at departure (90%)
```

### PPO Hyperparameters

```yaml
ppo:
  learning_rate: 3e-4            # Learning rate
  n_steps: 2048                   # Steps per update
  batch_size: 64                  # Batch size
  gamma: 0.99                     # Discount factor
  clip_range: 0.2                 # PPO clipping range
```

## 🎯 Next Steps

1. **Train**: Run training with default or custom config
2. **Evaluate**: Test ONNX model with inference script
3. **Deploy**: Transfer ONNX model to Snapdragon device
4. **Integrate**: Connect to real EV charging system
5. **Monitor**: Track performance in production

## 📚 Additional Resources

- [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Snapdragon AI Hub](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct)

## 📄 License

See LICENSE file for details.
