# Solar-Arbitrage EV Controller - Implementation Summary

## Overview
This document summarizes the transformation of the CartPole RL project into a Solar-Arbitrage EV Controller for training on Snapdragon X PC and deployment to Snapdragon 8 Elite NPU.

## Key Changes Made

### 1. Environment Transformation (`src/environment.py`)

**Before:** `CartPoleEnv` - Custom cart-pole physics simulation
**After:** `SolarArbitrageEnv` - Gymnasium-compatible environment for EV charging optimization

#### State Space (Box, 4D):
- `solar_output` (0-10 kW): Diurnal sine wave + Gaussian noise
- `ev_soc` (0.0-1.0): Current EV battery level
- `time_remaining` (0-24 hours): Hours until departure
- `grid_price` ($/kWh): 3-tier Time-of-Use tariff (Peak, Mid, Off-peak)

#### Action Space (Discrete, 2):
- **Action 0 (Solar Only)**: `P = min(solar_output, 7.4 kW)`, Cost = $0
- **Action 1 (Solar + Grid)**: `P = 7.4 kW`, Grid provides `P_grid = max(0, 7.4 - solar_output)`

#### Constraints:
- EV battery capacity: 75 kWh
- Charging efficiency: η = 0.90
- NO house battery logic
- Each step = 1 hour

#### Reward Function:
```
R = (1.5 * P_solar) - (P_grid * price) - Penalty_deadline
```

Where:
- `P_solar`: Solar power used (kW)
- `P_grid`: Grid power used (kW)
- `price`: Current grid price ($/kWh)
- `Penalty_deadline`: `-100 * (0.9 - final_soc)` if `time_remaining == 0` and `ev_soc < 0.9`

### 2. Training Pipeline (`src/sb3_training.py`)

**New File:** Complete SB3 training script with ONNX export capability

#### Features:
- Uses Stable Baselines 3 PPO with `MlpPolicy`
- Implements `OnnxablePolicy` wrapper for ONNX compatibility
- Exports to ONNX with fixed input shape `(1, 4)` for NPU batch-1 inference
- Includes checkpointing and evaluation callbacks

#### ONNX Export:
- Removes broadcast layers for NPU compatibility
- Fixed input shape: `(batch_size=1, state_dim=4)`
- Output: Action probabilities `(batch_size=1, num_actions=2)`
- Uses ONNX opset version 11 (compatible with onnxruntime-qnn)

### 3. Inference Verification (`src/inference_onnx.py`)

**New File:** ONNX inference script for verification and evaluation

#### Features:
- `ONNXInferenceAgent`: Loads and runs ONNX models
- Supports QNN provider for Snapdragon NPU acceleration
- Falls back to CPU if QNN unavailable
- Single prediction test (mimics Galaxy S25 execution)
- Full episode evaluation with metrics

#### Plotting Function:
- **Plot 1**: Solar vs Grid Power Ratio over time (stacked area chart)
- **Plot 2**: EV SoC over time with target line (0.9)
- **Plot 3**: Solar output and grid price over time (dual y-axis)

### 4. Configuration Updates (`config.yaml`)

**Updated:** Complete configuration for Solar-Arbitrage environment

#### New Parameters:
- EV constraints: `ev_capacity_kwh`, `charging_efficiency`, `max_charging_power_kw`
- Solar generation: `max_solar_output_kw`, `solar_noise_std`
- ToU pricing: `tou_prices` (peak, mid, off_peak)
- Episode: `max_hours`, `target_soc`, `departure_penalty_scale`
- Reward weights: `solar_reward_weight`, `grid_cost_weight`

#### SB3 PPO Parameters:
- Updated to SB3-specific parameters (`n_steps`, `batch_size`, `n_epochs`, etc.)
- Training: `total_timesteps`, `model_save_path`, `onnx_export_path`

### 5. Dependencies (`requirements.txt`)

**Added:**
- `gymnasium>=0.29.0` - Gym-compatible environment interface
- `stable-baselines3>=2.0.0` - PPO implementation
- `matplotlib>=3.5.0` - Visualization for evaluation plots

## File Structure

```
WattList/
├── src/
│   ├── environment.py          # ✅ TRANSFORMED: SolarArbitrageEnv
│   ├── sb3_training.py          # ✅ NEW: SB3 training + ONNX export
│   ├── inference_onnx.py        # ✅ NEW: ONNX inference verification
│   ├── config.py                # ✅ PRESERVED: Config loader
│   └── utils.py                 # ✅ PRESERVED: Utility functions
├── config.yaml                  # ✅ UPDATED: Solar-Arbitrage config
├── requirements.txt             # ✅ UPDATED: Added gymnasium, SB3, matplotlib
└── SOLAR_ARBITRAGE_CHANGES.md   # ✅ NEW: This file
```

## Usage

### Training:
```bash
python -m src.sb3_training
```

This will:
1. Train a PPO agent using SB3
2. Save the model to `models/solar_agent`
3. Export ONNX model to `solar_agent.onnx`

### Inference Verification:
```bash
# Single prediction test
python -m src.inference_onnx solar_agent.onnx --single-test

# Full evaluation (with plots)
python -m src.inference_onnx solar_agent.onnx --episodes 3
```

## Key Implementation Details

### Environment Implementation:
- **Solar Generation**: Diurnal sine wave peaking at noon (hour 12), zero at night
- **ToU Pricing**: 3-tier system with peak (4-9 PM), mid (9 AM-4 PM, 9-11 PM), off-peak (11 PM-9 AM)
- **Charging Logic**: Accounts for efficiency (90%) when updating SoC
- **Terminal Condition**: Episode ends when `time_remaining <= 0`

### ONNX Export:
- Uses `OnnxablePolicy` wrapper to extract actor network from SB3 policy
- Traces through: `features_extractor` → `mlp_extractor.forward_actor` → `action_net` → `softmax`
- Fixed batch size of 1 for NPU compatibility
- Verified with onnxruntime for correctness

### Reward Function:
The reward balances three objectives:
1. **Sustainability**: Positive reward for using free solar energy (1.5x multiplier)
2. **Cost**: Negative penalty for grid power usage (proportional to price)
3. **Satisfaction**: Terminal penalty if SoC < 90% at departure (-100 per 0.1 deficit)

## Compatibility

- **Training**: Snapdragon X PC (x86_64, Windows/Linux)
- **Deployment**: Snapdragon 8 Elite NPU (via onnxruntime-qnn)
- **Python**: 3.10+
- **Framework**: PyTorch, Stable Baselines 3, Gymnasium

## Notes

- The original CartPole code structure is preserved where possible
- Utility functions (`utils.py`, `config.py`) remain unchanged
- The environment follows Gymnasium API for compatibility with SB3
- ONNX model uses fixed input shape for NPU batch-1 inference
- All code follows PEP8 style guidelines
