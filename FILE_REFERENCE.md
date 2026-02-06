# Complete File Reference

## 📁 Directory Structure

```
WattList/
├── envs/                    # Environment implementation
├── baselines/               # Baseline policies
├── models/                  # Training & export scripts
├── eval/                    # Evaluation scripts
├── android/                 # Mobile inference
├── configs/                 # Configuration files
├── src/                     # Legacy utilities (minimal)
├── tests/                   # Test files (legacy)
├── example/                 # Example files (legacy)
└── [documentation files]
```

---

## 🔧 Core Files

### Environment

#### `envs/ev_charging_env.py` ⭐ **CORE**
- **Purpose**: Main RL environment
- **Class**: `EVChargingEnv` (Gymnasium-compatible)
- **Key Features**:
  - 12D observation space with PV forecasts
  - Hidden departure model (non-stationary)
  - AR(1) correlated PV noise
  - 15-minute timesteps
- **Usage**: Imported by training and evaluation scripts

#### `envs/__init__.py`
- **Purpose**: Package initialization
- **Exports**: `EVChargingEnv`

---

### Baselines

#### `baselines/heuristics.py` ⭐ **COMPARISON**
- **Purpose**: Rule-based policies for comparison
- **Classes**:
  - `SolarFirstGreedy`: Always use solar when available
  - `ConservativeDeadline`: Always charge at max
  - `EmpiricalSurvival`: Tracks historical departures
- **Usage**: Used by `eval/evaluate.py` for comparison

#### `baselines/__init__.py`
- **Purpose**: Package initialization
- **Exports**: All baseline classes

---

### Training

#### `models/train_recurrent_ppo.py` ⭐ **MAIN TRAINING**
- **Purpose**: Train RecurrentPPO agent
- **Key Functions**:
  - `load_config()`: Load YAML config
  - `make_env()`: Create environment factory
  - `train_recurrent_ppo()`: Main training function
- **Usage**: 
  ```bash
  python models/train_recurrent_ppo.py --config configs/default.yaml
  ```
- **Output**: `models/recurrent_ppo_ev_charging.zip`

#### `models/export_onnx.py` ⭐ **ONNX EXPORT**
- **Purpose**: Export trained model to ONNX
- **Key Classes**:
  - `OnnxableLSTMPolicy`: ONNX-compatible wrapper
- **Key Functions**:
  - `export_to_onnx()`: Main export function
- **Usage**:
  ```bash
  python models/export_onnx.py --model models/recurrent_ppo_ev_charging
  ```
- **Output**: `models/ev_charging_policy.onnx`

---

### Evaluation

#### `eval/evaluate.py` ⭐ **EVALUATION**
- **Purpose**: Evaluate policies and compare with baselines
- **Key Functions**:
  - `evaluate_policy()`: Run episodes and collect metrics
  - `main()`: Compare all policies
- **Usage**:
  ```bash
  python eval/evaluate.py --model models/recurrent_ppo_ev_charging --episodes 100
  ```
- **Output**: Console comparison table

---

### Android

#### `android/inference_example.py` ⭐ **MOBILE INFERENCE**
- **Purpose**: Demonstrate ONNX inference for Android
- **Key Class**: `EVChargingInference`
  - `__init__()`: Load ONNX model
  - `reset()`: Reset LSTM state
  - `predict()`: Run inference
- **Usage**: 
  ```bash
  python android/inference_example.py
  ```
- **Integration**: Use as template for Android app

---

### Configuration

#### `configs/default.yaml` ⭐ **CONFIGURATION**
- **Purpose**: Central configuration file
- **Sections**:
  - `environment`: EV, PV, departure, pricing parameters
  - `recurrent_ppo`: Training hyperparameters
  - `training`: Training settings
  - `evaluation`: Evaluation settings
  - `onnx_export`: ONNX export settings
- **Usage**: Referenced by all scripts

---

## 📄 Documentation Files

### `README.md` ⭐ **START HERE**
- **Purpose**: Main project documentation
- **Contents**: Quick start, usage, architecture overview

### `CODEBASE_EXPLANATION.md` ⭐ **DETAILED GUIDE**
- **Purpose**: Comprehensive explanation of how everything works
- **Contents**: 
  - System overview
  - File-by-file explanation
  - Workflow diagrams
  - Technical details

### `TRANSFORMATION_SUMMARY.md`
- **Purpose**: Documents transformation from old codebase
- **Contents**: Changes made, migration guide

### `CHANGES.md`
- **Purpose**: Technical change log
- **Contents**: Detailed changes, API differences

### `FINAL_SUMMARY.md`
- **Purpose**: Quick reference for transformation
- **Contents**: Summary of changes, status

### `FILE_REFERENCE.md` (this file)
- **Purpose**: Quick file reference
- **Contents**: All files and their purposes

---

## 🗑️ Legacy Files (Can Be Removed)

### `src/` Directory
- `src/config.py`: Old config loader (not used by new code)
- `src/utils.py`: Old utilities (logging formatter, may be useful)

### `tests/` Directory
- All test files: Old tests for removed code
- **Status**: Not compatible with new system

### `example/` Directory
- `example/debug_qnn.py`: QNN debugging (may be useful)
- `example/compare_models.py`: Old model comparison
- `example/model.*`: Old CartPole models (not compatible)

### `config.yaml`
- **Status**: Old config file (replaced by `configs/default.yaml`)

### `docker-compose.yaml`
- **Status**: May be outdated

### `scripts/`
- Test runner scripts: May be outdated

---

## 🎯 Essential Files for Operation

### Minimum Required Files:

1. **Environment**: `envs/ev_charging_env.py`
2. **Training**: `models/train_recurrent_ppo.py`
3. **Export**: `models/export_onnx.py`
4. **Config**: `configs/default.yaml`
5. **Dependencies**: `requirements.txt`

### For Evaluation:

6. **Baselines**: `baselines/heuristics.py`
7. **Evaluation**: `eval/evaluate.py`

### For Android:

8. **Inference**: `android/inference_example.py`

---

## 📊 File Dependencies

```
configs/default.yaml
    │
    ├─→ models/train_recurrent_ppo.py
    │       └─→ envs/ev_charging_env.py
    │
    ├─→ models/export_onnx.py
    │       └─→ [trained model]
    │
    └─→ eval/evaluate.py
            ├─→ envs/ev_charging_env.py
            └─→ baselines/heuristics.py

android/inference_example.py
    └─→ [ONNX model from export_onnx.py]
```

---

## 🔍 File Purposes Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `envs/ev_charging_env.py` | RL environment | Always (core) |
| `models/train_recurrent_ppo.py` | Train agent | Training phase |
| `models/export_onnx.py` | Export to ONNX | After training |
| `eval/evaluate.py` | Evaluate policies | After training |
| `android/inference_example.py` | Mobile inference | Android deployment |
| `baselines/heuristics.py` | Baseline policies | Comparison |
| `configs/default.yaml` | Configuration | Always |

---

## 🚀 Quick Start File Order

1. **Read**: `README.md` (overview)
2. **Read**: `CODEBASE_EXPLANATION.md` (details)
3. **Edit**: `configs/default.yaml` (customize)
4. **Run**: `models/train_recurrent_ppo.py` (train)
5. **Run**: `eval/evaluate.py` (evaluate)
6. **Run**: `models/export_onnx.py` (export)
7. **Use**: `android/inference_example.py` (deploy)

---

## 📝 Notes

- **Core files** are marked with ⭐
- **Legacy files** can be removed but kept for reference
- **Documentation** files explain different aspects
- All **core files** are self-contained and well-commented
