# Files No Longer Needed - Solar-Arbitrage Migration

## Core Files to Remove (CartPole-Specific)

### 1. **`src/agent.py`**
- **Reason**: Custom PPO agent implementation replaced by Stable Baselines 3
- **Replacement**: SB3's PPO agent (handled in `src/sb3_training.py`)

### 2. **`src/network.py`**
- **Reason**: Custom PPO network architecture replaced by SB3's MlpPolicy
- **Replacement**: SB3 handles network architecture internally

### 3. **`src/training.py`**
- **Reason**: Custom training loop replaced by SB3's training API
- **Replacement**: `src/sb3_training.py` uses SB3's `model.learn()`

### 4. **`main.py`**
- **Reason**: Entry point for old CartPole training system
- **Replacement**: Use `python -m src.sb3_training` directly

### 5. **`src/web_server.py`**
- **Reason**: CartPole-specific Flask web server for visualization
- **Note**: Could be kept if you want to adapt it for Solar-Arbitrage visualization

### 6. **`src/visualization/`** (entire directory)
- **Reason**: CartPole-specific HTML/JS/CSS visualization
- **Files**:
  - `src/visualization/index.html`
  - `src/visualization/styles.css`
  - `src/visualization/visualization.js`

## Files That May Need Updates (Not Remove, But Update)

### 7. **`src/model_loader.py`**
- **Status**: Uses `PPONetwork` (CartPole-specific)
- **Action**: Update to work with SB3 models or remove if not needed
- **Note**: ONNX loading part might still be useful

### 8. **`src/aihub_conversion.py`**
- **Status**: Uses `PPONetwork` for conversion
- **Action**: Update to work with SB3 models or remove
- **Note**: Conversion logic could be adapted for SB3 models

## Test Files to Remove/Update

### 9. **`tests/test_agent.py`**
- **Reason**: Tests for custom PPOAgent class

### 10. **`tests/test_network.py`**
- **Reason**: Tests for PPONetwork class

### 11. **`tests/test_training.py`**
- **Reason**: Tests for custom training loop

### 12. **`tests/test_main.py`**
- **Reason**: Tests for old main.py entry point

### 13. **`tests/test_environment.py`**
- **Status**: Needs update (tests CartPoleEnv, should test SolarArbitrageEnv)

### 14. **`tests/test_model.py`**
- **Status**: May need update if it tests PPONetwork

## Example Files (Optional to Remove)

### 15. **`example/model.pth`**
- **Reason**: CartPole trained model (not compatible with Solar-Arbitrage)

### 16. **`example/model.pt`**
- **Reason**: CartPole TorchScript model

### 17. **`example/model.onnx`**
- **Reason**: CartPole ONNX model

### 18. **`example/compare_models.py`**
- **Status**: Uses PPONetwork, needs update or remove

### 19. **`example/debug_qnn.py`**
- **Status**: QNN debugging script - might still be useful, check if it's CartPole-specific

### 20. **`example/training-log.log`**
- **Reason**: Old training log file

## Assets (Optional to Remove)

### 21. **`assets/cart-pole.mov`**
- **Reason**: CartPole demonstration video

### 22. **`assets/cartpole.gif`**
- **Reason**: CartPole demonstration GIF

## Summary

### Must Remove (7 files/directories):
1. `src/agent.py`
2. `src/network.py`
3. `src/training.py`
4. `main.py`
5. `src/web_server.py`
6. `src/visualization/` (directory)
7. `assets/cart-pole.mov` and `assets/cartpole.gif`

### Should Update or Remove (2 files):
8. `src/model_loader.py` (update for SB3 or remove)
9. `src/aihub_conversion.py` (update for SB3 or remove)

### Test Files to Update/Remove (5+ files):
10-14. Various test files in `tests/` directory

### Example Files (Optional):
15-20. Files in `example/` directory (CartPole models and scripts)

## Files to KEEP:
- ✅ `src/environment.py` (transformed to SolarArbitrageEnv)
- ✅ `src/sb3_training.py` (new SB3 training script)
- ✅ `src/inference_onnx.py` (new ONNX inference script)
- ✅ `src/config.py` (still used)
- ✅ `src/utils.py` (still used)
- ✅ `config.yaml` (updated for Solar-Arbitrage)
- ✅ `requirements.txt` (updated)
- ✅ `src/__init__.py` (package marker)
