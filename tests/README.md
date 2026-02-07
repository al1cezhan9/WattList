# Test Files Overview

## ✅ New Test Files (For EV Charging)

### `test_environment.py`
Tests the EV Charging Environment:
- Environment initialization
- Observation and action spaces
- Reset functionality
- Step function (solar only and grid only)
- Charging calculations
- Reward function
- Terminal penalties
- PV output generation
- Grid pricing schedule
- Departure sampling

### `test_baselines.py`
Tests baseline policies:
- SolarFirstGreedy policy
- ConservativeDeadline policy
- Action selection logic

### `test_training.py`
Tests training functionality:
- Model creation
- Training steps
- Model saving/loading
- Model prediction

## ❌ Old Test Files (Outdated - CartPole)

These tests are for the old CartPole project and should be ignored or deleted:

- `test_agent.py` - Old CartPole agent
- `test_config.py` - Old config loading
- `test_network.py` - Old MLP network
- `test_main.py` - Old main entry point
- `test_web_server.py` - Old Flask server
- `test_formats.py` - Old model formats
- `test_model.py` - Old model loading
- `test_utils.py` - Old utilities
- `test_integration.py` - Old integration tests
- `test_onnx_model.py` - May work with updates

## 🚀 How to Run Tests

### Install pytest (if not already installed)

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py

# Run specific test class
pytest tests/test_environment.py::TestEVChargingEnv

# Run specific test function
pytest tests/test_environment.py::TestEVChargingEnv::test_reset
```

### Run with coverage

```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage report
pytest tests/ --cov=envs --cov=baselines --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
```

### Run only new tests (skip old ones)

```bash
# Run only new test files
pytest tests/test_environment.py tests/test_baselines.py tests/test_training.py
```

## 📊 What Tests Verify

### Environment Tests (`test_environment.py`)

1. **Initialization**: Environment creates correctly with config
2. **Spaces**: Observation (12D) and action (2) spaces correct
3. **Reset**: Environment resets to valid initial state
4. **Actions**: 
   - Solar only (action 0) uses no grid, costs nothing
   - Grid only (action 1) uses grid, costs money
5. **Charging**: SoC updates correctly based on charging power
6. **Rewards**: Step rewards and terminal penalties calculated correctly
7. **PV Model**: PV output follows diurnal pattern
8. **Pricing**: Grid prices follow ToU schedule
9. **Departure**: Departure sampling works correctly

### Baseline Tests (`test_baselines.py`)

1. **SolarFirstGreedy**: Chooses solar when available, grid otherwise
2. **ConservativeDeadline**: Charges aggressively when below target

### Training Tests (`test_training.py`)

1. **Model Creation**: RecurrentPPO model can be created
2. **Training**: Model can train without errors
3. **Save/Load**: Model can be saved and loaded
4. **Prediction**: Model can predict actions

## 🐛 Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root:

```bash
cd /path/to/WattList
pytest tests/
```

### Missing Dependencies

Install all requirements:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Old Tests Failing

Old CartPole tests will fail - this is expected. Run only new tests:

```bash
pytest tests/test_environment.py tests/test_baselines.py tests/test_training.py
```

## 📝 Writing New Tests

### Example Test Structure

```python
import pytest
from envs import EVChargingEnv

class TestMyFeature:
    """Test suite for my feature."""
    
    def test_feature_works(self):
        """Test that feature works correctly."""
        env = EVChargingEnv()
        obs, _ = env.reset()
        
        # Test something
        assert obs.shape == (12,)
```

### Best Practices

1. **One test per behavior**: Each test should verify one specific behavior
2. **Descriptive names**: Test names should describe what they test
3. **Use fixtures**: Share common setup code using fixtures
4. **Assert clearly**: Use clear assertions with helpful messages
5. **Test edge cases**: Test boundary conditions and error cases

## 🎯 Continuous Integration

To run tests automatically:

```bash
# Run tests before committing
pytest tests/test_environment.py tests/test_baselines.py tests/test_training.py

# Or add to CI/CD pipeline
pytest tests/ --cov=envs --cov=baselines --cov-report=xml
```
