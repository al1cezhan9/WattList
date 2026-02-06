# Complete Codebase Explanation

## 🎯 System Overview

This is a **Recurrent Reinforcement Learning** system for optimizing EV charging under:
- **Solar uncertainty** (variable PV output with forecasts)
- **Unknown departure behavior** (hidden, non-stationary process)
- **Time-of-use pricing** (variable grid costs)

The agent learns when to use **solar-only** vs **solar+grid** charging to minimize costs while ensuring the EV reaches target charge before departure.

---

## 📁 File Structure & Purpose

### Core Environment (`envs/`)

#### `envs/ev_charging_env.py` - The RL Environment

**Purpose**: Gymnasium-compatible environment that simulates EV charging scenarios.

**Key Components**:

1. **Observation Space (12D)**:
   ```
   [soc, time_elapsed_norm, pv_current, 
    pv_forecast_mean (4), pv_forecast_std (4), 
    grid_price]
   ```
   - `soc`: Current battery level (0-1)
   - `time_elapsed_norm`: Normalized time since plug-in (0-1)
   - `pv_current`: Current solar output (kW)
   - `pv_forecast_mean`: Forecasted solar for next 4 steps (1 hour)
   - `pv_forecast_std`: Forecast uncertainty for next 4 steps
   - `grid_price`: Current electricity price ($/kWh)

2. **Action Space**: Discrete(2)
   - `0`: Solar only (free, limited by PV output)
   - `1`: Solar + Grid (max power, costs money)

3. **PV Generation Model**:
   - **Diurnal curve**: Bell-shaped (sinusoidal) peaking at noon
   - **AR(1) noise**: Correlated forecast errors (realistic)
   - **Forecast bias**: Systematically overestimates (10% bias)

4. **Hidden Departure Model**:
   - **3 user types**: Different departure habits (early/medium/late)
   - **Hazard-based**: Probability increases with time elapsed
   - **Non-stationary**: Distribution drifts during training
   - **NOT observable**: Agent must learn from interaction

5. **Reward Function**:
   ```python
   # Per step: negative cost
   reward = -grid_energy_kwh * grid_price
   
   # Terminal: penalty if undercharged
   if terminated and soc < 0.9:
       reward -= 100 * (0.9 - soc)
   ```

**How it works**:
- Each timestep = 15 minutes
- Episode starts when EV plugs in (random SoC, random time)
- Episode ends when EV departs (hidden hazard process)
- Agent observes state, chooses action, receives reward
- Must learn to balance cost vs. deadline satisfaction

---

### Baselines (`baselines/`)

#### `baselines/heuristics.py` - Comparison Policies

**Purpose**: Simple rule-based policies to compare against learned RL policy.

1. **SolarFirstGreedy**:
   - Always use solar when available (>0.1 kW)
   - Use grid otherwise
   - No planning, just immediate optimization

2. **ConservativeDeadline**:
   - Always charge at max power
   - Assumes worst-case departure
   - Ensures target SoC but wastes money

3. **EmpiricalSurvival**:
   - Tracks historical departure times
   - Estimates survival probability
   - Charges aggressively when survival prob is low

**Why baselines matter**: They establish performance floors. RL should outperform them.

---

### Training (`models/`)

#### `models/train_recurrent_ppo.py` - Main Training Script

**Purpose**: Trains a RecurrentPPO agent with LSTM policy.

**How it works**:

1. **Loads Configuration**:
   ```python
   config = load_config("configs/default.yaml")
   ```

2. **Creates Environment**:
   ```python
   env = DummyVecEnv([make_env(config)])
   ```

3. **Creates RecurrentPPO Agent**:
   ```python
   model = RecurrentPPO(
       'MlpLstmPolicy',  # LSTM policy
       env,
       lstm_hidden_size=64,
       learning_rate=3e-4,
       # ... other hyperparameters
   )
   ```

4. **Training Loop**:
   - Collects trajectories (2048 steps)
   - Updates policy using PPO algorithm
   - Handles LSTM state automatically
   - Saves checkpoints periodically

5. **Callbacks**:
   - **CheckpointCallback**: Saves model every 50k steps
   - **EvalCallback**: Evaluates performance periodically

**Key Features**:
- **CPU-only**: Optimized for ARM64 Snapdragon
- **LSTM memory**: Handles temporal dependencies
- **Reproducible**: Fixed random seeds

**Output**: Trained model saved to `models/recurrent_ppo_ev_charging.zip`

---

#### `models/export_onnx.py` - ONNX Export for Mobile

**Purpose**: Converts trained PyTorch model to ONNX format for Android deployment.

**How it works**:

1. **Loads Trained Model**:
   ```python
   model = RecurrentPPO.load(model_path)
   ```

2. **Creates ONNX-Compatible Wrapper**:
   ```python
   class OnnxableLSTMPolicy:
       # Extracts: features_extractor -> mlp_extractor -> lstm -> action_net
       # Wraps for ONNX export
   ```

3. **Exports to ONNX**:
   ```python
   torch.onnx.export(
       onnxable_policy,
       (dummy_obs, dummy_lstm_state),
       output_path,
       input_names=['observation', 'lstm_state'],
       output_names=['action_probs']
   )
   ```

**Key Challenges**:
- **LSTM state**: Must be passed as input (not internal)
- **Fixed shapes**: Required for mobile deployment
- **ONNX compatibility**: Uses opset 11 for mobile support

**Output**: `models/ev_charging_policy.onnx` (ready for Android)

---

### Evaluation (`eval/`)

#### `eval/evaluate.py` - Policy Evaluation

**Purpose**: Evaluates and compares RL policy with baselines.

**How it works**:

1. **Runs Multiple Episodes**:
   - For each policy (baselines + RL)
   - Collects metrics: cost, final SoC, undercharge rate

2. **Handles LSTM State** (for RecurrentPPO):
   ```python
   action, lstm_states = model.predict(
       obs,
       state=lstm_states,
       episode_start=episode_starts
   )
   ```

3. **Computes Metrics**:
   - Average cost per episode
   - Final SoC distribution
   - Undercharge rate (% episodes < 90% SoC)
   - Average episode length

4. **Prints Comparison Table**:
   - Shows which policy performs best
   - Helps validate RL learning

**Usage**:
```bash
python eval/evaluate.py --model models/recurrent_ppo_ev_charging --episodes 100
```

---

### Android Deployment (`android/`)

#### `android/inference_example.py` - Mobile Inference

**Purpose**: Demonstrates how to run ONNX model on Android for inference.

**Key Class**: `EVChargingInference`

**How it works**:

1. **Initialization**:
   ```python
   inference = EVChargingInference("models/ev_charging_policy.onnx")
   # Loads ONNX model
   # Configures NNAPI provider (for NPU)
   # Initializes LSTM state to zeros
   ```

2. **Episode Start**:
   ```python
   inference.reset()  # Reset LSTM state
   ```

3. **Per-Timestep Inference**:
   ```python
   action, probs = inference.predict(observation)
   # observation: [soc, time_elapsed, pv_current, ...]
   # Returns: action (0 or 1), probabilities
   ```

4. **LSTM State Management**:
   - State persists across timesteps within episode
   - Must reset at episode start
   - State shape: `(1, lstm_hidden_size * 2)`

**Android Integration**:
- Uses `onnxruntime` with NNAPI provider
- Falls back to CPU if NPU unavailable
- Batch size = 1 (single observation)

---

### Configuration (`configs/`)

#### `configs/default.yaml` - Central Configuration

**Purpose**: Single source of truth for all parameters.

**Sections**:

1. **Environment**:
   - EV parameters (capacity, efficiency, max power)
   - PV parameters (max output, noise, forecasts)
   - Grid pricing (ToU tiers)
   - Departure model (drift rate, user types, hazards)

2. **Recurrent PPO**:
   - Policy type (`MlpLstmPolicy`)
   - Hyperparameters (learning rate, batch size, etc.)
   - LSTM settings (hidden size, layers)

3. **Training**:
   - Total timesteps
   - Save/eval frequencies
   - Logging settings

4. **ONNX Export**:
   - Output path
   - Input shapes
   - Opset version

**Why YAML**: Easy to modify without code changes, version control friendly.

---

## 🔄 Complete Workflow

### 1. Training Phase (Snapdragon X PC)

```
┌─────────────────────────────────────────┐
│ 1. Load Config                          │
│    configs/default.yaml                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. Create Environment                    │
│    EVChargingEnv(config)                 │
│    - Random initial SoC                  │
│    - Hidden departure model              │
│    - PV forecasts                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. Create RecurrentPPO Agent            │
│    - MlpLstmPolicy                       │
│    - LSTM hidden size: 64                │
│    - CPU device                          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 4. Training Loop                         │
│    For each timestep:                    │
│    - Agent observes state                │
│    - Agent selects action                │
│    - Environment steps forward           │
│    - Reward calculated                   │
│    - LSTM state updated                  │
│    - Policy updated (PPO)                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 5. Save Model                            │
│    models/recurrent_ppo_ev_charging.zip  │
└─────────────────────────────────────────┘
```

### 2. Export Phase

```
┌─────────────────────────────────────────┐
│ 1. Load Trained Model                   │
│    RecurrentPPO.load(...)                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. Extract Policy Network               │
│    - Features extractor                  │
│    - MLP extractor                       │
│    - LSTM layer                          │
│    - Action network                      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. Wrap for ONNX                         │
│    OnnxableLSTMPolicy                    │
│    - Handles LSTM state as input         │
│    - Fixed input shapes                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 4. Export to ONNX                       │
│    torch.onnx.export(...)                │
│    → models/ev_charging_policy.onnx      │
└─────────────────────────────────────────┘
```

### 3. Inference Phase (Android)

```
┌─────────────────────────────────────────┐
│ 1. Load ONNX Model                      │
│    ort.InferenceSession(...)             │
│    - NNAPI provider (NPU)                │
│    - CPU fallback                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. Reset for New Episode                │
│    inference.reset()                     │
│    - LSTM state → zeros                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. For Each Timestep:                    │
│    a. Get observation from sensors      │
│       [soc, time, pv, forecasts, price] │
│    b. Run inference                      │
│       action, probs = predict(obs)       │
│    c. Update LSTM state                  │
│       (managed internally)               │
│    d. Execute action                     │
│       (control EV charger)               │
└─────────────────────────────────────────┘
```

---

## 🧠 How the LSTM Policy Works

### Architecture

```
Observation (12D)
    │
    ▼
Features Extractor (Flatten/Identity)
    │
    ▼
MLP Extractor (Shared Layers)
    │
    ▼
LSTM Layer (Hidden Size: 64)
    │
    ├─→ Hidden State (h_t)
    └─→ Cell State (c_t)
    │
    ▼
Action Network (Linear)
    │
    ▼
Action Probabilities (2D)
    │
    └─→ [P(solar_only), P(solar_grid)]
```

### Why LSTM?

1. **Temporal Dependencies**: 
   - Current decision depends on history
   - Need to remember past PV patterns
   - Need to track departure patterns

2. **Partial Observability**:
   - Departure probability NOT in observation
   - Must infer from interaction history
   - LSTM maintains hidden state

3. **Non-Stationary Learning**:
   - Departure patterns drift over time
   - LSTM can adapt to changing distributions
   - Memory helps track trends

### State Management

**During Training**:
- SB3 handles LSTM state automatically
- State reset at episode boundaries
- State passed through trajectory collection

**During Inference**:
- State must be managed manually
- Reset at episode start: `inference.reset()`
- State persists across timesteps
- Shape: `(1, hidden_size * 2)` = `(1, 128)` for hidden_size=64

---

## 📊 Environment Dynamics

### PV Generation

**Model**: Diurnal sine wave + AR(1) noise

```python
# Base curve (bell-shaped)
if 6 <= hour <= 18:
    pv_base = max_output * cos((hour - 12) * π/12)
else:
    pv_base = 0

# AR(1) correlated noise
noise_t = 0.7 * noise_{t-1} + random_normal(0, 0.5)
pv_output = clip(pv_base + noise_t, 0, max_output)
```

**Forecast**:
- Mean: Base forecast + 10% bias
- Std: Increases with current noise level
- Horizon: 4 steps (1 hour ahead)

### Departure Model

**Hidden Process**:
```python
# Sample user type (non-stationary distribution)
user_type = sample(weights=[w1, w2, w3])

# Get hazard rate
hazard = base_rates[user_type] * (1 + 2 * time_elapsed/max_steps)

# Sample departure
departs = random() < hazard
```

**Non-Stationary**:
- User type weights drift: `weights += normal(0, 0.01)`
- Distribution changes over training
- Agent must adapt online

### Charging Dynamics

**Action 0 (Solar Only)**:
```python
charging_power = min(pv_output, 7.4 kW)
grid_power = 0
cost = 0
```

**Action 1 (Solar + Grid)**:
```python
charging_power = 7.4 kW  # Max
grid_power = max(0, 7.4 - pv_output)
cost = grid_power * 0.25 hours * grid_price
```

**SoC Update**:
```python
energy_added = charging_power * 0.25 hours * 0.90 efficiency
soc_new = soc_old + energy_added / 75 kWh
```

---

## 🎯 Reward Function Explained

### Step Reward
```python
reward = -grid_energy_kwh * grid_price
```
- **Negative cost**: Minimizing cost = maximizing reward
- **Dense signal**: Every step provides feedback
- **Encourages**: Using free solar, avoiding expensive grid

### Terminal Reward
```python
if terminated and soc < 0.9:
    penalty = -100 * (0.9 - soc)
    reward += penalty
```
- **Large penalty**: Ensures target SoC is met
- **Proportional**: Larger deficit = larger penalty
- **One-time**: Only at episode end

### Total Episode Reward
```python
total_reward = sum(step_rewards) + terminal_reward
            = -total_cost - penalty_if_undercharged
```

**Optimal Strategy**:
- Minimize grid usage (especially during peak hours)
- Maximize solar usage
- Ensure 90% SoC before departure
- Balance immediate costs with future needs

---

## 🔧 Configuration Deep Dive

### Environment Parameters

```yaml
environment:
  ev_capacity_kwh: 75.0        # Battery size
  charging_efficiency: 0.90     # 90% efficiency loss
  max_charging_power_kw: 7.4    # Max charging rate
  target_soc: 0.90              # Goal: 90% charge
  lambda_penalty: 100.0         # Terminal penalty weight
  
  # PV Settings
  max_pv_output_kw: 10.0        # Peak solar output
  pv_noise_std: 0.5            # Forecast uncertainty
  pv_ar_coef: 0.7              # Noise correlation
  pv_forecast_bias: 0.1        # 10% overestimate
  
  # Departure Model
  departure_drift_rate: 0.01   # How fast patterns change
  num_user_types: 3            # Early/medium/late departers
  base_hazard_rates: [0.05, 0.10, 0.15]  # Per-step probabilities
```

### Training Parameters

```yaml
recurrent_ppo:
  policy: "MlpLstmPolicy"      # LSTM policy type
  learning_rate: 3e-4           # How fast to learn
  n_steps: 2048                 # Steps before update
  batch_size: 64                # Batch for updates
  n_epochs: 10                  # Update iterations
  gamma: 0.99                   # Future reward discount
  lstm_hidden_size: 64          # LSTM memory size
```

---

## 📱 Android Integration Details

### ONNX Model Inputs

1. **observation**: `(1, 12)` float32
   - Current state vector

2. **lstm_state**: `(1, 128)` float32 (for hidden_size=64)
   - Concatenated [h_t, c_t]
   - Must reset to zeros at episode start

### ONNX Model Outputs

1. **action_probs**: `(1, 2)` float32
   - `[P(action_0), P(action_1)]`
   - Select action = argmax(probs)

### State Management Pattern

```python
# Episode start
inference.reset()  # LSTM state = zeros

# Each timestep
action, probs = inference.predict(obs)
# LSTM state updated internally

# Episode end
# State automatically reset at next reset() call
```

---

## 🧪 Testing & Validation

### Environment Test

```python
from envs import EVChargingEnv
import yaml

config = yaml.safe_load(open('configs/default.yaml'))
env = EVChargingEnv(config['environment'])

obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step: SoC={info['soc']:.2f}, Cost=${info['step_cost']:.2f}")
    if done or truncated:
        break
```

### Baseline Comparison

```bash
python eval/evaluate.py --episodes 50
```

Shows:
- Which baseline performs best
- RL improvement over baselines
- Cost vs. satisfaction trade-offs

---

## 🐛 Common Issues & Solutions

### Issue: LSTM State Not Resetting

**Symptom**: Poor performance after first episode

**Solution**: Always call `inference.reset()` at episode start

### Issue: ONNX Export Fails

**Symptom**: Export error about LSTM structure

**Solution**: Check LSTM hidden size matches config

### Issue: Training Too Slow

**Symptom**: Takes hours to train

**Solution**: 
- Reduce `total_timesteps` for testing
- Reduce `n_steps` or `batch_size`
- Normal on CPU (ARM64)

### Issue: Poor Performance

**Symptom**: High costs or low SoC

**Solution**:
- Train longer (more timesteps)
- Adjust hyperparameters
- Check environment configuration

---

## 📚 Key Concepts Explained

### Recurrent vs Non-Recurrent

**Non-Recurrent (MLP)**:
- No memory
- Each observation independent
- Cannot learn temporal patterns

**Recurrent (LSTM)**:
- Maintains hidden state
- Remembers past observations
- Learns temporal dependencies

### Partial Observability

**Fully Observable**: Agent sees everything (e.g., departure time)

**Partially Observable**: Agent doesn't see departure probability
- Must infer from interaction
- LSTM helps maintain belief state

### Non-Stationary

**Stationary**: Departure patterns fixed

**Non-Stationary**: Patterns change over time
- User types drift
- Agent must adapt online
- LSTM helps track changes

---

## 🎓 Learning Process

### What Gets Learned

1. **PV Patterns**: When solar is available
2. **Pricing Patterns**: When grid is expensive
3. **Departure Patterns**: When EV typically leaves
4. **Planning**: Balance immediate vs future costs

### Training Progress

**Early**: Random actions, high costs, frequent failures

**Mid**: Starts using solar, avoids peak pricing

**Late**: Optimal policy, low costs, consistent success

---

## 📝 Summary

This codebase implements a complete RL system:

1. **Environment**: Realistic EV charging simulation with hidden departure
2. **Policy**: Recurrent LSTM that learns temporal patterns
3. **Training**: RecurrentPPO optimized for CPU
4. **Export**: ONNX format for mobile deployment
5. **Inference**: Android-ready with LSTM state management
6. **Evaluation**: Comparison with baselines

The system learns to optimize EV charging by balancing:
- **Cost**: Minimize grid usage
- **Sustainability**: Maximize solar usage
- **Satisfaction**: Ensure target charge before departure

All while adapting to unknown, changing departure patterns!
