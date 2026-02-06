# Complete Workflow Guide

## 🔄 End-to-End Workflow

### Phase 1: Training (Snapdragon X PC)

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Environment Setup                                         │
│     ┌─────────────────────────────────────┐                  │
│     │  SolarArbitrageEnv                  │                  │
│     │  - Generates solar output           │                  │
│     │  - Tracks EV SoC                    │                  │
│     │  - Applies ToU pricing              │                  │
│     │  - Calculates rewards               │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ State: [solar, soc, time, price]              │
│              ▼                                                │
│  2. Agent Training                                            │
│     ┌─────────────────────────────────────┐                  │
│     │  PPO Agent (SB3)                     │                  │
│     │  - MlpPolicy network                 │                  │
│     │  - Learns optimal actions            │                  │
│     │  - Updates via PPO algorithm         │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ Action: 0 (Solar) or 1 (Solar+Grid)           │
│              ▼                                                │
│  3. Learning Loop                                             │
│     ┌─────────────────────────────────────┐                  │
│     │  For each timestep:                 │                  │
│     │  1. Agent selects action            │                  │
│     │  2. Environment steps forward       │                  │
│     │  3. Reward calculated               │                  │
│     │  4. Agent updates policy            │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ After 100k+ timesteps                         │
│              ▼                                                │
│  4. Model Export                                              │
│     ┌─────────────────────────────────────┐                  │
│     │  OnnxablePolicy                     │                  │
│     │  - Extracts actor network           │                  │
│     │  - Converts to ONNX format          │                  │
│     │  - Fixed shape (1, 4) → (1, 2)     │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ Output: solar_agent.onnx                       │
│              ▼                                                │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Deployment (Snapdragon 8 Elite NPU)

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PHASE                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Model Loading                                             │
│     ┌─────────────────────────────────────┐                  │
│     │  ONNX Runtime                       │                  │
│     │  - Loads solar_agent.onnx           │                  │
│     │  - Initializes QNN provider          │                  │
│     │  - Optimizes for NPU                 │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ Ready for inference                            │
│              ▼                                                │
│  2. Real-Time Inference                                      │
│     ┌─────────────────────────────────────┐                  │
│     │  Observation Input                  │                  │
│     │  [solar_output, ev_soc,            │                  │
│     │   time_remaining, grid_price]       │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ Shape: (1, 4)                                  │
│              ▼                                                │
│  3. NPU Execution                                            │
│     ┌─────────────────────────────────────┐                  │
│     │  Snapdragon 8 Elite NPU             │                  │
│     │  - Processes inference              │                  │
│     │  - Low latency (< 10ms)              │                  │
│     │  - Low power consumption             │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ Output: Action probabilities                  │
│              ▼                                                │
│  4. Action Selection                                         │
│     ┌─────────────────────────────────────┐                  │
│     │  Action Probabilities               │                  │
│     │  [P(action_0), P(action_1)]        │                  │
│     │  → Select argmax(action)            │                  │
│     └─────────────────────────────────────┘                  │
│              │                                                │
│              │ Action: 0 or 1                                 │
│              ▼                                                │
│  5. Control EV Charger                                       │
│     ┌─────────────────────────────────────┐                  │
│     │  EV Charging Controller              │                  │
│     │  - Action 0: Solar only             │                  │
│     │  - Action 1: Max charge (solar+grid)│                  │
│     └─────────────────────────────────────┘                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

### Training Data Flow

```
Environment State
    │
    ├─→ [solar_output: 5.2 kW]
    ├─→ [ev_soc: 0.45]
    ├─→ [time_remaining: 12.0 hours]
    └─→ [grid_price: $0.20/kWh]
         │
         ▼
    Agent (PPO)
         │
         ├─→ Action: 1 (Solar + Grid)
         │
         ▼
    Environment Step
         │
         ├─→ Charging: 7.4 kW
         ├─→ Solar used: 5.2 kW
         ├─→ Grid used: 2.2 kW
         ├─→ SoC increase: +0.088 (with 90% efficiency)
         ├─→ Time: -1 hour
         └─→ Reward: (1.5 × 5.2) - (2.2 × 0.20) = 7.36
         │
         ▼
    Next State + Reward
         │
         └─→ Agent updates policy
```

### Inference Data Flow

```
Real-World Sensors
    │
    ├─→ Solar panel output: 6.5 kW
    ├─→ EV SoC: 0.52 (from BMS)
    ├─→ Time until departure: 8.0 hours
    └─→ Current grid price: $0.30/kWh (peak)
         │
         ▼
    Observation Array: [6.5, 0.52, 8.0, 0.30]
         │
         ▼
    ONNX Model (NPU)
         │
         ├─→ Input shape: (1, 4)
         ├─→ Processing: < 10ms
         └─→ Output: [0.15, 0.85] (action probabilities)
         │
         ▼
    Action Selection: argmax([0.15, 0.85]) = 1
         │
         ▼
    EV Charger Control
         │
         ├─→ Set charging power: 7.4 kW
         ├─→ Use solar: 6.5 kW
         └─→ Use grid: 0.9 kW
```

## 🔄 Complete Cycle

### Single Episode Flow

```
1. INITIALIZE
   ├─→ Random SoC: 0.3
   ├─→ Random time: 18 hours
   └─→ Random hour: 8 AM
   
2. FOR EACH HOUR (until departure):
   
   a. OBSERVE
      ├─→ Get solar output (from model/environment)
      ├─→ Get current SoC (from EV)
      ├─→ Calculate time remaining
      └─→ Get grid price (from ToU schedule)
   
   b. DECIDE
      ├─→ Agent receives observation
      ├─→ Model outputs action probabilities
      └─→ Select action (0 or 1)
   
   c. ACT
      ├─→ If action 0: Charge with solar only
      └─→ If action 1: Charge at max (solar + grid)
   
   d. UPDATE
      ├─→ Update SoC (accounting for efficiency)
      ├─→ Decrease time remaining
      ├─→ Calculate reward
      └─→ Move to next hour
   
3. TERMINATE
   ├─→ When time_remaining <= 0
   ├─→ Check final SoC
   └─→ Apply terminal penalty if SoC < 0.9
```

## 🎯 Decision Making Process

### What the Agent Considers

**Short-term factors:**
- Current solar availability
- Current grid price
- Current SoC level

**Long-term factors:**
- Time until departure
- Expected future solar (diurnal pattern)
- Expected future prices (ToU schedule)
- Need to reach 90% SoC

### Example Decision Scenarios

**Scenario 1: Morning, Low SoC, Many Hours**
```
State: [2.0 kW solar, 0.3 SoC, 18 hours, $0.20/kWh]
Decision: Action 0 (Solar only)
Reason: Plenty of time, can wait for more solar later
```

**Scenario 2: Afternoon, Medium SoC, Few Hours**
```
State: [8.5 kW solar, 0.6 SoC, 4 hours, $0.30/kWh]
Decision: Action 1 (Max charge)
Reason: Need to charge quickly, solar is abundant
```

**Scenario 3: Evening, Low SoC, Peak Price**
```
State: [0.5 kW solar, 0.4 SoC, 6 hours, $0.30/kWh]
Decision: Action 0 (Solar only) or wait
Reason: Avoid expensive peak pricing, charge later
```

## 📈 Performance Metrics

### Training Metrics

- **Episode Reward**: Total reward per episode (should increase)
- **Episode Length**: Hours per episode (varies)
- **Final SoC**: Battery level at departure (target: ≥ 0.9)
- **Grid Cost**: Total cost per episode (should decrease)
- **Solar Utilization**: % of solar used vs available

### Deployment Metrics

- **Inference Latency**: < 10ms on NPU
- **Power Consumption**: Low (NPU optimized)
- **Action Accuracy**: Matches training performance
- **Cost Savings**: Compared to naive charging

## 🔧 Integration Points

### With Real EV System

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Solar     │────▶│   ONNX       │────▶│   EV        │
│   Sensors   │     │   Agent      │     │   Charger   │
└─────────────┘     └──────────────┘     └─────────────┘
     │                    │                    │
     │                    │                    │
     ▼                    ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Grid      │     │   EV BMS     │     │   ToU       │
│   Meter     │     │   (SoC)      │     │   Schedule  │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Required Inputs

1. **Solar Output**: From solar panel sensors or forecast
2. **EV SoC**: From EV Battery Management System (BMS)
3. **Time Remaining**: User input or calendar integration
4. **Grid Price**: From utility ToU schedule or API

### Outputs

1. **Action**: 0 (Solar only) or 1 (Max charge)
2. **Charging Power**: Setpoint for EV charger
3. **Confidence**: Action probability (for monitoring)

## 🎓 Learning Process

### What Gets Learned

**Policy Network (Actor):**
- Maps states → action probabilities
- Learns when to use solar vs grid
- Balances cost, time, and satisfaction

**Value Network (Critic):**
- Estimates expected future rewards
- Helps guide policy updates
- Provides value estimates for states

### Training Progress

**Early Training:**
- Random actions
- Low rewards
- Frequent failures (SoC < 90%)

**Mid Training:**
- Starts recognizing patterns
- Uses solar when available
- Avoids peak pricing

**Late Training:**
- Optimal policy
- High rewards
- Consistent success (SoC ≥ 90%)

## 📝 Summary

1. **Train**: Agent learns optimal charging strategy
2. **Export**: Convert to ONNX for NPU deployment
3. **Deploy**: Run on Snapdragon 8 Elite NPU
4. **Integrate**: Connect to real EV charging system
5. **Monitor**: Track performance and costs

The complete system optimizes EV charging to maximize solar usage, minimize costs, and ensure user satisfaction!
