"""
ONNX Inference Example for Samsung Android
===========================================

This script demonstrates how to run inference with the exported ONNX model
on Android (or any platform with ONNX Runtime).

Key Points for Mobile Inference:
--------------------------------
1. Load ONNX model with ONNX Runtime
2. Initialize LSTM hidden state (zeros) at episode start
3. Run inference: observation + state -> action + new_state
4. Pass new LSTM state to next timestep
5. Reset LSTM state when episode ends

This is a SIMULATION of the Android inference loop.
On actual Android, you would:
- Use onnxruntime-android package
- Load model from assets
- Integrate with real EV charging system

Usage:
    python inference_example.py --model exported_models/solar_ev_policy.onnx
"""

import os
import sys
import argparse
import logging
import numpy as np
from typing import Tuple, Dict, Optional

# ONNX Runtime (cross-platform, works on Android)
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class OnnxLSTMInferenceSession:
    """
    ONNX Runtime inference session for LSTM-based EV charging policy.
    
    This class handles:
    1. Loading the ONNX model
    2. Managing LSTM hidden state across timesteps
    3. Running inference to get actions
    
    Designed to be portable to Android with minimal changes.
    
    Android Integration:
    --------------------
    In Android (Kotlin/Java), you would use:
    
    ```kotlin
    import ai.onnxruntime.*
    
    class EvChargingPolicy(context: Context) {
        private val session: OrtSession
        private var lstmHidden: OnnxTensor
        private var lstmCell: OnnxTensor
        
        init {
            val env = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open("solar_ev_policy.onnx").readBytes()
            session = env.createSession(modelBytes)
            resetState()
        }
        
        fun resetState() {
            // Initialize LSTM state for new episode
            lstmHidden = OnnxTensor.createTensor(env, FloatArray(64))  // hidden_size
            lstmCell = OnnxTensor.createTensor(env, FloatArray(64))
        }
        
        fun getAction(observation: FloatArray): Int {
            val inputs = mapOf(
                "observation" to OnnxTensor.createTensor(env, observation),
                "lstm_hidden" to lstmHidden,
                "lstm_cell" to lstmCell
            )
            val outputs = session.run(inputs)
            val actionProbs = (outputs[0].value as Array<FloatArray>)[0]
            lstmHidden = outputs[1] as OnnxTensor
            lstmCell = outputs[2] as OnnxTensor
            return if (actionProbs[0] > actionProbs[1]) 0 else 1
        }
    }
    ```
    """
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX inference session.
        
        Args:
            model_path: Path to ONNX model file
        """
        logger.info(f"Loading ONNX model from: {model_path}")
        
        # Create ONNX Runtime session
        # Use CPU execution provider (works on mobile)
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get model metadata
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")
        
        # Extract dimensions from model
        obs_input = self.session.get_inputs()[0]
        lstm_input = self.session.get_inputs()[1]
        
        self.obs_dim = obs_input.shape[1]
        self.lstm_hidden_size = lstm_input.shape[2]
        
        logger.info(f"Observation dimension: {self.obs_dim}")
        logger.info(f"LSTM hidden size: {self.lstm_hidden_size}")
        
        # Initialize LSTM state
        self.reset_state()
    
    def reset_state(self):
        """
        Reset LSTM hidden state for new episode.
        
        CRITICAL: Call this at the start of each episode!
        The LSTM state carries information about the episode history.
        Starting a new episode with old state will confuse the policy.
        """
        self.lstm_hidden = np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
        self.lstm_cell = np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
        logger.debug("LSTM state reset to zeros")
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        Get action from policy given current observation.
        
        Args:
            observation: Current state observation (obs_dim,) or (1, obs_dim)
            deterministic: If True, take argmax action; else sample
        
        Returns:
            action: Discrete action (0 or 1)
            action_probs: Action probabilities [P(solar_only), P(solar+grid)]
        """
        # Ensure observation has batch dimension
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        # Ensure float32
        observation = observation.astype(np.float32)
        
        # Prepare inputs
        inputs = {
            'observation': observation,
            'lstm_hidden': self.lstm_hidden,
            'lstm_cell': self.lstm_cell,
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Parse outputs
        action_probs = outputs[0]  # (1, n_actions)
        self.lstm_hidden = outputs[1]  # (1, 1, hidden_size)
        self.lstm_cell = outputs[2]  # (1, 1, hidden_size)
        
        # Select action
        if deterministic:
            action = int(np.argmax(action_probs[0]))
        else:
            # Sample from distribution
            action = int(np.random.choice(len(action_probs[0]), p=action_probs[0]))
        
        return action, action_probs[0]
    
    def get_action_probs(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities without taking action (for analysis)."""
        _, probs = self.get_action(observation, deterministic=True)
        return probs


def simulate_episode(
    inference_session: OnnxLSTMInferenceSession,
    n_steps: int = 50,
    seed: int = 42
) -> Dict:
    """
    Simulate an episode using the ONNX model.
    
    This simulates what would happen on Android when receiving
    real sensor data from the EV charging system.
    
    Args:
        inference_session: ONNX inference session
        n_steps: Maximum number of steps
        seed: Random seed for observation generation
    
    Returns:
        Dictionary with episode statistics
    """
    np.random.seed(seed)
    
    # Reset LSTM state for new episode
    inference_session.reset_state()
    
    logger.info("=" * 50)
    logger.info("Simulating Episode")
    logger.info("=" * 50)
    
    # Track episode
    actions = []
    action_probs_history = []
    
    # Simulate observations (in real Android app, these come from sensors)
    current_soc = np.random.uniform(0.2, 0.5)
    start_hour = np.random.uniform(0, 24)
    
    for step in range(n_steps):
        # Generate observation (simulated)
        time_normalized = step / n_steps
        current_hour = (start_hour + step * 0.25) % 24  # 15-min steps
        
        # Solar output (simplified bell curve)
        if 6 <= current_hour <= 18:
            solar = 10 * np.sin(np.pi * (current_hour - 6) / 12)
            solar = max(0, solar + np.random.normal(0, 0.5))
        else:
            solar = 0
        
        # Price (TOU)
        if 16 <= current_hour < 21:
            price = 0.30
        elif current_hour >= 23 or current_hour < 9:
            price = 0.10
        else:
            price = 0.20
        
        # Forecast (simplified)
        forecast_mean = np.array([solar * 0.9] * 4)
        forecast_std = np.array([0.5] * 4)
        
        # Build observation vector
        observation = np.array([
            current_soc,
            time_normalized,
            solar,
            *forecast_mean,
            *forecast_std,
            price / 0.30  # Normalized price
        ], dtype=np.float32)
        
        # Get action from ONNX model
        action, probs = inference_session.get_action(observation)
        
        actions.append(action)
        action_probs_history.append(probs.copy())
        
        # Log every 10 steps
        if step % 10 == 0:
            logger.info(
                f"Step {step:3d} | SoC: {current_soc:.2%} | "
                f"Solar: {solar:.1f}kW | "
                f"Action: {'Solar' if action == 0 else 'Grid'} "
                f"(p={probs[action]:.2f})"
            )
        
        # Update SoC (simplified simulation)
        if action == 0:
            charge_power = min(solar, 7.4)
        else:
            charge_power = 7.4
        
        energy = charge_power * 0.25 * 0.9  # 15 min * efficiency
        current_soc = min(1.0, current_soc + energy / 75.0)
        
        # Check if charged
        if current_soc >= 0.9:
            logger.info(f"Target SoC reached at step {step}")
            break
    
    results = {
        'n_steps': len(actions),
        'final_soc': current_soc,
        'solar_only_ratio': sum(a == 0 for a in actions) / len(actions),
        'mean_confidence': np.mean([max(p) for p in action_probs_history]),
    }
    
    logger.info("")
    logger.info("Episode Summary:")
    logger.info(f"  Steps: {results['n_steps']}")
    logger.info(f"  Final SoC: {results['final_soc']:.1%}")
    logger.info(f"  Solar-only ratio: {results['solar_only_ratio']:.1%}")
    logger.info(f"  Mean action confidence: {results['mean_confidence']:.2f}")
    
    return results


def demo_android_integration():
    """
    Print example Android/Kotlin integration code.
    """
    code = '''
    // =========================================
    // Android Integration Example (Kotlin)
    // =========================================
    
    // 1. Add to build.gradle:
    // implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.0'
    
    // 2. Copy solar_ev_policy.onnx to app/src/main/assets/
    
    // 3. Create inference class:
    
    class EvChargingPolicy(private val context: Context) {
        private val ortEnv = OrtEnvironment.getEnvironment()
        private val session: OrtSession
        private var lstmHidden: FloatArray
        private var lstmCell: FloatArray
        private val hiddenSize = 64  // Must match training config
        
        init {
            // Load model from assets
            val modelBytes = context.assets.open("solar_ev_policy.onnx")
                .use { it.readBytes() }
            session = ortEnv.createSession(modelBytes)
            
            // Initialize LSTM state
            lstmHidden = FloatArray(hiddenSize)
            lstmCell = FloatArray(hiddenSize)
        }
        
        fun resetEpisode() {
            // Call when EV plugs in
            lstmHidden = FloatArray(hiddenSize)
            lstmCell = FloatArray(hiddenSize)
        }
        
        fun getChargingAction(
            currentSoc: Float,
            timeNormalized: Float,
            solarPower: Float,
            forecastMean: FloatArray,  // 4 values
            forecastStd: FloatArray,   // 4 values
            gridPrice: Float
        ): Int {
            // Build observation vector (12 values)
            val observation = floatArrayOf(
                currentSoc,
                timeNormalized,
                solarPower,
                *forecastMean,
                *forecastStd,
                gridPrice / 0.30f  // Normalize
            )
            
            // Create input tensors
            val obsShape = longArrayOf(1, 12)
            val stateShape = longArrayOf(1, 1, hiddenSize.toLong())
            
            val inputs = mapOf(
                "observation" to OnnxTensor.createTensor(ortEnv, 
                    observation.toTypedArray(), obsShape),
                "lstm_hidden" to OnnxTensor.createTensor(ortEnv,
                    lstmHidden.toTypedArray(), stateShape),
                "lstm_cell" to OnnxTensor.createTensor(ortEnv,
                    lstmCell.toTypedArray(), stateShape)
            )
            
            // Run inference
            val outputs = session.run(inputs)
            
            // Get action probabilities
            val actionProbs = (outputs[0].value as Array<FloatArray>)[0]
            
            // Update LSTM state for next call
            lstmHidden = (outputs[1].value as Array<Array<FloatArray>>)[0][0]
            lstmCell = (outputs[2].value as Array<Array<FloatArray>>)[0][0]
            
            // Return action: 0 = solar only, 1 = solar + grid
            return if (actionProbs[0] > actionProbs[1]) 0 else 1
        }
        
        fun close() {
            session.close()
            ortEnv.close()
        }
    }
    
    // 4. Usage in your charging controller:
    
    class ChargingController(context: Context) {
        private val policy = EvChargingPolicy(context)
        
        fun onEvPluggedIn() {
            policy.resetEpisode()
        }
        
        fun onTimerTick(sensorData: SensorData) {
            val action = policy.getChargingAction(
                currentSoc = sensorData.batterySoc,
                timeNormalized = sensorData.elapsedTime / 24f,
                solarPower = sensorData.solarGeneration,
                forecastMean = sensorData.solarForecastMean,
                forecastStd = sensorData.solarForecastStd,
                gridPrice = sensorData.currentGridPrice
            )
            
            if (action == 0) {
                setChargingMode(ChargingMode.SOLAR_ONLY)
            } else {
                setChargingMode(ChargingMode.SOLAR_PLUS_GRID)
            }
        }
    }
    '''
    
    print(code)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ONNX inference example for Android deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="exported_models/solar_ev_policy.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to simulate"
    )
    parser.add_argument(
        "--show-android",
        action="store_true",
        help="Show Android/Kotlin integration example"
    )
    
    args = parser.parse_args()
    
    if args.show_android:
        demo_android_integration()
        return
    
    # Check model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        logger.error("Run export_onnx.py first to create the ONNX model")
        return
    
    # Create inference session
    session = OnnxLSTMInferenceSession(args.model)
    
    # Run simulation episodes
    logger.info("")
    for ep in range(args.episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {ep + 1}/{args.episodes}")
        simulate_episode(session, seed=42 + ep)
    
    # Show Android integration info
    logger.info("\n" + "=" * 60)
    logger.info("For Android integration, run with --show-android flag")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
