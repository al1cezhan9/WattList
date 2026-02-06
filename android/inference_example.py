"""
Android ONNX Inference Example for EV Charging Policy.

This demonstrates how to run inference on Android using ONNX Runtime.
The LSTM state must be managed between timesteps.
"""

import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional


class EVChargingInference:
    """
    ONNX-based inference for EV charging policy on Android.
    
    Manages LSTM state between timesteps for recurrent policy.
    """
    
    def __init__(self, onnx_model_path: str, lstm_hidden_size: int = 64):
        """
        Initialize ONNX inference session.
        
        Args:
            onnx_model_path: Path to ONNX model file
            lstm_hidden_size: LSTM hidden size (must match training)
        """
        # Configure providers for Android
        providers = []
        
        # Try NNAPI provider for Android NPU acceleration
        available_providers = ort.get_available_providers()
        if 'NnapiExecutionProvider' in available_providers:
            providers.append('NnapiExecutionProvider')
            print("✅ NNAPI provider available (NPU acceleration)")
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        # Create inference session
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # LSTM state management
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_state = np.zeros((1, lstm_hidden_size * 2), dtype=np.float32)
        
        print(f"✅ ONNX model loaded: {onnx_model_path}")
        print(f"   Providers: {self.session.get_providers()}")
        print(f"   Inputs: {self.input_names}")
        print(f"   Outputs: {self.output_names}")
    
    def reset(self):
        """Reset LSTM state (call at start of new episode)."""
        self.lstm_state = np.zeros((1, self.lstm_hidden_size * 2), dtype=np.float32)
        print("🔄 LSTM state reset")
    
    def predict(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Run inference on single observation.
        
        Args:
            observation: Observation array (shape: [12] or [1, 12])
                          [soc, time_elapsed, pv_current, pv_forecast_mean (4), 
                           pv_forecast_std (4), grid_price]
        
        Returns:
            action: Selected action (0 or 1)
            action_probs: Action probabilities [P(action_0), P(action_1)]
        """
        # Ensure correct shape: (1, obs_dim) for batch-1 inference
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        observation = observation.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: observation,  # observation
                self.input_names[1]: self.lstm_state  # lstm_state
            }
        )
        
        # Get action probabilities
        action_probs = outputs[0][0]  # Remove batch dimension
        
        # Update LSTM state (if model returns it)
        # Note: For ONNX export, we may need to handle state update differently
        # This depends on how the ONNX model is structured
        
        # Select action (deterministic: argmax)
        action = int(np.argmax(action_probs))
        
        return action, action_probs
    
    def predict_with_state_update(self, observation: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Run inference and return updated LSTM state.
        
        This is needed if the ONNX model doesn't handle state internally.
        For now, we assume state is managed externally.
        
        Returns:
            action, action_probs, updated_lstm_state
        """
        action, probs = self.predict(observation)
        
        # In a real implementation, you might need to extract updated state
        # from the ONNX model outputs if it's exposed
        # For now, we'll manage state externally (reset at episode start)
        
        return action, probs, self.lstm_state.copy()


def example_usage():
    """Example usage of ONNX inference."""
    print("="*60)
    print("📱 Android ONNX Inference Example")
    print("="*60)
    
    # Initialize inference
    onnx_model_path = "models/ev_charging_policy.onnx"
    inference = EVChargingInference(onnx_model_path, lstm_hidden_size=64)
    
    # Simulate an episode
    print("\nSimulating episode...")
    inference.reset()
    
    # Example observations (normally from sensors)
    # Format: [soc, time_elapsed, pv_current, pv_forecast_mean (4), pv_forecast_std (4), grid_price]
    example_observations = [
        np.array([0.3, 0.0, 5.0, 6.0, 5.5, 5.0, 4.5, 0.5, 0.4, 0.3, 0.2, 0.20]),  # Step 1
        np.array([0.35, 0.01, 6.0, 7.0, 6.5, 6.0, 5.5, 0.5, 0.4, 0.3, 0.2, 0.20]),  # Step 2
        np.array([0.40, 0.02, 7.0, 8.0, 7.5, 7.0, 6.5, 0.5, 0.4, 0.3, 0.2, 0.20]),  # Step 3
    ]
    
    for i, obs in enumerate(example_observations):
        action, probs = inference.predict(obs)
        action_name = "Solar Only" if action == 0 else "Solar + Grid"
        print(f"\nStep {i+1}:")
        print(f"  Observation: SoC={obs[0]:.2f}, PV={obs[2]:.1f}kW, Price=${obs[-1]:.2f}/kWh")
        print(f"  Action: {action} ({action_name})")
        print(f"  Probabilities: Solar={probs[0]:.2f}, Grid={probs[1]:.2f}")
    
    print("\n" + "="*60)
    print("✅ Inference example complete")
    print("="*60)


if __name__ == "__main__":
    example_usage()
