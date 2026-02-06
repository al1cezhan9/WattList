"""
Export trained Recurrent PPO policy to ONNX format for mobile deployment.

This exports ONLY the policy network (actor) with LSTM, suitable for
inference on Android devices via ONNX Runtime.
"""

import os
import yaml
import torch
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class OnnxableLSTMPolicy(torch.nn.Module):
    """
    ONNX-compatible wrapper for LSTM policy.
    Handles LSTM state management for inference.
    
    Note: For RecurrentPPO, the policy structure is:
    features_extractor -> mlp_extractor (shared) -> lstm_actor -> action_net
    """
    
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        
        # Extract components from MlpLstmPolicy
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        
        # Get LSTM layer - check common locations
        if hasattr(policy, 'lstm_actor'):
            self.lstm = policy.lstm_actor
        elif hasattr(policy.mlp_extractor, 'lstm_actor'):
            self.lstm = policy.mlp_extractor.lstm_actor
        else:
            # Try to find LSTM in the network
            for name, module in policy.named_modules():
                if isinstance(module, torch.nn.LSTM):
                    self.lstm = module
                    break
            else:
                raise ValueError("Could not find LSTM layer in policy. Policy structure: " + 
                               str([name for name, _ in policy.named_modules()]))
        
        self.action_net = policy.action_net
    
    def forward(self, observation: torch.Tensor, lstm_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ONNX export.
        
        Args:
            observation: Input observation (batch_size, obs_dim)
            lstm_states: LSTM hidden states (batch_size, lstm_hidden_size * 2)
                        Format: [h_0, c_0] concatenated
        
        Returns:
            action_probs: Action probabilities (batch_size, num_actions)
        """
        # Extract features
        features = self.features_extractor(observation)
        
        # Process through MLP extractor to get latent representation
        latent_pi = self.mlp_extractor.forward_actor(features)
        
        # Prepare LSTM states
        # lstm_states shape: (batch, hidden_size * 2)
        # Split into h and c: (batch, hidden_size) each
        batch_size = observation.shape[0]
        lstm_hidden_size = self.lstm.hidden_size
        
        h_t = lstm_states[:, :lstm_hidden_size].unsqueeze(0)  # (1, batch, hidden)
        c_t = lstm_states[:, lstm_hidden_size:].unsqueeze(0)  # (1, batch, hidden)
        
        # LSTM forward
        # Input: (seq_len=1, batch, features)
        lstm_input = latent_pi.unsqueeze(0)  # (1, batch, features)
        lstm_out, (h_new, c_new) = self.lstm(lstm_input, (h_t, c_t))
        
        # Get action logits from LSTM output
        # lstm_out shape: (1, batch, hidden) -> squeeze to (batch, hidden)
        action_logits = self.action_net(lstm_out.squeeze(0))
        
        # Apply softmax to get probabilities
        action_probs = torch.softmax(action_logits, dim=-1)
        
        return action_probs


def export_to_onnx(model_path: str, config_path: str = "configs/default.yaml",
                   output_path: str = None):
    """
    Export trained RecurrentPPO model to ONNX format.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        output_path: Output ONNX file path (optional)
    """
    config = load_config(config_path)
    onnx_config = config.get('onnx_export', {})
    
    if output_path is None:
        output_path = onnx_config.get('output_path', 'models/ev_charging_policy.onnx')
    
    print("="*60)
    print("📦 Exporting Recurrent PPO Policy to ONNX")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    
    # Load trained model
    print("\nLoading trained model...")
    model = RecurrentPPO.load(model_path, device='cpu')
    print("✅ Model loaded")
    
    # Create ONNX-compatible wrapper
    print("\nCreating ONNX-compatible policy wrapper...")
    onnxable_policy = OnnxableLSTMPolicy(model.policy)
    onnxable_policy.eval()
    print("✅ Policy wrapper created")
    
    # Get LSTM hidden size
    lstm_hidden_size = config['recurrent_ppo']['lstm_hidden_size']
    obs_dim = config['onnx_export']['input_shape'][1]
    
    # Create dummy inputs
    dummy_obs = torch.randn(onnx_config['input_shape'], dtype=torch.float32)
    dummy_lstm_state = torch.zeros(1, lstm_hidden_size * 2, dtype=torch.float32)
    
    print(f"\nInput shapes:")
    print(f"  Observation: {dummy_obs.shape}")
    print(f"  LSTM state: {dummy_lstm_state.shape}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        torch.onnx.export(
            onnxable_policy,
            (dummy_obs, dummy_lstm_state),
            output_path,
            export_params=True,
            opset_version=onnx_config.get('opset_version', 11),
            do_constant_folding=True,
            input_names=['observation', 'lstm_state'],
            output_names=['action_probs'],
            dynamic_axes=None,  # Fixed shape for mobile
            verbose=False
        )
        
        print(f"✅ ONNX model exported successfully!")
        print(f"   Saved to: {output_path}")
        
        # Verify ONNX model
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
            
            # Test inference
            test_obs = np.random.randn(*onnx_config['input_shape']).astype(np.float32)
            test_lstm = np.zeros((1, lstm_hidden_size * 2), dtype=np.float32)
            
            outputs = session.run(None, {
                'observation': test_obs,
                'lstm_state': test_lstm
            })
            
            print(f"\n✅ ONNX model verification successful")
            print(f"   Output shape: {outputs[0].shape}")
            
        except ImportError:
            print("\n⚠️ onnxruntime not available for verification")
        except Exception as e:
            print(f"\n⚠️ ONNX verification failed: {e}")
        
    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        raise
    
    print("\n" + "="*60)
    print("✅ Export complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Recurrent PPO to ONNX')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file path')
    
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.config, args.output)
