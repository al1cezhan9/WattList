"""
ONNX Export for RecurrentPPO Policy
====================================

This script exports a trained RecurrentPPO model to ONNX format
for deployment on Samsung Android with ONNX Runtime Mobile.

Key Challenges with LSTM Export:
--------------------------------
1. LSTM has hidden state that persists between timesteps
2. ONNX requires explicit handling of recurrent state
3. Mobile inference needs fixed input shapes

Solution:
---------
We export ONLY the policy network (not value function) with:
- Fixed batch size of 1
- Explicit hidden state inputs/outputs
- Optimized ONNX graph for mobile

Usage:
    python export_onnx.py --model trained_models/run_xxx/best_model.zip --output solar_policy.onnx
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import checker, helper
from typing import Tuple, Optional

# sb3-contrib for loading RecurrentPPO
from sb3_contrib import RecurrentPPO

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.ev_charging_env import EVChargingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class OnnxableLSTMPolicy(nn.Module):
    """
    ONNX-compatible wrapper for RecurrentPPO's LSTM policy.
    
    This wrapper handles:
    1. Extracting just the policy (actor) network
    2. Managing LSTM hidden state explicitly
    3. Providing fixed input shapes for mobile inference
    
    Input:
        observation: (1, obs_dim) - current state
        lstm_hidden: (1, 1, hidden_size) - LSTM hidden state
        lstm_cell: (1, 1, hidden_size) - LSTM cell state
    
    Output:
        action_probs: (1, n_actions) - action probabilities
        new_lstm_hidden: (1, 1, hidden_size) - updated hidden state
        new_lstm_cell: (1, 1, hidden_size) - updated cell state
    """
    
    def __init__(self, recurrent_ppo: RecurrentPPO):
        super().__init__()
        
        # Extract policy components from RecurrentPPO
        policy = recurrent_ppo.policy
        
        # Feature extractor (MLP that processes observations)
        self.features_extractor = policy.features_extractor
        
        # LSTM layer
        self.lstm = policy.lstm_actor
        
        # Action network (outputs logits)
        self.action_net = policy.action_net
        
        # Store dimensions for reference
        self.lstm_hidden_size = policy.lstm_hidden_size
        self.observation_dim = policy.observation_space.shape[0]
        self.n_actions = policy.action_space.n
        
        logger.info(f"Extracted policy components:")
        logger.info(f"  Observation dim: {self.observation_dim}")
        logger.info(f"  LSTM hidden size: {self.lstm_hidden_size}")
        logger.info(f"  Number of actions: {self.n_actions}")
    
    def forward(
        self,
        observation: torch.Tensor,
        lstm_hidden: torch.Tensor,
        lstm_cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.
        
        Args:
            observation: (1, obs_dim) current observation
            lstm_hidden: (1, 1, hidden_size) LSTM hidden state
            lstm_cell: (1, 1, hidden_size) LSTM cell state
        
        Returns:
            action_probs: (1, n_actions) softmax probabilities
            new_hidden: (1, 1, hidden_size) updated hidden state
            new_cell: (1, 1, hidden_size) updated cell state
        """
        # Extract features from observation
        # Shape: (1, obs_dim) -> (1, features_dim)
        features = self.features_extractor(observation)
        
        # Add sequence dimension for LSTM: (1, features_dim) -> (1, 1, features_dim)
        features = features.unsqueeze(0)
        
        # Pack LSTM state
        # lstm_hidden and lstm_cell are already (1, 1, hidden_size)
        lstm_state = (lstm_hidden, lstm_cell)
        
        # LSTM forward pass
        # Input: (seq_len=1, batch=1, input_size)
        # Output: (seq_len=1, batch=1, hidden_size), (h_n, c_n)
        lstm_output, (new_hidden, new_cell) = self.lstm(features, lstm_state)
        
        # Remove sequence dimension: (1, 1, hidden_size) -> (1, hidden_size)
        lstm_output = lstm_output.squeeze(0)
        
        # Get action logits
        action_logits = self.action_net(lstm_output)
        
        # Convert to probabilities
        action_probs = torch.softmax(action_logits, dim=-1)
        
        return action_probs, new_hidden, new_cell
    
    def get_initial_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get zero-initialized LSTM state for episode start."""
        hidden = torch.zeros(1, 1, self.lstm_hidden_size)
        cell = torch.zeros(1, 1, self.lstm_hidden_size)
        return hidden, cell


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 14,
    verify: bool = True
) -> str:
    """
    Export trained RecurrentPPO model to ONNX format.
    
    Args:
        model_path: Path to saved RecurrentPPO model (.zip)
        output_path: Path for output ONNX file
        opset_version: ONNX opset version (14+ recommended for LSTM)
        verify: Whether to verify the exported model
    
    Returns:
        Path to exported ONNX model
    """
    logger.info("=" * 60)
    logger.info("ONNX Export for RecurrentPPO")
    logger.info("=" * 60)
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    logger.info(f"Loading model from: {model_path}")
    
    # Load with CPU (no CUDA needed for export)
    model = RecurrentPPO.load(model_path, device='cpu')
    model.policy.eval()
    
    logger.info("Model loaded successfully")
    
    # =========================================================================
    # CREATE ONNX-COMPATIBLE WRAPPER
    # =========================================================================
    logger.info("Creating ONNX-compatible policy wrapper...")
    
    onnx_policy = OnnxableLSTMPolicy(model)
    onnx_policy.eval()
    
    # =========================================================================
    # PREPARE DUMMY INPUTS
    # =========================================================================
    obs_dim = onnx_policy.observation_dim
    hidden_size = onnx_policy.lstm_hidden_size
    
    # Create dummy inputs with correct shapes
    dummy_obs = torch.randn(1, obs_dim)
    dummy_hidden = torch.zeros(1, 1, hidden_size)
    dummy_cell = torch.zeros(1, 1, hidden_size)
    
    logger.info(f"Dummy observation shape: {dummy_obs.shape}")
    logger.info(f"Dummy LSTM hidden shape: {dummy_hidden.shape}")
    logger.info(f"Dummy LSTM cell shape: {dummy_cell.shape}")
    
    # =========================================================================
    # TEST FORWARD PASS
    # =========================================================================
    logger.info("Testing forward pass...")
    
    with torch.no_grad():
        action_probs, new_hidden, new_cell = onnx_policy(
            dummy_obs, dummy_hidden, dummy_cell
        )
    
    logger.info(f"Action probs shape: {action_probs.shape}")
    logger.info(f"Action probs: {action_probs.numpy()}")
    
    # =========================================================================
    # EXPORT TO ONNX
    # =========================================================================
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Export with explicit input/output names for mobile inference
    torch.onnx.export(
        onnx_policy,
        (dummy_obs, dummy_hidden, dummy_cell),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimize for mobile
        input_names=['observation', 'lstm_hidden', 'lstm_cell'],
        output_names=['action_probs', 'new_lstm_hidden', 'new_lstm_cell'],
        dynamic_axes=None,  # Fixed batch size for mobile
    )
    
    logger.info(f"Model exported to: {output_path}")
    
    # =========================================================================
    # VERIFY ONNX MODEL
    # =========================================================================
    if verify:
        logger.info("Verifying ONNX model...")
        
        # Load and check
        onnx_model = onnx.load(output_path)
        checker.check_model(onnx_model)
        
        # Print model info
        logger.info(f"ONNX model inputs:")
        for input in onnx_model.graph.input:
            logger.info(f"  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
        
        logger.info(f"ONNX model outputs:")
        for output in onnx_model.graph.output:
            logger.info(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
        
        # Test with ONNX Runtime
        logger.info("Testing with ONNX Runtime...")
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path)
        
        # Run inference
        ort_inputs = {
            'observation': dummy_obs.numpy(),
            'lstm_hidden': dummy_hidden.numpy(),
            'lstm_cell': dummy_cell.numpy(),
        }
        ort_outputs = session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        torch_probs = action_probs.numpy()
        onnx_probs = ort_outputs[0]
        
        max_diff = np.abs(torch_probs - onnx_probs).max()
        logger.info(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            logger.info("✓ ONNX model verified successfully!")
        else:
            logger.warning(f"⚠ Large numerical difference: {max_diff}")
    
    # =========================================================================
    # PRINT USAGE INSTRUCTIONS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To use this model on Android:")
    logger.info("1. Copy the .onnx file to your Android assets")
    logger.info("2. Add onnxruntime-android to your dependencies")
    logger.info("3. Initialize LSTM state at episode start (zeros)")
    logger.info("4. Pass observation and state to get action")
    logger.info("5. Update LSTM state for next timestep")
    logger.info("")
    
    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export RecurrentPPO model to ONNX for mobile inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exported_models/solar_ev_policy.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX verification"
    )
    
    args = parser.parse_args()
    
    export_to_onnx(
        args.model,
        args.output,
        args.opset,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
