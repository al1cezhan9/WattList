#!/usr/bin/env python3
"""
Simple Flask web app for Solar-Arbitrage EV Controller inference.
Run this to test ONNX model via web interface.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)

# Load ONNX model
onnx_model_path = 'solar_agent.onnx'
if not os.path.exists(onnx_model_path):
    print(f"⚠️ Warning: Model not found at {onnx_model_path}")
    print("   Please train the model first: python -m src.sb3_training")
    session = None
else:
    # Initialize ONNX Runtime session
    providers = []
    available_providers = ort.get_available_providers()
    
    # Try QNN provider first (for Snapdragon X PC)
    if 'QNNExecutionProvider' in available_providers:
        providers.append('QNNExecutionProvider')
        print("✅ QNN provider available")
    
    # Always add CPU as fallback
    providers.append('CPUExecutionProvider')
    
    try:
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"✅ Model loaded: {onnx_model_path}")
        print(f"   Providers: {session.get_providers()}")
        print(f"   Input: {input_name}, Output: {output_name}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        session = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/inference', methods=['POST'])
def inference():
    if session is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.json
        solar_output = float(data.get('solar_output', 0))
        ev_soc = float(data.get('ev_soc', 0))
        time_remaining = float(data.get('time_remaining', 0))
        grid_price = float(data.get('grid_price', 0))
        
        # Validate inputs
        if not (0 <= solar_output <= 10):
            return jsonify({'error': 'Solar output must be 0-10 kW'}), 400
        if not (0 <= ev_soc <= 1):
            return jsonify({'error': 'EV SoC must be 0.0-1.0'}), 400
        if not (0 <= time_remaining <= 24):
            return jsonify({'error': 'Time remaining must be 0-24 hours'}), 400
        if not (0 <= grid_price <= 0.5):
            return jsonify({'error': 'Grid price must be 0-0.5 $/kWh'}), 400
        
        # Prepare input: [solar_output, ev_soc, time_remaining, grid_price]
        observation = np.array([[solar_output, ev_soc, time_remaining, grid_price]], 
                              dtype=np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: observation})
        probabilities = outputs[0][0]
        
        # Select action
        action = int(np.argmax(probabilities))
        confidence = float(probabilities[action])
        
        result = {
            'action': action,
            'action_name': 'Solar Only' if action == 0 else 'Solar + Grid',
            'probabilities': {
                'solar_only': float(probabilities[0]),
                'solar_grid': float(probabilities[1])
            },
            'confidence': confidence,
            'recommendation': (
                'Charge using free solar energy only' if action == 0
                else 'Charge at maximum power (solar + grid)'
            )
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🌐 Solar-Arbitrage EV Controller Web App")
    print("="*60)
    print("\nStarting Flask server...")
    print("Open your browser to: http://localhost:5000")
    print("To access from phone: http://YOUR_IP:5000")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
