# Web-Based Deployment Guide

## 🌐 Simple Web UI for Testing

A lightweight web-based UI for testing ONNX inference without building a native app.

## 🚀 Option 1: Python Flask Web App

### Step 1: Create Flask App

**`web_app.py`:**

```python
from flask import Flask, render_template, request, jsonify
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)

# Load ONNX model
onnx_model_path = 'solar_agent.onnx'
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"Model not found: {onnx_model_path}")

# Initialize ONNX Runtime session
providers = []
if 'QNNExecutionProvider' in ort.get_available_providers():
    providers.append('QNNExecutionProvider')
providers.append('CPUExecutionProvider')

session = ort.InferenceSession(onnx_model_path, providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"✅ Model loaded: {onnx_model_path}")
print(f"   Providers: {session.get_providers()}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/inference', methods=['POST'])
def inference():
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
        
        # Prepare input
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
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### Step 2: Create HTML Template

**`templates/index.html`:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solar-Arbitrage EV Controller</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            font-size: 28px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        
        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 10px;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .result-card {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
            display: none;
        }
        
        .result-card.show {
            display: block;
        }
        
        .result-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        .action-name {
            font-size: 24px;
            color: #28a745;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .recommendation {
            color: #666;
            margin: 10px 0;
        }
        
        .confidence-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .probabilities {
            margin-top: 15px;
            font-size: 14px;
            color: #666;
        }
        
        .error {
            margin-top: 20px;
            padding: 15px;
            background: #fee;
            border-left: 4px solid #dc3545;
            border-radius: 4px;
            color: #dc3545;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .loading {
            text-align: center;
            margin: 20px 0;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌞 Solar-Arbitrage EV Controller</h1>
        
        <div class="form-group">
            <label for="solar">Solar Output (kW)</label>
            <input type="number" id="solar" step="0.1" min="0" max="10" value="5.0">
        </div>
        
        <div class="form-group">
            <label for="soc">EV State of Charge (0.0 - 1.0)</label>
            <input type="number" id="soc" step="0.01" min="0" max="1" value="0.5">
        </div>
        
        <div class="form-group">
            <label for="time">Time Until Departure (hours)</label>
            <input type="number" id="time" step="0.1" min="0" max="24" value="12.0">
        </div>
        
        <div class="form-group">
            <label for="price">Grid Price ($/kWh)</label>
            <input type="number" id="price" step="0.01" min="0" max="0.5" value="0.20">
        </div>
        
        <button id="runBtn" onclick="runInference()">Run Inference</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Running inference...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result-card" id="resultCard">
            <div class="result-title">Recommendation</div>
            <div class="action-name" id="actionName"></div>
            <div class="recommendation" id="recommendation"></div>
            
            <div style="margin-top: 20px;">
                <div style="margin-bottom: 5px;">Confidence</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill"></div>
                </div>
            </div>
            
            <div class="probabilities" id="probabilities"></div>
        </div>
    </div>
    
    <script>
        async function runInference() {
            const solar = parseFloat(document.getElementById('solar').value);
            const soc = parseFloat(document.getElementById('soc').value);
            const time = parseFloat(document.getElementById('time').value);
            const price = parseFloat(document.getElementById('price').value);
            
            // Validate inputs
            if (isNaN(solar) || solar < 0 || solar > 10) {
                showError('Solar output must be 0-10 kW');
                return;
            }
            if (isNaN(soc) || soc < 0 || soc > 1) {
                showError('EV SoC must be 0.0-1.0');
                return;
            }
            if (isNaN(time) || time < 0 || time > 24) {
                showError('Time remaining must be 0-24 hours');
                return;
            }
            if (isNaN(price) || price < 0 || price > 0.5) {
                showError('Grid price must be 0-0.5 $/kWh');
                return;
            }
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('error').classList.remove('show');
            document.getElementById('resultCard').classList.remove('show');
            document.getElementById('runBtn').disabled = true;
            
            try {
                const response = await fetch('/api/inference', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        solar_output: solar,
                        ev_soc: soc,
                        time_remaining: time,
                        grid_price: price
                    })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Inference failed');
                }
                
                // Display results
                document.getElementById('actionName').textContent = data.action_name;
                document.getElementById('recommendation').textContent = data.recommendation;
                
                const confidencePercent = Math.round(data.confidence * 100);
                const confidenceFill = document.getElementById('confidenceFill');
                confidenceFill.style.width = confidencePercent + '%';
                confidenceFill.textContent = confidencePercent + '%';
                
                document.getElementById('probabilities').innerHTML = 
                    `Probabilities:<br>` +
                    `Solar Only: ${Math.round(data.probabilities.solar_only * 100)}%<br>` +
                    `Solar + Grid: ${Math.round(data.probabilities.solar_grid * 100)}%`;
                
                document.getElementById('resultCard').classList.add('show');
                
            } catch (error) {
                showError(error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('runBtn').disabled = false;
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = 'Error: ' + message;
            errorDiv.classList.add('show');
        }
    </script>
</body>
</html>
```

### Step 3: Run Web App

```bash
# Install Flask if not already installed
pip install flask

# Run the app
python web_app.py

# Open browser to http://localhost:5000
```

## 📱 Access from Phone

### Option A: Same Network

1. Find your computer's IP address:
   ```bash
   # Linux/Mac
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```

2. On your phone, open browser to: `http://YOUR_IP:5000`

### Option B: ngrok (Public URL)

```bash
# Install ngrok
# Download from https://ngrok.com/

# Expose local server
ngrok http 5000

# Use the provided URL on your phone
```

## 🎯 Summary

**For Android Native App:**
- Use `DEPLOYMENT_ANDROID.md` - Full Kotlin/Java implementation
- Best performance, native UI, direct NPU access

**For Flutter App:**
- Use `DEPLOYMENT_FLUTTER.md` - Cross-platform solution
- Single codebase for Android/iOS

**For Quick Testing:**
- Use `DEPLOYMENT_WEB.md` - Simple web interface
- Fastest to set up, works on any device with browser

Choose based on your needs!
