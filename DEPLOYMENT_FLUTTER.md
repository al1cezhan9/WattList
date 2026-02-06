# Flutter Deployment Guide - Snapdragon 8 Elite NPU

## 📱 Flutter App for Solar-Arbitrage EV Controller

This guide shows how to create a Flutter app that runs ONNX inference on Snapdragon NPU.

## 🚀 Setup

### Step 1: Create Flutter Project

```bash
flutter create solar_arbitrage_app
cd solar_arbitrage_app
```

### Step 2: Add Dependencies

**`pubspec.yaml`:**

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # ONNX Runtime for Flutter
  onnxruntime: ^1.15.0
  
  # State management
  provider: ^6.1.1
  
  # UI components
  cupertino_icons: ^1.0.6
```

### Step 3: Add ONNX Model

**Project Structure:**
```
assets/
  solar_agent.onnx  ← Copy your model here
```

**`pubspec.yaml`:**
```yaml
flutter:
  assets:
    - assets/solar_agent.onnx
```

### Step 4: Create ONNX Service

**`lib/services/onnx_service.dart`:**

```dart
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class OnnxService {
  OrtEnv? _ortEnv;
  OrtSession? _ortSession;
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Initialize ONNX Runtime environment
      _ortEnv = OrtEnv();
      
      // Load model from assets
      final modelBytes = await rootBundle.load('assets/solar_agent.onnx');
      
      // Create session options
      final sessionOptions = OrtSessionOptions();
      
      // Try to use NNAPI (Android Neural Networks API)
      // This will use NPU on Snapdragon devices
      try {
        sessionOptions.addNnapi();
      } catch (e) {
        print('NNAPI not available, using CPU: $e');
      }
      
      // Create session
      _ortSession = _ortEnv!.createSessionFromBuffer(
        modelBytes.buffer.asUint8List(),
        sessionOptions,
      );
      
      _isInitialized = true;
      print('✅ ONNX model loaded successfully');
      
    } catch (e) {
      print('❌ Failed to load ONNX model: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>> predict({
    required double solarOutput,
    required double evSoc,
    required double timeRemaining,
    required double gridPrice,
  }) async {
    if (!_isInitialized || _ortSession == null) {
      throw Exception('Model not initialized');
    }

    try {
      // Prepare input: [solar_output, ev_soc, time_remaining, grid_price]
      final inputArray = [
        [solarOutput, evSoc, timeRemaining, gridPrice]
      ];
      
      // Create input tensor
      final inputTensor = OrtValueTensor.createTensorWithDataAsList(
        _ortEnv!,
        inputArray,
      );
      
      // Run inference
      final inputs = {
        _ortSession!.inputNames[0]: inputTensor,
      };
      
      final outputs = _ortSession!.run(OrtValueTensor.createTensorMap(inputs));
      
      // Get output probabilities
      final outputTensor = outputs[_ortSession!.outputNames[0]]!;
      final probabilities = outputTensor.value as List<List<List<double>>>;
      final probs = probabilities[0][0];
      
      // Select action
      final action = probs[0] > probs[1] ? 0 : 1;
      final confidence = probs[action];
      
      // Clean up
      inputTensor.release();
      outputs.values.forEach((tensor) => tensor.release());
      
      return {
        'action': action,
        'actionName': action == 0 ? 'Solar Only' : 'Solar + Grid',
        'probabilities': probs,
        'confidence': confidence,
        'recommendation': action == 0
            ? 'Charge using free solar energy only'
            : 'Charge at maximum power (solar + grid)',
      };
      
    } catch (e) {
      print('Inference failed: $e');
      rethrow;
    }
  }

  void dispose() {
    _ortSession?.release();
    _ortEnv?.release();
    _isInitialized = false;
  }
}
```

### Step 5: Create Provider/ViewModel

**`lib/providers/inference_provider.dart`:**

```dart
import 'package:flutter/foundation.dart';
import '../services/onnx_service.dart';

class InferenceResult {
  final int action;
  final String actionName;
  final List<double> probabilities;
  final double confidence;
  final String recommendation;

  InferenceResult({
    required this.action,
    required this.actionName,
    required this.probabilities,
    required this.confidence,
    required this.recommendation,
  });
}

class InferenceProvider with ChangeNotifier {
  final OnnxService _onnxService = OnnxService();
  bool _isInitialized = false;
  bool _isLoading = false;
  InferenceResult? _result;
  String? _error;

  bool get isInitialized => _isInitialized;
  bool get isLoading => _isLoading;
  InferenceResult? get result => _result;
  String? get error => _error;

  Future<void> initialize() async {
    if (_isInitialized) return;
    
    try {
      await _onnxService.initialize();
      _isInitialized = true;
      notifyListeners();
    } catch (e) {
      _error = 'Failed to initialize: $e';
      notifyListeners();
    }
  }

  Future<void> runInference({
    required double solarOutput,
    required double evSoc,
    required double timeRemaining,
    required double gridPrice,
  }) async {
    if (!_isInitialized) {
      await initialize();
    }

    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      final prediction = await _onnxService.predict(
        solarOutput: solarOutput,
        evSoc: evSoc,
        timeRemaining: timeRemaining,
        gridPrice: gridPrice,
      );

      _result = InferenceResult(
        action: prediction['action'] as int,
        actionName: prediction['actionName'] as String,
        probabilities: prediction['probabilities'] as List<double>,
        confidence: prediction['confidence'] as double,
        recommendation: prediction['recommendation'] as String,
      );
    } catch (e) {
      _error = 'Inference failed: $e';
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _onnxService.dispose();
    super.dispose();
  }
}
```

### Step 6: Create UI

**`lib/main.dart`:**

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/inference_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => InferenceProvider()..initialize(),
      child: MaterialApp(
        title: 'Solar-Arbitrage EV Controller',
        theme: ThemeData(
          primarySwatch: Colors.green,
          useMaterial3: true,
        ),
        home: const MainScreen(),
      ),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final _solarController = TextEditingController(text: '5.0');
  final _socController = TextEditingController(text: '0.5');
  final _timeController = TextEditingController(text: '12.0');
  final _priceController = TextEditingController(text: '0.20');

  @override
  void dispose() {
    _solarController.dispose();
    _socController.dispose();
    _timeController.dispose();
    _priceController.dispose();
    super.dispose();
  }

  void _runInference(BuildContext context) {
    final provider = Provider.of<InferenceProvider>(context, listen: false);
    
    final solar = double.tryParse(_solarController.text) ?? 0.0;
    final soc = double.tryParse(_socController.text) ?? 0.0;
    final time = double.tryParse(_timeController.text) ?? 0.0;
    final price = double.tryParse(_priceController.text) ?? 0.0;

    // Validate inputs
    if (solar < 0 || solar > 10) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Solar output must be 0-10 kW')),
      );
      return;
    }
    if (soc < 0 || soc > 1) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('EV SoC must be 0.0-1.0')),
      );
      return;
    }
    if (time < 0 || time > 24) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Time remaining must be 0-24 hours')),
      );
      return;
    }
    if (price < 0 || price > 0.5) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Grid price must be 0-0.5 \$/kWh')),
      );
      return;
    }

    provider.runInference(
      solarOutput: solar,
      evSoc: soc,
      timeRemaining: time,
      gridPrice: price,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Solar-Arbitrage EV Controller'),
        backgroundColor: Colors.green,
      ),
      body: Consumer<InferenceProvider>(
        builder: (context, provider, child) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Title
                const Text(
                  'Input Parameters',
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 24),

                // Solar Output
                TextField(
                  controller: _solarController,
                  decoration: const InputDecoration(
                    labelText: 'Solar Output (kW)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 16),

                // EV SoC
                TextField(
                  controller: _socController,
                  decoration: const InputDecoration(
                    labelText: 'EV State of Charge (0.0 - 1.0)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 16),

                // Time Remaining
                TextField(
                  controller: _timeController,
                  decoration: const InputDecoration(
                    labelText: 'Time Until Departure (hours)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 16),

                // Grid Price
                TextField(
                  controller: _priceController,
                  decoration: const InputDecoration(
                    labelText: 'Grid Price (\$/kWh)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 24),

                // Run Inference Button
                ElevatedButton(
                  onPressed: provider.isLoading
                      ? null
                      : () => _runInference(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  child: provider.isLoading
                      ? const CircularProgressIndicator(color: Colors.white)
                      : const Text(
                          'Run Inference',
                          style: TextStyle(fontSize: 18, color: Colors.white),
                        ),
                ),
                const SizedBox(height: 24),

                // Error Message
                if (provider.error != null)
                  Card(
                    color: Colors.red[50],
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Text(
                        provider.error!,
                        style: const TextStyle(color: Colors.red),
                      ),
                    ),
                  ),

                // Result Card
                if (provider.result != null)
                  Card(
                    elevation: 4,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Recommendation',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            provider.result!.actionName,
                            style: TextStyle(
                              fontSize: 18,
                              color: Colors.green[700],
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(provider.result!.recommendation),
                          const SizedBox(height: 16),
                          const Text(
                            'Confidence',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 8),
                          LinearProgressIndicator(
                            value: provider.result!.confidence,
                            backgroundColor: Colors.grey[300],
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.green,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '${(provider.result!.confidence * 100).toInt()}% confident',
                            style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'Probabilities:\n'
                            'Solar Only: ${(provider.result!.probabilities[0] * 100).toInt()}%\n'
                            'Solar + Grid: ${(provider.result!.probabilities[1] * 100).toInt()}%',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }
}
```

### Step 7: Run on Device

```bash
# Connect Android device
flutter devices

# Run app
flutter run

# Build APK
flutter build apk --release
```

## 🎨 UI Enhancements

### Add Charts

```yaml
# pubspec.yaml
dependencies:
  fl_chart: ^0.65.0
```

### Add Real-Time Updates

```dart
// Use StreamBuilder for real-time sensor data
StreamBuilder<double>(
  stream: solarSensorStream,
  builder: (context, snapshot) {
    if (snapshot.hasData) {
      _solarController.text = snapshot.data!.toString();
    }
    return TextField(...);
  },
)
```

## 📱 Summary

1. ✅ Create Flutter project
2. ✅ Add ONNX Runtime dependency
3. ✅ Copy model to assets
4. ✅ Create ONNX service
5. ✅ Create UI with Provider
6. ✅ Run on device

The Flutter app will automatically use NPU when available on Snapdragon devices!
