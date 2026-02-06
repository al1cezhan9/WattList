# Android Deployment Guide - Snapdragon 8 Elite NPU

## 📱 Overview

This guide shows how to deploy the Solar-Arbitrage EV Controller ONNX model to a Samsung Galaxy S25 (or any Snapdragon 8 Elite device) and create a UI for real-time inference.

## 🎯 Deployment Options

### Option 1: Android Native App (Recommended)
- **Language**: Kotlin/Java
- **Framework**: Android SDK
- **ONNX Runtime**: onnxruntime-android with QNN provider
- **UI**: Native Android UI (XML/Compose)

### Option 2: Flutter App
- **Language**: Dart
- **Framework**: Flutter
- **ONNX Runtime**: onnxruntime-dart plugin
- **UI**: Flutter widgets

### Option 3: React Native App
- **Language**: JavaScript/TypeScript
- **Framework**: React Native
- **ONNX Runtime**: Native module bridge
- **UI**: React Native components

## 🚀 Option 1: Android Native App (Detailed)

### Step 1: Prepare ONNX Model

```bash
# Ensure you have the trained ONNX model
ls solar_agent.onnx

# Verify model is correct
python -m src.inference_onnx solar_agent.onnx --single-test
```

### Step 2: Create Android Project

**Using Android Studio:**

1. Create new Android project (Kotlin)
2. Minimum SDK: API 24 (Android 7.0)
3. Target SDK: API 34+ (Android 14+)

### Step 3: Add Dependencies

**`app/build.gradle.kts`:**

```kotlin
dependencies {
    // ONNX Runtime for Android
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
    
    // QNN provider (for Snapdragon NPU)
    // Note: QNN provider may need to be built from source or obtained from Qualcomm
    // For now, use CPU provider which works on all devices
    
    // Coroutines for async operations
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // ViewModel and LiveData
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
    
    // Material Design
    implementation("com.google.android.material:material:1.11.0")
}
```

### Step 4: Add ONNX Model to Assets

**Project Structure:**
```
app/
  src/
    main/
      assets/
        solar_agent.onnx  ← Copy your ONNX model here
```

**Copy model:**
```bash
cp solar_agent.onnx app/src/main/assets/solar_agent.onnx
```

### Step 5: Create ONNX Inference Class

**`app/src/main/java/com/yourcompany/solararbitrage/OnnxInference.kt`:**

```kotlin
package com.yourcompany.solararbitrage

import android.content.Context
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class OnnxInference(context: Context) {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    
    init {
        initializeModel(context)
    }
    
    private fun initializeModel(context: Context) {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            
            // Load model from assets
            val modelBytes = context.assets.open("solar_agent.onnx").readBytes()
            
            // Create session options
            val sessionOptions = OrtSession.SessionOptions()
            
            // Try to use QNN provider if available (Snapdragon NPU)
            try {
                sessionOptions.addNnapi() // Use Android Neural Networks API
                // Note: For QNN specifically, you may need Qualcomm's SDK
            } catch (e: Exception) {
                // Fallback to CPU
                println("QNN/NNAPI not available, using CPU")
            }
            
            // Create session
            ortSession = ortEnv!!.createSession(modelBytes, sessionOptions)
            
            println("✅ ONNX model loaded successfully")
            println("Input shape: ${ortSession!!.inputNames[0]}")
            println("Output shape: ${ortSession!!.outputNames[0]}")
            
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Failed to load ONNX model", e)
        }
    }
    
    /**
     * Run inference on the ONNX model
     * 
     * @param solarOutput Solar output in kW (0-10)
     * @param evSoc EV state of charge (0.0-1.0)
     * @param timeRemaining Hours until departure (0-24)
     * @param gridPrice Grid price in $/kWh (0-0.5)
     * @return Pair of (action: Int, probabilities: FloatArray)
     *         action: 0 = Solar Only, 1 = Solar + Grid
     */
    suspend fun predict(
        solarOutput: Float,
        evSoc: Float,
        timeRemaining: Float,
        gridPrice: Float
    ): Pair<Int, FloatArray> = withContext(Dispatchers.Default) {
        try {
            // Prepare input: [solar_output, ev_soc, time_remaining, grid_price]
            val inputArray = floatArrayOf(solarOutput, evSoc, timeRemaining, gridPrice)
            
            // Create input tensor with shape (1, 4)
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                arrayOf(arrayOf(inputArray))
            )
            
            // Run inference
            val inputs = mapOf(ortSession!!.inputNames[0] to inputTensor)
            val outputs = ortSession!!.run(inputs)
            
            // Get output probabilities
            val outputTensor = outputs[0].value as Array<Array<FloatArray>>
            val probabilities = outputTensor[0][0] // Shape: (2,)
            
            // Select action with highest probability
            val action = if (probabilities[0] > probabilities[1]) 0 else 1
            
            // Clean up
            inputTensor.close()
            outputs.close()
            
            Pair(action, probabilities)
            
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Inference failed", e)
        }
    }
    
    fun close() {
        ortSession?.close()
        ortEnv?.close()
    }
}
```

### Step 6: Create ViewModel

**`app/src/main/java/com/yourcompany/solararbitrage/MainViewModel.kt`:**

```kotlin
package com.yourcompany.solararbitrage

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

data class InferenceResult(
    val action: Int,
    val actionName: String,
    val probabilities: FloatArray,
    val confidence: Float,
    val recommendation: String
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val onnxInference = OnnxInference(application)
    
    private val _inferenceResult = MutableLiveData<InferenceResult?>()
    val inferenceResult: LiveData<InferenceResult?> = _inferenceResult
    
    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading
    
    fun runInference(
        solarOutput: Float,
        evSoc: Float,
        timeRemaining: Float,
        gridPrice: Float
    ) {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val (action, probabilities) = onnxInference.predict(
                    solarOutput,
                    evSoc,
                    timeRemaining,
                    gridPrice
                )
                
                val actionName = if (action == 0) "Solar Only" else "Solar + Grid"
                val confidence = probabilities[action]
                val recommendation = when {
                    action == 0 -> "Charge using free solar energy only"
                    else -> "Charge at maximum power (solar + grid)"
                }
                
                _inferenceResult.value = InferenceResult(
                    action,
                    actionName,
                    probabilities,
                    confidence,
                    recommendation
                )
            } catch (e: Exception) {
                e.printStackTrace()
                _inferenceResult.value = null
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        onnxInference.close()
    }
}
```

### Step 7: Create UI Layout

**`app/src/main/res/layout/activity_main.xml`:**

```xml
<?xml version="1.0" encoding="utf-8"?>
<ScrollView xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical"
        android:gravity="center">

        <!-- Title -->
        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Solar-Arbitrage EV Controller"
            android:textSize="24sp"
            android:textStyle="bold"
            android:layout_marginBottom="32dp" />

        <!-- Solar Output Input -->
        <com.google.android.material.textfield.TextInputLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:hint="Solar Output (kW)"
            android:layout_marginBottom="16dp">
            <com.google.android.material.textfield.TextInputEditText
                android:id="@+id/editSolarOutput"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="numberDecimal"
                android:text="5.0" />
        </com.google.android.material.textfield.TextInputLayout>

        <!-- EV SoC Input -->
        <com.google.android.material.textfield.TextInputLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:hint="EV State of Charge (0.0 - 1.0)"
            android:layout_marginBottom="16dp">
            <com.google.android.material.textfield.TextInputEditText
                android:id="@+id/editEvSoc"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="numberDecimal"
                android:text="0.5" />
        </com.google.android.material.textfield.TextInputLayout>

        <!-- Time Remaining Input -->
        <com.google.android.material.textfield.TextInputLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:hint="Time Until Departure (hours)"
            android:layout_marginBottom="16dp">
            <com.google.android.material.textfield.TextInputEditText
                android:id="@+id/editTimeRemaining"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="numberDecimal"
                android:text="12.0" />
        </com.google.android.material.textfield.TextInputLayout>

        <!-- Grid Price Input -->
        <com.google.android.material.textfield.TextInputLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:hint="Grid Price ($/kWh)"
            android:layout_marginBottom="24dp">
            <com.google.android.material.textfield.TextInputEditText
                android:id="@+id/editGridPrice"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="numberDecimal"
                android:text="0.20" />
        </com.google.android.material.textfield.TextInputLayout>

        <!-- Run Inference Button -->
        <com.google.android.material.button.MaterialButton
            android:id="@+id/buttonRunInference"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Run Inference"
            android:textSize="18sp"
            android:layout_marginBottom="24dp" />

        <!-- Loading Indicator -->
        <ProgressBar
            android:id="@+id/progressBar"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:visibility="gone"
            android:layout_marginBottom="16dp" />

        <!-- Result Card -->
        <com.google.android.material.card.MaterialCardView
            android:id="@+id/cardResult"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:visibility="gone"
            app:cardElevation="4dp"
            android:layout_marginBottom="16dp">

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="vertical"
                android:padding="16dp">

                <TextView
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Recommendation"
                    android:textSize="20sp"
                    android:textStyle="bold"
                    android:layout_marginBottom="8dp" />

                <TextView
                    android:id="@+id/textAction"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:textSize="18sp"
                    android:textColor="@android:color/holo_green_dark"
                    android:layout_marginBottom="8dp" />

                <TextView
                    android:id="@+id/textRecommendation"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:textSize="16sp"
                    android:layout_marginBottom="16dp" />

                <TextView
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Confidence"
                    android:textSize="16sp"
                    android:textStyle="bold"
                    android:layout_marginBottom="8dp" />

                <ProgressBar
                    android:id="@+id/progressConfidence"
                    style="?android:attr/progressBarStyleHorizontal"
                    android:layout_width="match_parent"
                    android:layout_height="wrap_content"
                    android:max="100"
                    android:layout_marginBottom="8dp" />

                <TextView
                    android:id="@+id/textConfidence"
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:textSize="14sp" />

            </LinearLayout>
        </com.google.android.material.card.MaterialCardView>

    </LinearLayout>
</ScrollView>
```

### Step 8: Create Main Activity

**`app/src/main/java/com/yourcompany/solararbitrage/MainActivity.kt`:**

```kotlin
package com.yourcompany.solararbitrage

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {
    private lateinit var viewModel: MainViewModel
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize ViewModel
        viewModel = ViewModelProvider(this)[MainViewModel::class.java]
        
        // Observe inference results
        viewModel.inferenceResult.observe(this) { result ->
            result?.let { displayResult(it) }
        }
        
        // Observe loading state
        viewModel.isLoading.observe(this) { isLoading ->
            progressBar.visibility = if (isLoading) android.view.View.VISIBLE else android.view.View.GONE
            buttonRunInference.isEnabled = !isLoading
        }
        
        // Set up button click listener
        buttonRunInference.setOnClickListener {
            runInference()
        }
    }
    
    private fun runInference() {
        try {
            val solarOutput = editSolarOutput.text.toString().toFloatOrNull() ?: 0f
            val evSoc = editEvSoc.text.toString().toFloatOrNull() ?: 0f
            val timeRemaining = editTimeRemaining.text.toString().toFloatOrNull() ?: 0f
            val gridPrice = editGridPrice.text.toString().toFloatOrNull() ?: 0f
            
            // Validate inputs
            if (solarOutput < 0 || solarOutput > 10) {
                Toast.makeText(this, "Solar output must be 0-10 kW", Toast.LENGTH_SHORT).show()
                return
            }
            if (evSoc < 0 || evSoc > 1) {
                Toast.makeText(this, "EV SoC must be 0.0-1.0", Toast.LENGTH_SHORT).show()
                return
            }
            if (timeRemaining < 0 || timeRemaining > 24) {
                Toast.makeText(this, "Time remaining must be 0-24 hours", Toast.LENGTH_SHORT).show()
                return
            }
            if (gridPrice < 0 || gridPrice > 0.5) {
                Toast.makeText(this, "Grid price must be 0-0.5 $/kWh", Toast.LENGTH_SHORT).show()
                return
            }
            
            viewModel.runInference(solarOutput, evSoc, timeRemaining, gridPrice)
            
        } catch (e: Exception) {
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun displayResult(result: InferenceResult) {
        cardResult.visibility = android.view.View.VISIBLE
        
        textAction.text = result.actionName
        textRecommendation.text = result.recommendation
        
        val confidencePercent = (result.confidence * 100).toInt()
        progressConfidence.progress = confidencePercent
        textConfidence.text = "$confidencePercent% confident"
        
        // Show probabilities
        val probText = "Solar Only: ${(result.probabilities[0] * 100).toInt()}%\n" +
                       "Solar + Grid: ${(result.probabilities[1] * 100).toInt()}%"
        Toast.makeText(this, probText, Toast.LENGTH_LONG).show()
    }
}
```

### Step 9: Add Permissions (if needed)

**`app/src/main/AndroidManifest.xml`:**

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <!-- Add other permissions as needed -->
    
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.AppCompat.Light.DarkActionBar">
        
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

## 🔧 QNN Provider Setup (Snapdragon NPU)

### Option A: Use Android Neural Networks API (NNAPI)

The code above uses `sessionOptions.addNnapi()` which will use the device's NPU if available.

### Option B: Qualcomm QNN SDK (Advanced)

For direct QNN provider access, you need:

1. **Qualcomm AI Engine Direct SDK**
   - Download from Qualcomm Developer Portal
   - Requires Qualcomm developer account

2. **Build ONNX Runtime with QNN**
   - Build onnxruntime from source with QNN support
   - Include QNN libraries in your Android app

3. **Use QNN Provider**
```kotlin
// In OnnxInference.kt
sessionOptions.addSessionConfigEntry("session.use_qnn", "1")
```

## 📱 Testing on Device

### Build and Install

```bash
# Build APK
./gradlew assembleDebug

# Install on connected device
adb install app/build/outputs/apk/debug/app-debug.apk

# Or use Android Studio: Run → Run 'app'
```

### Verify NPU Usage

```kotlin
// Add logging to check which provider is used
val providers = ortSession?.providers
Log.d("ONNX", "Available providers: $providers")
```

## 🎨 UI Enhancements

### Add Real-Time Updates

```kotlin
// Update inputs from sensors
fun updateFromSensors(solar: Float, soc: Float, price: Float) {
    editSolarOutput.setText(solar.toString())
    editEvSoc.setText(soc.toString())
    editGridPrice.setText(price.toString())
    // Auto-run inference
    runInference()
}
```

### Add Charts

Use **MPAndroidChart** library to visualize:
- Solar output over time
- SoC progress
- Action history

## 📊 Performance Optimization

### 1. Model Quantization

```python
# Quantize model to INT8 for faster inference
# (Do this before deploying)
```

### 2. Batch Processing

```kotlin
// Process multiple predictions at once
val batchInput = Array(batchSize) { inputArray }
```

### 3. Caching

```kotlin
// Cache recent predictions
private val predictionCache = LRUCache<String, InferenceResult>(10)
```

## 🚀 Alternative: Flutter App

If you prefer Flutter, see `DEPLOYMENT_FLUTTER.md` for Flutter-specific instructions.

## 📝 Summary

1. ✅ Copy `solar_agent.onnx` to `app/src/main/assets/`
2. ✅ Add ONNX Runtime dependency
3. ✅ Create `OnnxInference` class
4. ✅ Create UI with input fields
5. ✅ Connect ViewModel to Activity
6. ✅ Build and deploy to device

The app will run inference on the Snapdragon NPU automatically when available!
