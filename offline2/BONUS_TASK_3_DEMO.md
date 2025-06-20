# Bonus Task 3: Real-Time Website Detection Demo

## 🎯 Overview

This bonus task implements **real-time website detection** by integrating the trained machine learning model directly into the Flask application. Users can now see predictions about which website they're visiting in an adjacent tab in real-time!

## ✅ What's Been Implemented

### Backend Features
1. **ML Model Integration**: The Complex CNN model (88.33% accuracy) is loaded when the Flask app starts
2. **Real-time Prediction API**: New endpoints for instant website classification
3. **Enhanced Trace Collection**: Combined trace collection + prediction in one action
4. **Model Status API**: Check if the ML model is loaded and what websites it can detect

### Frontend Features  
1. **ML Model Status Display**: Real-time indicator showing if the model is loaded
2. **Website Prediction UI**: Visual confidence scores and probability breakdown
3. **Enhanced Trace Collection**: One-click trace collection with instant prediction
4. **Prediction History**: All collected traces show their prediction results

## 🚀 How to Use

### Step 1: Start the Application
```bash
python app.py
```

The app will automatically try to load the trained model from `saved_models/complex_cnn_model.pth`.

### Step 2: Check Model Status
- Open http://localhost:5000 in your browser
- Look for the "🧠 Real-Time Website Detection" section
- You should see a green indicator showing "ML Model Loaded (Complex CNN)"
- The available websites for detection are listed:
  - https://cse.buet.ac.bd/moodle/
  - https://google.com
  - https://prothomalo.com

### Step 3: Test Real-Time Detection

#### Method 1: Collect Trace & Predict (Recommended)
1. Open one of the target websites in a new tab (e.g., google.com)
2. Click "🔍 Collect Trace & Predict" 
3. Wait ~10 seconds for data collection
4. See the real-time prediction with confidence score!

#### Method 2: Predict from Existing Trace
1. First collect a trace using the regular "Collect Trace" button
2. Then click "🎯 Predict Latest Trace" for instant prediction

### Step 4: View Results
- **Prediction Results**: Shows detected website, confidence percentage, and all probability scores
- **Confidence Bar**: Visual indicator of prediction confidence
- **Probability Breakdown**: See how confident the model is for each website
- **Heatmap Gallery**: All collected traces now show their prediction results

## 🎨 Features Demonstration

### Real-Time Prediction Display
- **Detected Website**: The predicted website URL
- **Confidence**: Percentage confidence (e.g., 89.3%)
- **Visual Confidence Bar**: Color-coded bar showing confidence level
- **All Probabilities**: Breakdown showing probability for each website

### Model Status Indicator
- **Green Dot**: Model loaded and ready
- **Red Dot**: Model not available
- **Model Type**: Shows which model is loaded (Complex CNN)

### Enhanced Heatmaps
- Each heatmap now shows prediction results
- Timestamp of when prediction was made
- Confidence score displayed with each trace

## 🧠 Technical Details

### Model Loading
- Automatically loads the best performing model (Complex CNN with 88.33% accuracy)
- Uses CPU inference for compatibility
- Fallback gracefully if model file is missing

### Preprocessing Pipeline
- Traces are automatically normalized to 1000 data points
- Same preprocessing used during training for consistency
- Real-time normalization for instant predictions

### API Endpoints
- `GET /api/model_status`: Check model availability and supported websites
- `POST /api/predict_website`: Get prediction for any trace data
- `POST /collect_trace_with_prediction`: Combined collection + prediction

## 🎯 Real-World Attack Scenario

This demonstrates a realistic side-channel attack:

1. **Attacker Setup**: Runs this Flask app on the victim's machine
2. **Victim Activity**: Opens websites in different browser tabs
3. **Real-Time Monitoring**: The attacker gets instant notifications about which websites the victim is visiting
4. **High Accuracy**: With 88.33% accuracy, the attacker has reliable intelligence

This is exactly the type of attack that makes browser security researchers worried about side-channel vulnerabilities!

## 🔧 Troubleshooting

### Model Not Loading
- Ensure `saved_models/complex_cnn_model.pth` exists
- Check that PyTorch is installed: `pip install torch`
- Look for model loading messages in the terminal

### Low Prediction Accuracy
- Ensure you're visiting one of the three trained websites
- Close other resource-intensive applications
- Try collecting longer traces for better signal

### Browser Compatibility
- Works best in Chrome/Chromium browsers
- Disable browser sandboxing if needed: `--no-sandbox` flag
- Ensure JavaScript workers are enabled

## 🏆 Achievement Unlocked!

You've successfully created a **real-time website fingerprinting attack** that can:
- ✅ Detect websites in real-time with 88.33% accuracy
- ✅ Provide instant feedback to users
- ✅ Show detailed confidence metrics
- ✅ Work entirely within a web browser
- ✅ Demonstrate a real security vulnerability

This is a powerful demonstration of how side-channel attacks can be weaponized for real-time surveillance!

## 🎊 Bonus Task 3 Complete!

Your implementation successfully "deploys" the trained model in the Flask app, providing users with real-time website detection capabilities. This creates an impressive demo of your side-channel attack that shows both the technical sophistication and the real-world security implications of this type of vulnerability. 