from flask import Flask, send_from_directory, request, jsonify
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import uuid
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

# ML Model Configuration
MODEL_PATH = "saved_models/complex_cnn_model.pth"
INPUT_SIZE = 1000
HIDDEN_SIZE = 128
WEBSITES = [
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com", 
    "https://prothomalo.com"
]

# Global variables for ML model
ml_model = None
scaler = None

# Global statistics calculated from the training dataset
# These are the exact mean and std values used during training
GLOBAL_MEAN = None
GLOBAL_STD = None

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def load_ml_model():
    """Load the trained ML model for real-time prediction."""
    global ml_model, scaler, GLOBAL_MEAN, GLOBAL_STD
    
    try:
        # Initialize model
        num_classes = len(WEBSITES)
        ml_model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
        
        # Load trained weights
        ml_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        ml_model.eval()
        
        # Load global statistics for proper normalization
        try:
            with open('global_stats.json', 'r') as f:
                stats = json.load(f)
                GLOBAL_MEAN = np.array(stats['mean'])
                GLOBAL_STD = np.array(stats['scale'])
            print("✅ Global normalization statistics loaded!")
        except Exception as e:
            print(f"⚠️ Could not load global stats: {e}")
            print("   Prediction accuracy may be reduced")
        
        print("✅ ML model loaded successfully!")
        print(f"Model can classify: {WEBSITES}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        return False

def preprocess_trace_for_prediction(trace, target_length=1000):
    """Preprocess a single trace for ML prediction to match training preprocessing."""
    global GLOBAL_MEAN, GLOBAL_STD
    
    try:
        trace = np.array(trace, dtype=np.float32)
        
        print(f"🔍 Debug - Input trace stats: len={len(trace)}, mean={np.mean(trace):.2f}, std={np.std(trace):.2f}, min={np.min(trace)}, max={np.max(trace)}")
        
        # Truncate or pad to target length (same as training)
        if len(trace) > target_length:
            trace = trace[:target_length]
        elif len(trace) < target_length:
            trace = np.pad(trace, (0, target_length - len(trace)), mode='constant')
        
        # Apply the EXACT same normalization as training using global statistics
        if GLOBAL_MEAN is not None and GLOBAL_STD is not None:
            # StandardScaler formula: (x - mean) / std
            trace = (trace - GLOBAL_MEAN) / GLOBAL_STD
            print(f"✅ Applied global normalization - new stats: mean={np.mean(trace):.2f}, std={np.std(trace):.2f}")
        else:
            print("⚠️  WARNING: No global stats available, using trace without normalization")
        
        processed_trace = trace.flatten()
        print(f"🔍 Debug - Final processed trace stats: mean={np.mean(processed_trace):.2f}, std={np.std(processed_trace):.2f}")
        
        return processed_trace
        
    except Exception as e:
        print(f"Error preprocessing trace: {e}")
        return None

def predict_website(trace):
    """Predict which website a trace belongs to."""
    global ml_model
    
    if ml_model is None:
        return None, None
    
    try:
        # Preprocess trace
        processed_trace = preprocess_trace_for_prediction(trace)
        if processed_trace is None:
            return None, None
        
        # Convert to tensor
        trace_tensor = torch.FloatTensor(processed_trace).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = ml_model(trace_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Debug output
        print(f"🔍 Debug - Raw model outputs: {outputs[0].cpu().numpy()}")
        print(f"🔍 Debug - Softmax probabilities: {probabilities[0].cpu().numpy()}")
        print(f"🔍 Debug - Predicted class: {predicted_class}")
        print(f"🔍 Debug - Website mapping: {dict(enumerate(WEBSITES))}")
        
        predicted_website = WEBSITES[predicted_class]
        
        # Get all probabilities for detailed results
        all_probs = {
            WEBSITES[i]: float(probabilities[0][i])
            for i in range(len(WEBSITES))
        }
        
        print(f"🔍 Debug - All probabilities: {all_probs}")
        print(f"🔍 Debug - Final prediction: {predicted_website} with confidence {confidence:.3f}")
        
        return predicted_website, confidence, all_probs
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

# Load the ML model when the app starts
if not load_ml_model():
    print("⚠️  App will run without ML prediction capabilities")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/results/<path:path>')
def serve_results(path):
    return send_from_directory('results', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in the backend temporarily
    4. Return the heatmap image and optionally other statistics to the frontend
    """
    try:
        data = request.get_json()
        trace_data = data['trace']
        timestamp = data.get('timestamp', int(datetime.now().timestamp() * 1000))
        
        # Generate heatmap
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}_{timestamp}.png"
        heatmap_path = os.path.join('results', heatmap_filename)
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 8))
        
        # Reshape data for better visualization
        data_array = np.array(trace_data)
        
        # Create 2D representation for heatmap
        # Reshape into a grid for better visualization
        rows = int(np.sqrt(len(data_array))) + 1
        cols = int(np.ceil(len(data_array) / rows))
        
        # Pad with zeros if necessary
        padded_data = np.pad(data_array, (0, rows * cols - len(data_array)), mode='constant')
        heatmap_data = padded_data.reshape((rows, cols))
        
        # Create heatmap
        im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Sweep Count')
        plt.title(f'Cache Trace Heatmap - {datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d %H:%M:%S")}')
        plt.xlabel('Time Window')
        plt.ylabel('Cache Access Pattern')
        
        # Save the heatmap
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Store trace data and heatmap info
        trace_info = {
            'trace': trace_data,
            'timestamp': timestamp,
            'heatmap_path': f'/results/{heatmap_filename}',
            'data_points': len(trace_data)
        }
        
        stored_traces.append(trace_info)
        stored_heatmaps.append({
            'path': f'/results/{heatmap_filename}',
            'timestamp': timestamp,
            'dataPoints': len(trace_data)
        })
        
        return jsonify({
            'success': True,
            'heatmap_path': f'/results/{heatmap_filename}',
            'data_points': len(trace_data),
            'timestamp': timestamp
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implment a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        global stored_traces, stored_heatmaps
        
        # Remove old heatmap files
        for heatmap in stored_heatmaps:
            filename = heatmap['path'].replace('/results/', '')
            filepath = os.path.join('results', filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Clear stored data
        stored_traces = []
        stored_heatmaps = []
        
        return jsonify({'success': True, 'message': 'All results cleared'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download_traces', methods=['GET'])
def download_traces():
    """
    Return all collected traces in JSON format for ML training
    """
    try:
        # Format data for ML training
        formatted_data = {
            'traces': [trace['trace'] for trace in stored_traces],
            'timestamps': [trace['timestamp'] for trace in stored_traces],
            'metadata': {
                'total_traces': len(stored_traces),
                'collection_time': datetime.now().isoformat(),
                'data_points_per_trace': [len(trace['trace']) for trace in stored_traces]
            }
        }
        
        return jsonify(formatted_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/collect_advanced_trace', methods=['POST'])
def collect_advanced_trace():
    """
    Handle advanced trace collection with multiple visualization techniques
    """
    try:
        data = request.get_json()
        trace_data = data['trace']
        technique = data.get('technique', 'unknown')
        metadata = data.get('metadata', {})
        timestamp = data.get('timestamp', int(datetime.now().timestamp() * 1000))
        
        # Generate advanced heatmap
        heatmap_filename = f"advanced_heatmap_{technique}_{uuid.uuid4().hex}_{timestamp}.png"
        heatmap_path = os.path.join('results', heatmap_filename)
        
        # Create advanced visualization based on technique
        plt.figure(figsize=(15, 10))
        
        if technique == 'combined':
            # Multi-panel visualization for combined techniques
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract different components
            if isinstance(trace_data, list) and len(trace_data) > 0:
                sample_data = trace_data[0] if isinstance(trace_data[0], dict) else {}
                
                # Panel 1: Multi-resolution data
                if 'multi' in sample_data:
                    multi_data = [item.get('multi', {}) for item in trace_data if isinstance(item, dict)]
                    if multi_data:
                        counts = [item.get('count', 0) for item in multi_data]
                        axes[0, 0].plot(counts, 'b-', linewidth=2)
                        axes[0, 0].set_title('Multi-Resolution Sweep Counts')
                        axes[0, 0].set_xlabel('Time Window')
                        axes[0, 0].set_ylabel('Sweep Count')
                        axes[0, 0].grid(True, alpha=0.3)
                
                # Panel 2: Variance analysis
                if 'variance' in sample_data:
                    var_data = [item.get('variance', {}) for item in trace_data if isinstance(item, dict)]
                    if var_data:
                        variances = [item.get('variance', 0) for item in var_data]
                        axes[0, 1].plot(variances, 'r-', linewidth=2)
                        axes[0, 1].set_title('Timing Variance Analysis')
                        axes[0, 1].set_xlabel('Time Window')
                        axes[0, 1].set_ylabel('Variance')
                        axes[0, 1].grid(True, alpha=0.3)
                
                # Panel 3: Frequency domain
                if 'frequency' in sample_data:
                    freq_data = [item.get('frequency', []) for item in trace_data if isinstance(item, dict)]
                    if freq_data and freq_data[0]:
                        freq_matrix = np.array([f[:min(len(f), 32)] for f in freq_data if f])
                        if freq_matrix.size > 0:
                            im3 = axes[1, 0].imshow(freq_matrix.T, cmap='plasma', aspect='auto')
                            axes[1, 0].set_title('Frequency Domain Analysis')
                            axes[1, 0].set_xlabel('Time Window')
                            axes[1, 0].set_ylabel('Frequency Bin')
                            plt.colorbar(im3, ax=axes[1, 0])
                
                # Panel 4: Combined heatmap
                try:
                    # Create a combined feature matrix
                    combined_features = []
                    for item in trace_data[:100]:  # Limit for visualization
                        if isinstance(item, dict):
                            features = []
                            if 'multi' in item:
                                features.append(item['multi'].get('count', 0))
                            if 'variance' in item:
                                features.append(item['variance'].get('variance', 0))
                            if 'frequency' in item and item['frequency']:
                                features.extend(item['frequency'][:5])  # First 5 freq bins
                            combined_features.append(features)
                    
                    if combined_features:
                        # Pad features to same length
                        max_len = max(len(f) for f in combined_features)
                        padded_features = [f + [0] * (max_len - len(f)) for f in combined_features]
                        feature_matrix = np.array(padded_features)
                        
                        im4 = axes[1, 1].imshow(feature_matrix.T, cmap='viridis', aspect='auto')
                        axes[1, 1].set_title('Combined Feature Heatmap')
                        axes[1, 1].set_xlabel('Time Window')
                        axes[1, 1].set_ylabel('Feature Index')
                        plt.colorbar(im4, ax=axes[1, 1])
                except:
                    axes[1, 1].text(0.5, 0.5, 'Feature visualization\nnot available', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.suptitle(f'Advanced Side Channel Analysis - {datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d %H:%M:%S")}')
            plt.tight_layout()
        
        else:
            # Single technique visualization
            if isinstance(trace_data, list) and trace_data:
                data_array = np.array(trace_data)
                
                # Create 2D representation
                if data_array.ndim == 1:
                    rows = int(np.sqrt(len(data_array))) + 1
                    cols = int(np.ceil(len(data_array) / rows))
                    padded_data = np.pad(data_array, (0, rows * cols - len(data_array)), mode='constant')
                    heatmap_data = padded_data.reshape((rows, cols))
                else:
                    heatmap_data = data_array
                
                im = plt.imshow(heatmap_data, cmap='plasma', aspect='auto')
                plt.colorbar(im, label='Measurement Value')
                plt.title(f'Advanced {technique.title()} Trace - {datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d %H:%M:%S")}')
                plt.xlabel('Feature/Time Index')
                plt.ylabel('Measurement Index')
        
        # Save the heatmap
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store advanced trace data
        trace_info = {
            'trace': trace_data,
            'technique': technique,
            'metadata': metadata,
            'timestamp': timestamp,
            'heatmap_path': f'/results/{heatmap_filename}',
            'data_points': len(trace_data) if isinstance(trace_data, list) else "Complex"
        }
        
        stored_traces.append(trace_info)
        stored_heatmaps.append({
            'path': f'/results/{heatmap_filename}',
            'timestamp': timestamp,
            'dataPoints': len(trace_data) if isinstance(trace_data, list) else "Advanced",
            'technique': technique
        })
        
        return jsonify({
            'success': True,
            'heatmap_path': f'/results/{heatmap_filename}',
            'technique': technique,
            'data_points': len(trace_data) if isinstance(trace_data, list) else "Complex",
            'timestamp': timestamp
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_results', methods=['GET'])
def get_results():
    """
    Get existing results for frontend initialization
    """
    try:
        return jsonify({
            'heatmaps': stored_heatmaps,
            'traces': stored_traces
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict_website', methods=['POST'])
def predict_website_endpoint():
    """
    Real-time website prediction endpoint.
    Takes trace data and returns prediction with confidence scores.
    """
    try:
        if ml_model is None:
            return jsonify({
                'success': False,
                'error': 'ML model not loaded. Please check model file exists.'
            }), 500
        
        data = request.get_json()
        trace_data = data.get('trace', [])
        
        if not trace_data:
            return jsonify({
                'success': False,
                'error': 'No trace data provided'
            }), 400
        
        # Make prediction
        predicted_website, confidence, all_probabilities = predict_website(trace_data)
        
        if predicted_website is None:
            return jsonify({
                'success': False,
                'error': 'Failed to make prediction'
            }), 500
        
        return jsonify({
            'success': True,
            'prediction': {
                'website': predicted_website,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            },
            'timestamp': int(datetime.now().timestamp() * 1000),
            'trace_length': len(trace_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model_status', methods=['GET'])
def model_status():
    """
    Get the status of the ML model and available websites.
    """
    try:
        return jsonify({
            'model_loaded': ml_model is not None,
            'model_path': MODEL_PATH,
            'available_websites': WEBSITES,
            'input_size': INPUT_SIZE,
            'model_type': 'Complex CNN' if ml_model is not None else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/collect_trace_with_prediction', methods=['POST'])
def collect_trace_with_prediction():
    """
    Enhanced trace collection that also provides real-time prediction.
    Combines the heatmap generation with ML prediction.
    """
    try:
        data = request.get_json()
        trace_data = data['trace']
        timestamp = data.get('timestamp', int(datetime.now().timestamp() * 1000))
        
        # Generate heatmap (existing functionality)
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}_{timestamp}.png"
        heatmap_path = os.path.join('results', heatmap_filename)
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 8))
        
        # Reshape data for better visualization
        data_array = np.array(trace_data)
        
        # Create 2D representation for heatmap
        rows = int(np.sqrt(len(data_array))) + 1
        cols = int(np.ceil(len(data_array) / rows))
        
        # Pad with zeros if necessary
        padded_data = np.pad(data_array, (0, rows * cols - len(data_array)), mode='constant')
        heatmap_data = padded_data.reshape((rows, cols))
        
        # Create heatmap
        im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Sweep Count')
        plt.title(f'Cache Trace Heatmap - {datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d %H:%M:%S")}')
        plt.xlabel('Time Window')
        plt.ylabel('Cache Access Pattern')
        
        # Save the heatmap
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Make ML prediction if model is available
        prediction_result = None
        if ml_model is not None:
            predicted_website, confidence, all_probabilities = predict_website(trace_data)
            if predicted_website is not None:
                prediction_result = {
                    'website': predicted_website,
                    'confidence': confidence,
                    'all_probabilities': all_probabilities
                }
        
        # Store trace data and heatmap info
        trace_info = {
            'trace': trace_data,
            'timestamp': timestamp,
            'heatmap_path': f'/results/{heatmap_filename}',
            'data_points': len(trace_data),
            'prediction': prediction_result
        }
        
        stored_traces.append(trace_info)
        stored_heatmaps.append({
            'path': f'/results/{heatmap_filename}',
            'timestamp': timestamp,
            'dataPoints': len(trace_data),
            'prediction': prediction_result
        })
        
        response = {
            'success': True,
            'heatmap_path': f'/results/{heatmap_filename}',
            'data_points': len(trace_data),
            'timestamp': timestamp
        }
        
        if prediction_result:
            response['prediction'] = prediction_result
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Additional endpoints can be implemented here as needed.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)