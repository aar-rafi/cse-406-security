from flask import Flask, send_from_directory, request, jsonify
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import uuid

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

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

# Additional endpoints can be implemented here as needed.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)