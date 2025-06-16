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