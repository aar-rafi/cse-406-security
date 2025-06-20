#!/usr/bin/env python3
"""
Quick sanity check to test if the trained model works on known training data
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Model architecture (copy from app.py)
class ComplexFingerprintClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        conv_output_size = input_size // 8
        self.fc_input_size = conv_output_size * 128
        
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(-1, self.fc_input_size)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def test_model():
    print("=== Model Sanity Check ===")
    
    # Load dataset
    with open('dataset.json', 'r') as f:
        data = json.load(f)
    
    # Prepare websites and sample data
    websites = list(data['websites'].keys())
    print(f"Websites: {websites}")
    
    # Load global stats
    with open('global_stats.json', 'r') as f:
        stats = json.load(f)
        global_mean = np.array(stats['mean'])
        global_std = np.array(stats['scale'])
    
    # Load model
    model = ComplexFingerprintClassifier(1000, 128, len(websites))
    model.load_state_dict(torch.load('saved_models/complex_cnn_model.pth', map_location='cpu'))
    model.eval()
    
    print("Model loaded successfully!")
    
    # Test on a few samples from each website
    for website_idx, (website, traces) in enumerate(data['websites'].items()):
        print(f"\n--- Testing {website} (expected class {website_idx}) ---")
        
        # Test on first 3 traces from this website
        for i in range(min(3, len(traces))):
            trace = np.array(traces[i]['trace'], dtype=np.float32)
            
            # Preprocess exactly like training
            if len(trace) > 1000:
                trace = trace[:1000]
            elif len(trace) < 1000:
                trace = np.pad(trace, (0, 1000 - len(trace)), mode='constant')
            
            # Apply global normalization
            trace = (trace - global_mean) / global_std
            
            # Convert to tensor and predict
            trace_tensor = torch.FloatTensor(trace).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(trace_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_website = websites[predicted_class]
            correct = predicted_class == website_idx
            
            print(f"  Sample {i+1}: Predicted={predicted_website} (class {predicted_class}), "
                  f"Confidence={confidence:.3f}, Correct={correct}")
            
            if not correct:
                print(f"    Raw outputs: {outputs[0].cpu().numpy()}")
                print(f"    Probabilities: {probabilities[0].cpu().numpy()}")

if __name__ == "__main__":
    test_model() 