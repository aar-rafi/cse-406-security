import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import train  # Import base training module

# Advanced configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "advanced_models"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
INPUT_SIZE = 1000
HIDDEN_SIZE = 256

os.makedirs(MODELS_DIR, exist_ok=True)

class AdvancedPreprocessor:
    """Advanced preprocessing with multiple feature extraction techniques"""
    
    def __init__(self, target_length=1000):
        self.target_length = target_length
        self.scaler = RobustScaler()  # More robust to outliers
        
    def extract_features(self, traces):
        """Extract multiple types of features from traces"""
        processed_traces = []
        
        for trace in traces:
            trace = np.array(trace, dtype=np.float32)
            
            # 1. Basic preprocessing
            trace = self.normalize_length(trace)
            
            # 2. Statistical features
            stats = self.extract_statistical_features(trace)
            
            # 3. Frequency domain features
            freq_features = self.extract_frequency_features(trace)
            
            # 4. Temporal features
            temporal_features = self.extract_temporal_features(trace)
            
            # 5. Wavelet features
            wavelet_features = self.extract_wavelet_features(trace)
            
            # Combine all features
            combined_features = np.concatenate([
                trace,  # Original signal
                stats,  # Statistical features
                freq_features,  # Frequency features
                temporal_features,  # Temporal features
                wavelet_features  # Wavelet features
            ])
            
            processed_traces.append(combined_features)
        
        # Normalize all features
        processed_traces = np.array(processed_traces)
        processed_traces = self.scaler.fit_transform(processed_traces)
        
        return processed_traces.tolist()
    
    def normalize_length(self, trace):
        """Normalize trace length with interpolation"""
        if len(trace) > self.target_length:
            # Downsample using decimation
            factor = len(trace) // self.target_length
            trace = signal.decimate(trace, factor, ftype='fir')[:self.target_length]
        elif len(trace) < self.target_length:
            # Upsample using interpolation
            trace = signal.resample(trace, self.target_length)
        return trace
    
    def extract_statistical_features(self, trace):
        """Extract statistical features"""
        return np.array([
            np.mean(trace),
            np.std(trace),
            np.var(trace),
            np.min(trace),
            np.max(trace),
            np.median(trace),
            np.percentile(trace, 25),
            np.percentile(trace, 75),
            len(trace[trace > np.mean(trace)]) / len(trace),  # Above mean ratio
            np.sum(np.diff(trace) > 0) / len(trace),  # Increasing trend ratio
        ])
    
    def extract_frequency_features(self, trace):
        """Extract frequency domain features"""
        # FFT
        fft_vals = np.abs(fft(trace))[:len(trace)//2]
        freqs = fftfreq(len(trace))[:len(trace)//2]
        
        # Power spectral density
        psd = np.abs(fft_vals) ** 2
        
        # Frequency features
        dominant_freq = freqs[np.argmax(psd)]
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        # Energy in different frequency bands
        low_freq_energy = np.sum(psd[:len(psd)//4])
        mid_freq_energy = np.sum(psd[len(psd)//4:3*len(psd)//4])
        high_freq_energy = np.sum(psd[3*len(psd)//4:])
        
        return np.array([
            dominant_freq,
            spectral_centroid,
            spectral_bandwidth,
            low_freq_energy,
            mid_freq_energy,
            high_freq_energy,
            np.sum(psd),  # Total energy
        ])
    
    def extract_temporal_features(self, trace):
        """Extract temporal pattern features"""
        # Differences and derivatives
        diff1 = np.diff(trace)
        diff2 = np.diff(diff1)
        
        # Autocorrelation features
        autocorr = np.correlate(trace, trace, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        return np.array([
            np.mean(diff1),
            np.std(diff1),
            np.mean(diff2),
            np.std(diff2),
            autocorr[1] if len(autocorr) > 1 else 0,  # Lag-1 autocorr
            autocorr[10] if len(autocorr) > 10 else 0,  # Lag-10 autocorr
            np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0,  # Peak lag
        ])
    
    def extract_wavelet_features(self, trace):
        """Extract wavelet-based features (simplified)"""
        # Simple wavelet-like decomposition using moving averages
        window_sizes = [4, 8, 16, 32]
        features = []
        
        for window in window_sizes:
            if len(trace) >= window:
                smoothed = np.convolve(trace, np.ones(window)/window, mode='valid')
                features.extend([
                    np.mean(smoothed),
                    np.std(smoothed),
                    np.max(smoothed) - np.min(smoothed)
                ])
            else:
                features.extend([0, 0, 0])
        
        return np.array(features)

class AttentionLayer(nn.Module):
    """Self-attention mechanism for sequence data"""
    
    def __init__(self, input_dim, attention_dim=64):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(input_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attention_weights = torch.tanh(self.W(x))  # (batch_size, seq_len, attention_dim)
        attention_weights = self.V(attention_weights).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights
        attended = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, input_dim)
        return attended, attention_weights

class TransformerBlock(nn.Module):
    """Simplified transformer block"""
    
    def __init__(self, embed_dim, num_heads=8, ff_dim=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class AdvancedCNN(nn.Module):
    """Advanced CNN with attention and residual connections"""
    
    def __init__(self, input_size, num_classes):
        super(AdvancedCNN, self).__init__()
        
        # Multi-scale convolutional layers
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(1, 32, kernel_size=3),
            self._make_conv_block(32, 64, kernel_size=5),
            self._make_conv_block(64, 128, kernel_size=7),
            self._make_conv_block(128, 256, kernel_size=9),
        ])
        
        # Attention mechanism
        self.attention = AttentionLayer(256, 128)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Multi-scale convolutions
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Prepare for attention (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Apply attention
        x, attention_weights = self.attention(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

class TransformerClassifier(nn.Module):
    """Transformer-based classifier"""
    
    def __init__(self, input_size, num_classes, embed_dim=256, num_heads=8, num_layers=4):
        super(TransformerClassifier, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(1, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(input_size, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_size)
        batch_size, seq_len = x.shape
        
        # Project to embedding dimension
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_projection(x)  # (batch_size, seq_len, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply transformer blocks
        x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        # Classification
        x = self.classifier(x)
        
        return x

class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average predictions
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

def advanced_train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, model_save_path):
    """Advanced training with learning rate scheduling and early stopping"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_accuracy = 0.0
    patience = 15
    patience_counter = 0
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = running_loss / len(test_loader.dataset)
        test_accuracy = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
        
        # Early stopping and model saving
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_accuracy, train_losses, test_losses, train_accuracies, test_accuracies

def main():
    print("=== Advanced Website Fingerprinting ML Training ===")
    
    # Load and preprocess data
    traces, labels, websites = train.load_dataset(DATASET_PATH)
    
    if traces is None or len(traces) < 30:
        print("Insufficient data for advanced training")
        return
    
    # Advanced preprocessing
    print("Applying advanced preprocessing...")
    preprocessor = AdvancedPreprocessor(INPUT_SIZE)
    processed_traces = preprocessor.extract_features(traces)
    
    print(f"Enhanced feature size: {len(processed_traces[0])}")
    
    # Update input size for enhanced features
    enhanced_input_size = len(processed_traces[0])
    
    # Split dataset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(sss.split(processed_traces, labels))
    
    train_dataset = train.WebsiteFingerprintDataset(
        [processed_traces[i] for i in train_indices],
        [labels[i] for i in train_indices]
    )
    
    test_dataset = train.WebsiteFingerprintDataset(
        [processed_traces[i] for i in test_indices],
        [labels[i] for i in test_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(websites)
    
    # Define advanced models
    models = {
        "Advanced CNN": AdvancedCNN(enhanced_input_size, num_classes),
        "Transformer": TransformerClassifier(enhanced_input_size, num_classes),
    }
    
    results = {}
    
    # Train individual models
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        model_save_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_model.pth")
        
        best_accuracy, train_losses, test_losses, train_accs, test_accs = advanced_train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS, model_save_path
        )
        
        # Evaluate
        model.load_state_dict(torch.load(model_save_path))
        predictions, true_labels = train.evaluate(model, test_loader, websites)
        
        results[model_name] = {
            'best_accuracy': best_accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accs,
            'test_accuracies': test_accs
        }
        
        print(f"{model_name} - Best Test Accuracy: {best_accuracy:.4f}")
    
    # Create ensemble
    print("\n=== Creating Ensemble Model ===")
    ensemble_models = [models[name] for name in models.keys()]
    for i, model in enumerate(ensemble_models):
        model_path = os.path.join(MODELS_DIR, f"{list(models.keys())[i].lower().replace(' ', '_')}_model.pth")
        model.load_state_dict(torch.load(model_path))
    
    ensemble = EnsembleModel(ensemble_models)
    ensemble_predictions, ensemble_true_labels = train.evaluate(ensemble, test_loader, websites)
    
    # Calculate ensemble accuracy
    ensemble_accuracy = sum(p == t for p, t in zip(ensemble_predictions, ensemble_true_labels)) / len(ensemble_predictions)
    results["Ensemble"] = {
        'best_accuracy': ensemble_accuracy,
        'predictions': ensemble_predictions,
        'true_labels': ensemble_true_labels
    }
    
    print(f"Ensemble - Test Accuracy: {ensemble_accuracy:.4f}")
    
    # Results summary
    print("\n=== Advanced Model Comparison ===")
    for model_name, result in results.items():
        accuracy = result['best_accuracy']
        print(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['best_accuracy'])
    best_accuracy = results[best_model_name]['best_accuracy']
    
    print(f"\nBest Advanced Model: {best_model_name} with {best_accuracy:.4f} accuracy")
    
    # Save advanced summary
    advanced_summary = {
        'dataset_info': {
            'total_traces': len(processed_traces),
            'enhanced_features': enhanced_input_size,
            'websites': websites,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset)
        },
        'advanced_techniques': [
            'Multi-resolution timing analysis',
            'Cache set monitoring',
            'Frequency domain features',
            'Attention mechanisms',
            'Transformer architecture',
            'Ensemble methods'
        ],
        'results': {name: {'accuracy': res['best_accuracy']} for name, res in results.items()},
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'improvement_over_basic': best_accuracy - 0.8883  # Compared to previous best
    }
    
    with open('advanced_training_summary.json', 'w') as f:
        json.dump(advanced_summary, f, indent=2)
    
    print(f"\nAdvanced training summary saved to advanced_training_summary.json")
    print(f"Improvement over basic models: {(best_accuracy - 0.8883)*100:.2f}%")

if __name__ == "__main__":
    main() 