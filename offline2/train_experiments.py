import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configuration for experiments
DATASET_PATH = "dataset.json"
MODELS_DIR = "experiment_models"
BATCH_SIZES = [32, 64, 128]
LEARNING_RATES = [1e-5, 1e-4, 1e-3]
EPOCHS = 20  # Reduced for faster experiments
TRAIN_SPLIT = 0.8 
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class WebsiteFingerprintDataset(Dataset):
    """Custom dataset class for website fingerprinting traces."""
    
    def __init__(self, traces, labels, transform=None):
        """
        Args:
            traces: List of trace data (each trace is a list of numbers)
            labels: List of labels corresponding to each trace
            transform: Optional transform to be applied on a sample
        """
        self.traces = traces
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        trace = torch.FloatTensor(trace)
        label = torch.LongTensor([label]).squeeze()
        
        if self.transform:
            trace = self.transform(trace)
        
        return trace, label


def load_dataset(dataset_path):
    """Load dataset from JSON file and return traces, labels, and website names."""
    print(f"Loading dataset from {dataset_path}...")
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Dataset file {dataset_path} not found!")
        print("Please run collect.py first to collect traces.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {dataset_path}")
        return None, None, None
    
    # Handle different JSON formats
    if 'traces' in data and 'websites' in data:
        # Database export format
        traces = []
        labels = []
        websites = list(data['websites'].keys())
        
        for website, website_traces in data['websites'].items():
            website_idx = websites.index(website)
            for trace_data in website_traces:
                traces.append(trace_data['trace'])
                labels.append(website_idx)
                
    elif 'website_data' in data:
        # Alternative format
        traces = []
        labels = []
        websites = list(data['website_data'].keys())
        
        for website, website_traces in data['website_data'].items():
            website_idx = websites.index(website)
            for trace_data in website_traces:
                traces.append(trace_data)
                labels.append(website_idx)
    else:
        print("Unknown dataset format!")
        return None, None, None
    
    print(f"Loaded {len(traces)} traces for {len(websites)} websites")
    print(f"Websites: {websites}")
    
    # Print class distribution
    from collections import Counter
    label_counts = Counter(labels)
    for website, count in zip(websites, [label_counts[i] for i in range(len(websites))]):
        print(f"  {website}: {count} traces")
    
    return traces, labels, websites


def preprocess_traces(traces, target_length=1000):
    """Preprocess traces to have consistent length and normalize."""
    print(f"Preprocessing traces to length {target_length}...")
    
    processed_traces = []
    
    for trace in traces:
        trace = np.array(trace, dtype=np.float32)
        
        # Truncate or pad to target length
        if len(trace) > target_length:
            trace = trace[:target_length]
        elif len(trace) < target_length:
            # Pad with zeros
            trace = np.pad(trace, (0, target_length - len(trace)), mode='constant')
        
        processed_traces.append(trace)
    
    processed_traces = np.array(processed_traces)
    
    # Normalize traces
    scaler = StandardScaler()
    processed_traces = scaler.fit_transform(processed_traces)
    
    print(f"Processed {len(processed_traces)} traces")
    print(f"Trace shape: {processed_traces.shape}")
    
    return processed_traces.tolist()  # Convert back to list for dataset


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
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



def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    """Train a PyTorch model and evaluate on the test set.
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
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
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    
    return best_accuracy



def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report with website names.
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for testing data
        website_names: List of website names for classification report
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report with website names instead of indices
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels


def main():
    """ Implement the main function to train and evaluate the models.
    1. Load the dataset from the JSON file, probably using a custom Dataset class
    2. Split the dataset into training and testing sets
    3. Create data loader for training and testing
    4. Define the models to train
    5. Train and evaluate each model
    6. Print comparison of results
    """
    
    print("=== Website Fingerprinting ML Training ===")
    
    # Step 1: Load dataset
    traces, labels, websites = load_dataset(DATASET_PATH)
    
    if traces is None:
        print("Failed to load dataset. Exiting.")
        return
    
    if len(traces) < 30:  # Need minimum data for training
        print(f"Not enough data for training. Have {len(traces)} traces, need at least 30.")
        print("Please collect more data using collect.py")
        return
    
    # Step 1.5: Preprocess traces
    traces = preprocess_traces(traces, INPUT_SIZE)
    
    # Step 2: Split dataset
    print("\nSplitting dataset...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_SPLIT, random_state=42)
    train_indices, test_indices = next(sss.split(traces, labels))
    
    # Create datasets
    train_dataset = WebsiteFingerprintDataset(
        [traces[i] for i in train_indices],
        [labels[i] for i in train_indices]
    )
    
    test_dataset = WebsiteFingerprintDataset(
        [traces[i] for i in test_indices],
        [labels[i] for i in test_indices]
    )
    
    print(f"Training set: {len(train_dataset)} traces")
    print(f"Test set: {len(test_dataset)} traces")
    
    # Step 3: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZES[0], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZES[0], shuffle=False)
    
    # Step 4: Define models to train
    num_classes = len(websites)
    
    models = {
        "Basic CNN": FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
        "Complex CNN": ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    }
    
    results = {}
    
    # Step 5: Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATES[0])
        model_save_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_model.pth")
        
        # Train model
        print(f"Training for {EPOCHS} epochs...")
        best_accuracy = train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, model_save_path)
        
        # Load best model and evaluate
        print(f"\nEvaluating {model_name}...")
        model.load_state_dict(torch.load(model_save_path))
        predictions, true_labels = evaluate(model, test_loader, websites)
        
        results[model_name] = {
            'best_accuracy': best_accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        print(f"{model_name} - Best Test Accuracy: {best_accuracy:.4f}")
    
    # Step 6: Print comparison of results
    print("\n=== Model Comparison ===")
    for model_name, result in results.items():
        accuracy = result['best_accuracy']
        print(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['best_accuracy'])
    best_accuracy = results[best_model_name]['best_accuracy']
    
    print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")
    
    # Save a summary
    summary = {
        'dataset_info': {
            'total_traces': len(traces),
            'websites': websites,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset)
        },
        'results': {name: {'accuracy': res['best_accuracy']} for name, res in results.items()},
        'best_model': best_model_name,
        'best_accuracy': best_accuracy
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to training_summary.json")
    
    if best_accuracy >= 0.6:
        print(f"✅ Success! Achieved target accuracy of 60%+")
    else:
        print(f"⚠️  Accuracy below 60%. Consider collecting more data or tuning hyperparameters.")

if __name__ == "__main__":
    main()
