#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import time
from train import *

def run_hyperparameter_experiments():
    """Run experiments with different hyperparameters"""
    
    # Load and preprocess data once
    print("Loading dataset...")
    traces, labels, websites = load_dataset("dataset.json")
    if traces is None:
        print("Failed to load dataset")
        return
    
    traces = preprocess_traces(traces, 1000)
    
    # Split dataset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(sss.split(traces, labels))
    
    train_traces = [traces[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_traces = [traces[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    results = []
    
    # Experiment configurations
    experiments = [
        # Batch size experiments
        {"name": "batch_32", "batch_size": 32, "lr": 1e-4, "epochs": 15},
        {"name": "batch_64", "batch_size": 64, "lr": 1e-4, "epochs": 15},
        {"name": "batch_128", "batch_size": 128, "lr": 1e-4, "epochs": 15},
        
        # Learning rate experiments  
        {"name": "lr_1e5", "batch_size": 64, "lr": 1e-5, "epochs": 15},
        {"name": "lr_1e4", "batch_size": 64, "lr": 1e-4, "epochs": 15},
        {"name": "lr_1e3", "batch_size": 64, "lr": 1e-3, "epochs": 15},
    ]
    
    for exp in experiments:
        print(f"\n=== Running Experiment: {exp['name']} ===")
        print(f"Batch Size: {exp['batch_size']}, Learning Rate: {exp['lr']}, Epochs: {exp['epochs']}")
        
        # Create datasets
        train_dataset = WebsiteFingerprintDataset(train_traces, train_labels)
        test_dataset = WebsiteFingerprintDataset(test_traces, test_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=exp['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=exp['batch_size'], shuffle=False)
        
        # Test both models
        models = {
            "Basic_CNN": FingerprintClassifier(1000, 128, len(websites)),
            "Complex_CNN": ComplexFingerprintClassifier(1000, 128, len(websites))
        }
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=exp['lr'])
            
            start_time = time.time()
            best_accuracy = train(model, train_loader, test_loader, criterion, optimizer, 
                                exp['epochs'], f"temp_{model_name}.pth")
            train_time = time.time() - start_time
            
            # Get detailed evaluation
            model.load_state_dict(torch.load(f"temp_{model_name}.pth"))
            predictions, true_labels = evaluate(model, test_loader, websites)
            
            # Calculate per-website accuracy
            website_accuracies = {}
            for i, website in enumerate(websites):
                website_mask = np.array(true_labels) == i
                if website_mask.sum() > 0:
                    website_acc = np.mean(np.array(predictions)[website_mask] == np.array(true_labels)[website_mask])
                    website_accuracies[website] = website_acc
            
            result = {
                "experiment": exp['name'],
                "model": model_name,
                "batch_size": exp['batch_size'],
                "learning_rate": exp['lr'],
                "epochs": exp['epochs'],
                "overall_accuracy": best_accuracy,
                "training_time_seconds": train_time,
                "website_accuracies": website_accuracies
            }
            
            results.append(result)
            print(f"{model_name} - Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            print(f"Training time: {train_time:.1f} seconds")
            
            # Clean up
            if os.path.exists(f"temp_{model_name}.pth"):
                os.remove(f"temp_{model_name}.pth")
    
    return results

def run_data_size_experiments():
    """Run experiments with different amounts of training data"""
    print("\n=== DATA SIZE EXPERIMENTS ===")
    
    # Load data
    traces, labels, websites = load_dataset("dataset.json")
    traces = preprocess_traces(traces, 1000)
    
    # Different data sizes to test
    data_sizes = [300, 600, 1200, 2000, 2999]  # Max is our full dataset
    results = []
    
    for size in data_sizes:
        if size > len(traces):
            continue
            
        print(f"\n--- Training with {size} total traces ---")
        
        # Stratified sampling to maintain class balance
        sss = StratifiedShuffleSplit(n_splits=1, train_size=min(size, len(traces)), random_state=42)
        subset_indices, _ = next(sss.split(traces, labels))
        
        subset_traces = [traces[i] for i in subset_indices]
        subset_labels = [labels[i] for i in subset_indices]
        
        # Train/test split
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, test_indices = next(sss2.split(subset_traces, subset_labels))
        
        train_dataset = WebsiteFingerprintDataset(
            [subset_traces[i] for i in train_indices],
            [subset_labels[i] for i in train_indices]
        )
        test_dataset = WebsiteFingerprintDataset(
            [subset_traces[i] for i in test_indices],
            [subset_labels[i] for i in test_indices]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Train basic model
        model = FingerprintClassifier(1000, 128, len(websites))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        accuracy = train(model, train_loader, test_loader, criterion, optimizer, 15, "temp_data_exp.pth")
        
        result = {
            "total_data_size": size,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "accuracy": accuracy
        }
        results.append(result)
        
        print(f"Size {size}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if os.path.exists("temp_data_exp.pth"):
            os.remove("temp_data_exp.pth")
    
    return results

def main():
    print("=== SYSTEMATIC ML EXPERIMENTS ===")
    
    # Run hyperparameter experiments
    hyperparameter_results = run_hyperparameter_experiments()
    
    # Run data size experiments  
    data_size_results = run_data_size_experiments()
    
    # Save all results
    all_results = {
        "hyperparameter_experiments": hyperparameter_results,
        "data_size_experiments": data_size_results,
        "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "total_traces": 2999,
            "websites": ["https://cse.buet.ac.bd/moodle/", "https://google.com", "https://prothomalo.com"]
        }
    }
    
    with open("experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n=== EXPERIMENT SUMMARY ===")
    print("Hyperparameter Results:")
    for result in hyperparameter_results:
        print(f"  {result['experiment']} - {result['model']}: {result['overall_accuracy']*100:.2f}%")
    
    print("\nData Size Results:")
    for result in data_size_results:
        print(f"  {result['total_data_size']} traces: {result['accuracy']*100:.2f}%")
    
    print(f"\nDetailed results saved to experiment_results.json")

if __name__ == "__main__":
    main() 