#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set style for better looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def create_hyperparameter_charts():
    """Create charts showing hyperparameter experiment results"""
    
    # Data from actual experiments
    batch_experiments = [
        {'batch_size': 32, 'basic_cnn': 87.83, 'complex_cnn': 88.67},
        {'batch_size': 64, 'basic_cnn': 87.67, 'complex_cnn': 88.00},
        {'batch_size': 128, 'basic_cnn': 86.33, 'complex_cnn': 88.33}
    ]
    
    lr_experiments = [
        {'lr': '1e-5', 'basic_cnn': 76.17, 'complex_cnn': 87.33},
        {'lr': '1e-4', 'basic_cnn': 87.33, 'complex_cnn': 87.83},
        {'lr': '1e-3', 'basic_cnn': 87.50, 'complex_cnn': 89.17}
    ]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Batch size chart
    batch_sizes = [exp['batch_size'] for exp in batch_experiments]
    basic_acc = [exp['basic_cnn'] for exp in batch_experiments]
    complex_acc = [exp['complex_cnn'] for exp in batch_experiments]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    ax1.bar(x - width/2, basic_acc, width, label='Basic CNN', alpha=0.8)
    ax1.bar(x + width/2, complex_acc, width, label='Complex CNN', alpha=0.8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Impact of Batch Size on Model Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(85, 90)
    
    # Add value labels on bars
    for i, v in enumerate(basic_acc):
        ax1.text(i - width/2, v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(complex_acc):
        ax1.text(i + width/2, v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Learning rate chart
    lr_labels = [exp['lr'] for exp in lr_experiments]
    basic_lr = [exp['basic_cnn'] for exp in lr_experiments]
    complex_lr = [exp['complex_cnn'] for exp in lr_experiments]
    
    x2 = np.arange(len(lr_labels))
    
    ax2.bar(x2 - width/2, basic_lr, width, label='Basic CNN', alpha=0.8)
    ax2.bar(x2 + width/2, complex_lr, width, label='Complex CNN', alpha=0.8)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Impact of Learning Rate on Model Performance')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(lr_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(75, 90)
    
    # Add value labels on bars
    for i, v in enumerate(basic_lr):
        ax2.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(complex_lr):
        ax2.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created hyperparameter_comparison.png")

def create_data_size_chart():
    """Create chart showing impact of training data size"""
    
    # Actual experimental data
    data_sizes = [300, 600, 1200, 2000, 2999]
    accuracies = [83.33, 87.50, 80.83, 87.50, 86.00]
    
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, accuracies, 'o-', linewidth=3, markersize=8, color='#2E86AB')
    plt.xlabel('Training Data Size (total traces)')
    plt.ylabel('Accuracy (%)')
    plt.title('Impact of Training Data Size on Classification Performance')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(data_sizes, accuracies)):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    # Highlight the unexpected drop
    plt.annotate('Unexpected drop\n(overfitting)', xy=(1200, 80.83), xytext=(1500, 82),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', ha='center')
    
    plt.ylim(78, 89)
    plt.tight_layout()
    plt.savefig('results/data_size_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created data_size_impact.png")

def create_website_classification_chart():
    """Create chart showing per-website classification performance"""
    
    # Actual results
    websites = ['BUET Moodle', 'Google.com', 'Prothomalo']
    precision = [0.77, 0.91, 0.98]
    recall = [0.97, 0.70, 0.96]
    f1_score = [0.86, 0.79, 0.97]
    
    x = np.arange(len(websites))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Website')
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance by Website')
    ax.set_xticks(x)
    ax.set_xticklabels(websites)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig('results/website_classification_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created website_classification_performance.png")

def create_cache_pattern_chart():
    """Create chart showing cache pattern characteristics by website"""
    
    # Actual data from analysis
    websites = ['BUET Moodle', 'Google.com', 'Prothomalo']
    mean_counts = [41.98, 38.16, 41.40]
    std_devs = [5.10, 8.30, 6.73]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mean sweep counts
    bars1 = ax1.bar(websites, mean_counts, alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_ylabel('Mean Sweep Count')
    ax1.set_title('Average Cache Sweep Counts by Website')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, mean_counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Standard deviation (variability)
    bars2 = ax2.bar(websites, std_devs, alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Cache Pattern Variability by Website')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, std_devs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight Google's high variability
    ax2.annotate('Highest variability\n(hardest to classify)', 
                xy=(1, 8.30), xytext=(1.5, 9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/cache_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created cache_pattern_analysis.png")

def create_training_time_chart():
    """Create chart showing training time vs accuracy trade-offs"""
    
    # Data from experiments
    configs = ['Basic CNN\n(batch 64)', 'Complex CNN\n(batch 32)', 'Complex CNN\n(batch 64)', 'Complex CNN\n(LR 1e-3)']
    accuracies = [87.67, 88.67, 88.00, 89.17]
    times = [14.4, 75.2, 61.3, 46.5]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    scatter = ax.scatter(times, accuracies, s=200, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, config in enumerate(configs):
        ax.annotate(config, (times[i], accuracies[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Time vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)
    
    # Highlight the optimal point
    optimal_idx = accuracies.index(max(accuracies))
    ax.annotate('Best Performance', 
                xy=(times[optimal_idx], accuracies[optimal_idx]), 
                xytext=(times[optimal_idx]-10, accuracies[optimal_idx]+0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, color='green', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/training_time_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created training_time_vs_accuracy.png")

def create_summary_dashboard():
    """Create a comprehensive dashboard summarizing all key findings"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Best model performance
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Basic CNN', 'Complex CNN\n(baseline)', 'Complex CNN\n(optimized)']
    accuracies = [88.17, 88.33, 89.17]
    bars = ax1.bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylim(87, 90)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Website difficulty ranking
    ax2 = fig.add_subplot(gs[0, 1])
    websites = ['Prothomalo\n(Easiest)', 'BUET Moodle\n(Medium)', 'Google\n(Hardest)']
    f1_scores = [0.97, 0.86, 0.79]
    bars = ax2.bar(websites, f1_scores, color=['#2ECC40', '#FFDC00', '#FF4136'], alpha=0.8)
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Website Classification Difficulty')
    ax2.set_ylim(0.75, 1.0)
    for bar, f1 in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{f1:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Data size impact
    ax3 = fig.add_subplot(gs[1, :])
    data_sizes = [300, 600, 1200, 2000, 2999]
    accuracies_data = [83.33, 87.50, 80.83, 87.50, 86.00]
    ax3.plot(data_sizes, accuracies_data, 'o-', linewidth=3, markersize=8, color='#2E86AB')
    ax3.set_xlabel('Training Data Size (traces)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Non-linear Relationship: Training Data Size vs Performance')
    ax3.grid(True, alpha=0.3)
    # Highlight sweet spot
    ax3.axvspan(600, 2000, alpha=0.2, color='green', label='Optimal Range')
    ax3.legend()
    
    # 4. Cache pattern characteristics
    ax4 = fig.add_subplot(gs[2, 0])
    websites_cache = ['BUET\nMoodle', 'Google', 'Prothomalo']
    std_devs = [5.10, 8.30, 6.73]
    recall_rates = [0.97, 0.70, 0.96]
    
    # Scatter plot showing relationship between variability and recall
    scatter = ax4.scatter(std_devs, recall_rates, s=200, 
                         c=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolors='black')
    for i, website in enumerate(websites_cache):
        ax4.annotate(website, (std_devs[i], recall_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Pattern Variability (Std Dev)')
    ax4.set_ylabel('Recall Rate')
    ax4.set_title('Pattern Consistency vs Classification Success')
    ax4.grid(True, alpha=0.3)
    
    # 5. Key insights text
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    insights_text = """Key Experimental Insights:

    ✓ Hyperparameters matter more than architecture
    ✓ More data ≠ better performance (quality > quantity)  
    ✓ Pattern consistency predicts classification success
    ✓ Google surprisingly hardest despite simplicity
    ✓ Optimal training range: 600-2000 traces
    ✓ Learning rate 1e-3 optimal for complex models
    ✓ Cache side-channels remain viable attack vector"""
    
    ax5.text(0.05, 0.95, insights_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Website Fingerprinting Attack - Experimental Results Summary', 
                fontsize=16, fontweight='bold')
    
    plt.savefig('results/experimental_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created experimental_summary_dashboard.png")

def main():
    """Generate all charts from experimental results"""
    
    print("=== GENERATING CHARTS FROM EXPERIMENTAL RESULTS ===")
    print()
    
    # Ensure results directory exists
    import os
    os.makedirs('results', exist_ok=True)
    
    # Generate all charts
    create_hyperparameter_charts()
    create_data_size_chart()
    create_website_classification_chart()
    create_cache_pattern_chart()
    create_training_time_chart()
    create_summary_dashboard()
    
    print()
    print("=== ALL CHARTS GENERATED ===")
    print("Charts saved to results/ directory:")
    print("- hyperparameter_comparison.png")
    print("- data_size_impact.png")
    print("- website_classification_performance.png")
    print("- cache_pattern_analysis.png")
    print("- training_time_vs_accuracy.png")
    print("- experimental_summary_dashboard.png")
    print()
    print("Add these to your report with:")
    print("![Chart Description](results/chart_name.png)")

if __name__ == "__main__":
    main() 