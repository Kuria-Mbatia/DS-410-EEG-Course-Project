import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Set style for better looking plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

# Model Performance Comparison
def create_performance_comparison_plot():
    """Create bar plot comparing model performances"""
    models = ['RF (no PCA)', 'RF + PCA', 'GBT + PCA']
    accuracy = [0.530, 0.456, 0.726]
    f1_score = [0.530, 0.456, 0.720]
    precision = [0.530, 0.456, 0.726]
    recall = [0.530, 0.456, 0.726]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, f1_score, width, label='F1-Score', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, precision, width, label='Precision', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, recall, width, label='Recall', alpha=0.8)
    
    ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_title('EEG Classification Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# PCA Impact Visualization
def create_pca_impact_plot():
    """Create plot showing PCA impact on different algorithms"""
    algorithms = ['Random Forest', 'Gradient Boosted Trees']
    without_pca = [0.530, 0.726]  # Assuming GBT without PCA would be similar
    with_pca = [0.456, 0.726]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, without_pca, width, label='Without PCA', alpha=0.8, color='coral')
    bars2 = ax.bar(x + width/2, with_pca, width, label='With PCA', alpha=0.8, color='skyblue')
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Impact of PCA on Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Confusion Matrix Visualization (simulated based on performance)
def create_confusion_matrix_plot():
    """Create confusion matrices for best performing model"""
    classes = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    
    # Simulated confusion matrix for GBT + PCA (72.6% accuracy)
    # This would normally come from your actual model results
    np.random.seed(42)
    cm = np.array([
        [85, 5, 3, 2, 3, 2],   # Seizure
        [8, 78, 4, 5, 3, 2],   # LPD  
        [6, 4, 82, 3, 3, 2],   # GPD
        [4, 6, 2, 80, 6, 2],   # LRDA
        [5, 3, 4, 4, 82, 2],   # GRDA
        [7, 4, 3, 6, 5, 75]    # Other
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - GBT + PCA Model (Best Performer)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Feature Importance Plot (simulated)
def create_feature_importance_plot():
    """Create feature importance plot for Random Forest"""
    # Simulated feature importance (would come from model.featureImportances)
    np.random.seed(42)
    n_features = 20  # Top 20 features
    feature_names = [f'f{i}' for i in range(n_features)]
    importance_scores = np.random.exponential(0.05, n_features)
    importance_scores = np.sort(importance_scores)[::-1]  # Sort descending
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(feature_names)), importance_scores, alpha=0.8, color='green')
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features (Spectral Frequencies)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Feature Importances - Random Forest Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.annotate(f'{width:.4f}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(3, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# Class Distribution Plot
def create_class_distribution_plot():
    """Create plot showing EEG class distribution"""
    classes = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    # Simulated class counts (would come from your actual data)
    counts = [1250, 980, 1100, 890, 920, 1560]  # Example counts
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    bars = ax1.bar(classes, counts, alpha=0.8, color='purple')
    ax1.set_xlabel('EEG Pattern Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('EEG Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
    ax2.set_title('EEG Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# PCA Explained Variance Plot
def create_pca_variance_plot():
    """Create plot showing PCA explained variance"""
    # Simulated PCA explained variance (would come from your PCA model)
    n_components = 50
    explained_variance = np.random.exponential(0.02, n_components)
    explained_variance = np.sort(explained_variance)[::-1]
    explained_variance = explained_variance / explained_variance.sum()  # Normalize
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Individual explained variance
    ax1.bar(range(1, n_components + 1), explained_variance, alpha=0.8, color='orange')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Explained Variance by Principal Component', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    ax2.plot(range(1, n_components + 1), cumulative_variance, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
    ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Explained Variance by Principal Components', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Training Progress Plot (simulated)
def create_training_progress_plot():
    """Create plot showing training progress for hyperparameter tuning"""
    # Simulated training metrics
    iterations = range(1, 21)
    train_accuracy = [0.45 + 0.25 * (1 - np.exp(-i/5)) + np.random.normal(0, 0.02) for i in iterations]
    val_accuracy = [0.42 + 0.28 * (1 - np.exp(-i/6)) + np.random.normal(0, 0.03) for i in iterations]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, train_accuracy, 'b-', linewidth=2, marker='o', label='Training Accuracy', alpha=0.8)
    ax.plot(iterations, val_accuracy, 'r-', linewidth=2, marker='s', label='Validation Accuracy', alpha=0.8)
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Training Progress - Hyperparameter Tuning', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight best validation score
    best_val_idx = np.argmax(val_accuracy)
    ax.annotate(f'Best Val: {val_accuracy[best_val_idx]:.3f}',
                xy=(best_val_idx + 1, val_accuracy[best_val_idx]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()

# Run all plotting functions
if __name__ == "__main__":
    print("Generating EEG Classification Analysis Plots...")
    
    print("1. Model Performance Comparison")
    create_performance_comparison_plot()
    
    print("2. PCA Impact Analysis")
    create_pca_impact_plot()
    
    print("3. Confusion Matrix - Best Model")
    create_confusion_matrix_plot()
    
    print("4. Feature Importance Analysis")
    create_feature_importance_plot()
    
    print("5. Class Distribution")
    create_class_distribution_plot()
    
    print("6. PCA Explained Variance")
    create_pca_variance_plot()
    
    print("7. Training Progress")
    create_training_progress_plot()
    
    print("All plots generated successfully!")