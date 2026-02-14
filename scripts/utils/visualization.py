"""
Visualization Utilities
Helper functions for plotting and visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history (accuracy, loss, etc.)
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
        show: Show plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Accuracy
    if 'accuracy' in history:
        axes[0, 0].plot(history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    if 'loss' in history:
        axes[0, 1].plot(history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        show: Show plot
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dataset_distribution(
    counts: Dict[str, int],
    title: str = "Dataset Distribution",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot dataset class distribution
    
    Args:
        counts: Dictionary of {class_name: count}
        title: Plot title
        save_path: Path to save figure
        show: Show plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = list(counts.keys())
    values = list(counts.values())
    colors = sns.color_palette("husl", len(classes))
    
    # Bar chart
    ax1.bar(classes, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title(f'{title} - Bar Chart', fontweight='bold')
    ax1.set_xlabel('Class', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax1.text(i, v + max(values)*0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(values, labels=classes, autopct='%1.1f%%', 
           colors=colors, startangle=90, textprops={'fontweight': 'bold'})
    ax2.set_title(f'{title} - Pie Chart', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: AUC score
        save_path: Path to save figure
        show: Show plot
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test visualization utilities
    print("="*70)
    print("VISUALIZATION UTILITIES TEST")
    print("="*70)
    print()
    
    # Test training history plot
    print("Testing plot_training_history...")
    dummy_history = {
        'accuracy': np.linspace(0.5, 0.95, 30) + np.random.normal(0, 0.02, 30),
        'val_accuracy': np.linspace(0.5, 0.92, 30) + np.random.normal(0, 0.03, 30),
        'loss': np.linspace(1.0, 0.1, 30) + np.random.normal(0, 0.05, 30),
        'val_loss': np.linspace(1.0, 0.15, 30) + np.random.normal(0, 0.06, 30),
        'precision': np.linspace(0.5, 0.94, 30) + np.random.normal(0, 0.02, 30),
        'val_precision': np.linspace(0.5, 0.91, 30) + np.random.normal(0, 0.03, 30),
        'recall': np.linspace(0.5, 0.93, 30) + np.random.normal(0, 0.02, 30),
        'val_recall': np.linspace(0.5, 0.90, 30) + np.random.normal(0, 0.03, 30),
    }
    
    plot_training_history(dummy_history, save_path="outputs/plots/test_history.png", show=False)
    print("✓ Training history plot created")
    print()
    
    # Test dataset distribution
    print("Testing plot_dataset_distribution...")
    dummy_counts = {'open': 5000, 'closed': 4800}
    plot_dataset_distribution(dummy_counts, save_path="outputs/plots/test_distribution.png", show=False)
    print("✓ Distribution plot created")
    print()
    
    # Test confusion matrix
    print("Testing plot_confusion_matrix...")
    dummy_cm = np.array([[450, 50], [30, 470]])
    plot_confusion_matrix(dummy_cm, ['Open', 'Closed'], save_path="outputs/plots/test_cm.png", show=False)
    print("✓ Confusion matrix created")
    
    print()
    print("="*70)
    print("Check outputs/plots/ for generated test plots")
    print("="*70)