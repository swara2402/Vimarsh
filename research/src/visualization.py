"""
Visualization module for creating publication-quality plots.
Generates all required charts for the baseline experiment.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

DPI = 300

def set_plot_style():
    """Set consistent plot styling across all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')

# Set default style on import
set_plot_style()

def plot_class_distribution(df: pd.DataFrame, target_column: str, save_path: Path) -> None:
    """Plot class distribution of the target variable."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=target_column, data=df)
    plt.title('Class Distribution - Wine Quality', fontsize=12, fontweight='bold')
    plt.xlabel('Good Quality (1) / Lower Quality (0)')
    plt.ylabel('Count')
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom')
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to {save_path}")

def plot_model_accuracy_comparison(metrics_list: List[Dict], save_path: Path) -> None:
    """Plot accuracy comparison across all models."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_names = [m['model_name'].replace('_', ' ').title() for m in metrics_list]
    accuracies = [m['accuracy'] for m in metrics_list]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Model accuracy comparison saved to {save_path}")

def plot_roc_curves(models_y_probs: Dict[str, np.ndarray], y_test: np.ndarray, save_path: Path) -> None:
    """Plot ROC curves for all models."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for (model_name, y_prob), color in zip(models_y_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"ROC curves plot saved to {save_path}")

def plot_confusion_matrix(cm: List[List[int]], model_name: str, save_path: Path) -> None:
    """Plot confusion matrix for a model."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    cm_np = np.array(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix\n{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved for {model_name}")

def plot_confidence_distributions(predictions_df: pd.DataFrame, model_name: str, save_path_correct: Path, save_path_incorrect: Path) -> None:
    """Plot confidence distributions for correct vs incorrect predictions."""
    save_path_correct.parent.mkdir(parents=True, exist_ok=True)
    
    # Correct predictions confidence distribution
    plt.figure(figsize=(8, 6))
    correct_conf = predictions_df[predictions_df['correctness'] == 1]['confidence']
    sns.histplot(correct_conf, bins=20, kde=True, color='#2ecc71')
    plt.title(f'Confidence Distribution - Correct Predictions\n{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.xlim(0.5, 1.0)
    plt.savefig(save_path_correct, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Incorrect predictions confidence distribution
    plt.figure(figsize=(8, 6))
    incorrect_conf = predictions_df[predictions_df['correctness'] == 0]['confidence']
    if len(incorrect_conf) > 0:
        sns.histplot(incorrect_conf, bins=20, kde=True, color='#e74c3c')
        plt.title(f'Confidence Distribution - Incorrect Predictions\n{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.xlim(0.5, 1.0)
    else:
        plt.text(0.5, 0.5, 'No incorrect predictions', ha='center', fontsize=14)
        plt.title(f'No Incorrect Predictions\n{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    plt.savefig(save_path_incorrect, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Confidence distribution plots saved for {model_name}")

def plot_confidence_vs_accuracy(bucket_analysis: pd.DataFrame, model_name: str, save_path: Path) -> None:
    """Plot confidence vs empirical accuracy."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    x = range(len(bucket_analysis))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], bucket_analysis['average_confidence'], width, label='Average Confidence', color='#3498db')
    plt.bar([i + width/2 for i in x], bucket_analysis['actual_accuracy'], width, label='Actual Accuracy', color='#2ecc71')
    
    plt.xticks(x, bucket_analysis['bucket_label'])
    plt.xlabel('Confidence Bucket')
    plt.ylabel('Value')
    plt.title(f'Confidence vs Actual Accuracy\n{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    plt.legend()
    plt.ylim(0.5, 1.05)
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Confidence vs accuracy plot saved for {model_name}")

def plot_brier_score_comparison(metrics_list: List[Dict], save_path: Path) -> None:
    """Plot Brier score comparison across models."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_names = [m['model_name'].replace('_', ' ').title() for m in metrics_list]
    brier_scores = [m['brier_score'] for m in metrics_list]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, brier_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Brier Score Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Brier Score')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Brier score comparison saved to {save_path}")

def plot_ece_comparison(metrics_list: List[Dict], save_path: Path) -> None:
    """Plot Expected Calibration Error comparison across models."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_names = [m['model_name'].replace('_', ' ').title() for m in metrics_list]
    ece_scores = [m['expected_calibration_error'] for m in metrics_list]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, ece_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Expected Calibration Error (ECE) Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('ECE')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"ECE comparison saved to {save_path}")