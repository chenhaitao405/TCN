"""
Unified visualization utilities for training and validation.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, List
from scipy import stats


class TrainingVisualizer:
    """Handle training visualization and logging."""

    def __init__(self, save_dir: str, use_tensorboard: bool = True):
        self.save_dir = save_dir
        self.use_tensorboard = use_tensorboard

        # Create directories
        self.plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize tensorboard if requested
        if use_tensorboard:
            self.tb_dir = os.path.join(save_dir, 'tensorboard')
            self.writer = SummaryWriter(self.tb_dir)
            print(f"Tensorboard logs saved to: {self.tb_dir}")
            print(f"Run 'tensorboard --logdir={self.tb_dir}' to view")
        else:
            self.writer = None

        # Storage for metrics
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'batch_losses': []
        }

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Log metrics for an epoch."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rate'].append(lr)

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
            self.writer.add_scalars('Loss_Comparison', {
                'Train': train_loss,
                'Validation': val_loss
            }, epoch)

    def log_batch(self, epoch: int, batch_idx: int, loss: float, total_batches: int):
        """Log batch-level metrics."""
        if self.writer:
            global_step = epoch * total_batches + batch_idx
            self.writer.add_scalar('Batch_Loss/Train', loss, global_step)
        self.metrics['batch_losses'].append(loss)

    def log_model_weights(self, model: Any, epoch: int):
        """Log model weight distributions."""
        if self.writer:
            for name, param in model.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def plot_training_curves(self, save_path: Optional[str] = None) -> str:
        """Create and save training curves."""
        if len(self.metrics['epochs']) == 0:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        ax1.plot(self.metrics['epochs'], self.metrics['train_loss'],
                label='Train Loss', marker='o', markersize=3)
        ax1.plot(self.metrics['epochs'], self.metrics['val_loss'],
                label='Validation Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning rate
        ax2 = axes[0, 1]
        ax2.plot(self.metrics['epochs'], self.metrics['learning_rate'],
                color='green', marker='o', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Loss difference (overfitting indicator)
        ax3 = axes[1, 0]
        loss_diff = np.array(self.metrics['val_loss']) - np.array(self.metrics['train_loss'])
        ax3.plot(self.metrics['epochs'], loss_diff,
                color='orange', marker='o', markersize=3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Val Loss - Train Loss')
        ax3.set_title('Overfitting Indicator')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Recent batch losses distribution
        ax4 = axes[1, 1]
        if len(self.metrics['batch_losses']) > 0:
            recent_losses = self.metrics['batch_losses'][-100:]  # Last 100 batches
            ax4.hist(recent_losses, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax4.axvline(x=np.mean(recent_losses), color='red',
                       linestyle='--', label=f'Mean: {np.mean(recent_losses):.4f}')
            ax4.set_xlabel('Loss Value')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Recent Batch Loss Distribution')
            ax4.legend()

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        return save_path

    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = os.path.join(self.save_dir, 'training_metrics.json')
        save_metrics = {
            'epochs': self.metrics['epochs'],
            'train_loss': self.metrics['train_loss'],
            'val_loss': self.metrics['val_loss'],
            'learning_rate': self.metrics['learning_rate'],
            'best_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None,
            'best_epoch': self.metrics['epochs'][np.argmin(self.metrics['val_loss'])] if self.metrics['val_loss'] else None
        }
        with open(metrics_file, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

    def close(self):
        """Close tensorboard writer."""
        if self.writer:
            self.writer.close()


class ValidationVisualizer:
    """Handle validation visualization."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.plots_dir = os.path.join(save_dir, 'validation_plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_metrics_summary(self, metrics: Any) -> str:
        """Create bar plots for RMSE and R² per label."""
        summary = metrics.get_summary()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Validation Metrics Summary', fontsize=16)

        # Prepare data
        labels = []
        rmse_means = []
        rmse_stds = []
        r2_means = []
        r2_stds = []

        for name in metrics.label_names:
            if summary[name]['n_samples'] > 0:
                labels.append(name)
                rmse_means.append(summary[name]['rmse_mean'])
                rmse_stds.append(summary[name]['rmse_std'])
                r2_means.append(summary[name]['r2_mean'])
                r2_stds.append(summary[name]['r2_std'])

        x = np.arange(len(labels))

        # RMSE plot
        ax1 = axes[0]
        ax1.bar(x, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Label')
        ax1.set_ylabel('RMSE (Nm/kg)')
        ax1.set_title('RMSE per Label')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(rmse_means):
            ax1.text(i, v + rmse_stds[i], f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        # R² plot
        ax2 = axes[1]
        bars = ax2.bar(x, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='green')
        ax2.set_xlabel('Label')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score per Label')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylim([0, 1.1])
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')
        ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='R²=0.7')
        ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='R²=0.9')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(r2_means):
            ax2.text(i, v + r2_stds[i], f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        # Color bars based on R² value
        for bar, r2 in zip(bars, r2_means):
            if r2 >= 0.9:
                bar.set_color('green')
            elif r2 >= 0.7:
                bar.set_color('orange')
            elif r2 >= 0.5:
                bar.set_color('yellow')
            else:
                bar.set_color('red')

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'metrics_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_predictions_vs_actual(self, metrics: Any, max_samples: int = 1000) -> str:
        """Create scatter plots of predictions vs actual values for each label."""
        n_labels = len(metrics.label_names)
        n_cols = min(3, n_labels)
        n_rows = (n_labels + n_cols - 1) // n_cols

        if n_labels == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        fig.suptitle('Predictions vs Actual Values', fontsize=16)

        for idx, name in enumerate(metrics.label_names):
            ax = axes[idx]

            estimates = np.array(metrics.metrics['per_label'][name]['estimates'])
            labels = np.array(metrics.metrics['per_label'][name]['labels'])

            if len(estimates) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{name}')
                continue

            # Sample if too many points
            if len(estimates) > max_samples:
                indices = np.random.choice(len(estimates), max_samples, replace=False)
                estimates_plot = estimates[indices]
                labels_plot = labels[indices]
            else:
                estimates_plot = estimates
                labels_plot = labels

            # Create scatter plot
            ax.scatter(labels_plot, estimates_plot, alpha=0.5, s=1)

            # Add perfect prediction line
            min_val = min(labels_plot.min(), estimates_plot.min())
            max_val = max(labels_plot.max(), estimates_plot.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect prediction')

            # Add regression line
            z = np.polyfit(labels_plot, estimates_plot, 1)
            p = np.poly1d(z)
            ax.plot([min_val, max_val], p([min_val, max_val]), 'g-', alpha=0.7, label='Regression line')

            # Calculate metrics for this label
            summary = metrics.get_summary()[name]

            # Add text with metrics
            text = f'RMSE: {summary["rmse_mean"]:.4f}\nR²: {summary["r2_mean"]:.3f}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Actual (Nm/kg)')
            ax.set_ylabel('Predicted (Nm/kg)')
            ax.set_title(f'{name}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_labels, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'predictions_vs_actual.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_error_distribution(self, metrics: Any) -> str:
        """Plot error distribution for each label."""
        n_labels = len(metrics.label_names)
        n_cols = min(3, n_labels)
        n_rows = (n_labels + n_cols - 1) // n_cols

        if n_labels == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        fig.suptitle('Error Distribution per Label', fontsize=16)

        for idx, name in enumerate(metrics.label_names):
            ax = axes[idx]

            estimates = np.array(metrics.metrics['per_label'][name]['estimates'])
            labels = np.array(metrics.metrics['per_label'][name]['labels'])

            if len(estimates) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{name}')
                continue

            # Calculate errors
            errors = estimates - labels

            # Create histogram
            ax.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero error')
            ax.axvline(x=np.mean(errors), color='green', linestyle='--', alpha=0.7,
                      label=f'Mean: {np.mean(errors):.4f}')

            # Add normal distribution overlay
            mu, std = np.mean(errors), np.std(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            ax2 = ax.twinx()
            ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', alpha=0.7, label='Normal fit')
            ax2.set_ylabel('Probability Density')

            ax.set_xlabel('Error (Predicted - Actual) [Nm/kg]')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add text with statistics
            text = f'Mean: {np.mean(errors):.4f}\nStd: {np.std(errors):.4f}\nSkew: {stats.skew(errors):.3f}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Hide empty subplots
        for idx in range(n_labels, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'error_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def create_metrics_table(self, metrics: Any) -> tuple:
        """Create a detailed metrics table and save as image."""
        summary = metrics.get_summary()

        # Prepare data for table
        data = []
        for name in metrics.label_names:
            if summary[name]['n_samples'] > 0:
                data.append([
                    name,
                    f"{summary[name]['rmse_mean']:.4f} ± {summary[name]['rmse_std']:.4f}",
                    f"{summary[name]['r2_mean']:.3f} ± {summary[name]['r2_std']:.3f}",
                    summary[name]['n_samples']
                ])

        # Add overall metrics
        data.append([
            'OVERALL',
            f"{summary['overall']['rmse']:.4f}",
            f"{summary['overall']['r2']:.3f}",
            '-'
        ])

        # Create table
        df = pd.DataFrame(data, columns=['Label', 'RMSE (mean ± std)', 'R² (mean ± std)', 'N Samples'])

        # Create figure and plot table
        fig, ax = plt.subplots(figsize=(12, len(data) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style the overall row
        for i in range(len(df.columns)):
            table[(len(data), i)].set_facecolor('#d4d4d4')
            table[(len(data), i)].set_text_props(weight='bold')

        # Color cells based on performance
        for i in range(1, len(data)):
            # Color R² cells
            r2_val = float(df.iloc[i - 1, 2].split(' ±')[0]) if i < len(data) else summary['overall']['r2']
            if not np.isnan(r2_val):
                if r2_val >= 0.9:
                    color = '#90EE90'  # Light green
                elif r2_val >= 0.7:
                    color = '#FFD700'  # Gold
                elif r2_val >= 0.5:
                    color = '#FFA500'  # Orange
                else:
                    color = '#FFB6C1'  # Light red
                table[(i, 2)].set_facecolor(color)

        plt.title('Validation Metrics Summary Table', fontsize=14, weight='bold', pad=20)

        save_path = os.path.join(self.plots_dir, 'metrics_table.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save as CSV
        csv_path = os.path.join(self.save_dir, 'metrics_table.csv')
        df.to_csv(csv_path, index=False)

        return save_path, csv_path