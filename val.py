import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config_utils import load_config
from dataloader import TcnDataset
from tcn import TCN
from utils import collate_function
import inspect


class ValidationMetrics:
    """Class to compute and store validation metrics"""

    def __init__(self, label_names):
        self.label_names = label_names
        self.metrics = {
            'per_label': {name: {'rmse': [], 'r2': [], 'estimates': [], 'labels': []}
                          for name in label_names},
            'overall': {'rmse': None, 'r2': None}
        }

    def compute_r2(self, y_true, y_pred):
        """Compute R-squared (coefficient of determination)"""
        if len(y_true) == 0:
            return float('nan')

        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

        # Handle edge case where ss_tot is 0
        if ss_tot == 0:
            return float('nan')

        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def compute_rmse(self, y_true, y_pred):
        """Compute Root Mean Square Error"""
        if len(y_true) == 0:
            return float('nan')
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

    def add_batch_results(self, estimates, labels, rmse_values, r2_values):
        """Store batch results for later aggregation"""
        for i, name in enumerate(self.label_names):
            # Check if i is within bounds and rmse is not NaN
            if i < len(rmse_values) and not np.isnan(rmse_values[i]):
                self.metrics['per_label'][name]['rmse'].append(rmse_values[i])
                self.metrics['per_label'][name]['r2'].append(r2_values[i])
                if i in estimates and estimates[i] is not None:
                    self.metrics['per_label'][name]['estimates'].extend(estimates[i].cpu().numpy())
                if i in labels and labels[i] is not None:
                    self.metrics['per_label'][name]['labels'].extend(labels[i].cpu().numpy())

    def compute_overall_metrics(self):
        """Compute overall metrics across all labels"""
        all_rmse = []
        all_r2 = []

        for name in self.label_names:
            label_metrics = self.metrics['per_label'][name]
            if label_metrics['rmse']:
                # Compute weighted average based on number of samples
                avg_rmse = np.mean(label_metrics['rmse'])
                avg_r2 = np.mean(label_metrics['r2'])
                all_rmse.append(avg_rmse)
                all_r2.append(avg_r2)

        self.metrics['overall']['rmse'] = np.mean(all_rmse) if all_rmse else float('nan')
        self.metrics['overall']['r2'] = np.mean(all_r2) if all_r2 else float('nan')

    def get_summary(self):
        """Get summary statistics for all metrics"""
        summary = {}

        for name in self.label_names:
            label_metrics = self.metrics['per_label'][name]
            if label_metrics['rmse']:
                summary[name] = {
                    'rmse_mean': np.mean(label_metrics['rmse']),
                    'rmse_std': np.std(label_metrics['rmse']),
                    'r2_mean': np.mean(label_metrics['r2']),
                    'r2_std': np.std(label_metrics['r2']),
                    'n_samples': len(label_metrics['estimates'])
                }
            else:
                summary[name] = {
                    'rmse_mean': float('nan'),
                    'rmse_std': float('nan'),
                    'r2_mean': float('nan'),
                    'r2_std': float('nan'),
                    'n_samples': 0
                }

        summary['overall'] = self.metrics['overall']
        return summary


class ValidationVisualizer:
    """Handle validation visualization"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.plots_dir = os.path.join(save_dir, 'validation_plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_metrics_summary(self, metrics: ValidationMetrics):
        """Create bar plots for RMSE and R² per label"""
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

    def plot_predictions_vs_actual(self, metrics: ValidationMetrics, max_samples=1000):
        """Create scatter plots of predictions vs actual values for each label"""
        n_labels = len(metrics.label_names)
        n_cols = min(3, n_labels)  # Use fewer columns if fewer labels
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

    def plot_error_distribution(self, metrics: ValidationMetrics):
        """Plot error distribution for each label"""
        n_labels = len(metrics.label_names)
        n_cols = min(3, n_labels)  # Use fewer columns if fewer labels
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
            from scipy import stats
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

    def create_metrics_table(self, metrics: ValidationMetrics):
        """Create a detailed metrics table and save as image"""
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


def load_pretrained_model(model_path: str, device: torch.device, config) -> tuple:
    """Load TCN model with pretrained weights"""
    model_info = torch.load(model_path, map_location=device)
    state_dict = model_info.get("state_dict", None)

    # Get TCN initialization parameters
    tcn_signature = inspect.signature(TCN.__init__)
    tcn_param_names = [param.name for param in tcn_signature.parameters.values()
                       if param.name != 'self']

    # Only pass parameters that TCN needs
    tcn_params = {k: v for k, v in model_info.items()
                  if k in tcn_param_names}

    # Create model
    tcn = TCN(**tcn_params).to(device)

    # Load pretrained weights
    if state_dict is not None:
        tcn.load_state_dict(state_dict)
        print("Loaded pretrained weights successfully!")
    else:
        raise ValueError("No state_dict found in model file!")

    return tcn, model_info


def validate_model(model, dataloader, device, config, label_names):
    """Validate model and compute per-label metrics"""
    model.eval()
    metrics = ValidationMetrics(label_names)
    model_history = model.get_effective_history()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')

        for batch_idx, (inputs, labels, seq_lengths) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.shape[0]

            # Skip batch if input contains NaN
            if torch.isnan(inputs).any():
                continue

            # Forward pass
            estimates = model(inputs)

            # Skip if output contains NaN
            if torch.isnan(estimates).any():
                continue

            # Process each sample in batch
            for i in range(batch_size):
                sample_rmse = []
                sample_r2 = []
                sample_estimates = {}
                sample_labels = {}

                for j, label_name in enumerate(label_names):
                    # Extract estimates and labels for this sample and label
                    # Ignore any starting or ending sequences that used zero padding
                    estimate = estimates[i, j, model_history:seq_lengths[i]]
                    label = labels[i, j, model_history:seq_lengths[i]]

                    # Correct for any intentional delays in model estimates
                    if config.model_delays[j] != 0:
                        if config.model_delays[j] > 0:
                            estimate = estimate[config.model_delays[j]:]
                            label = label[:-config.model_delays[j]]
                        else:
                            estimate = estimate[:config.model_delays[j]]
                            label = label[-config.model_delays[j]:]

                    # Ignore data points corresponding to nans in input or label data
                    valid_index = torch.where(~torch.isnan(estimate) & ~torch.isnan(label))
                    estimate = estimate[valid_index]
                    label = label[valid_index]

                    # Skip if no valid data
                    if len(estimate) == 0:
                        sample_rmse.append(float('nan'))
                        sample_r2.append(float('nan'))
                        sample_estimates[j] = None
                        sample_labels[j] = None
                        continue

                    # Compute metrics
                    rmse = metrics.compute_rmse(label, estimate)
                    r2 = metrics.compute_r2(label, estimate)

                    sample_rmse.append(rmse)
                    sample_r2.append(r2)
                    sample_estimates[j] = estimate
                    sample_labels[j] = label

                # Add this sample's results to metrics
                metrics.add_batch_results(sample_estimates, sample_labels, sample_rmse, sample_r2)

            # Update progress bar with current batch statistics
            all_rmse = []
            for name in label_names:
                if metrics.metrics['per_label'][name]['rmse']:
                    all_rmse.extend(metrics.metrics['per_label'][name]['rmse'])

            if all_rmse:
                pbar.set_postfix({
                    'avg_rmse': f'{np.mean(all_rmse):.4f}',
                    'n_samples': len(all_rmse)
                })

    # Compute overall metrics
    metrics.compute_overall_metrics()

    return metrics


def save_validation_results(metrics: ValidationMetrics, save_dir: str, config_path: str, model_path: str):
    """Save validation results to JSON file"""
    summary = metrics.get_summary()

    # Prepare results dictionary
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_path': config_path,
        'model_path': model_path,
        'per_label_metrics': {},
        'overall_metrics': summary['overall']
    }

    # Add per-label metrics
    for name in metrics.label_names:
        results['per_label_metrics'][name] = summary[name]

    # Save to JSON
    results_file = os.path.join(save_dir, 'validation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nValidation results saved to: {results_file}")
    return results_file


def print_validation_summary(metrics: ValidationMetrics):
    """Print validation summary to console"""
    summary = metrics.get_summary()

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print("\nPer-Label Metrics:")
    print("-" * 60)

    results_found = False
    for name in metrics.label_names:
        if summary[name]['n_samples'] > 0:
            results_found = True
            print(f"\n{name}:")
            print(f"  RMSE: {summary[name]['rmse_mean']:.4f} ± {summary[name]['rmse_std']:.4f} Nm/kg")
            print(f"  R²:   {summary[name]['r2_mean']:.3f} ± {summary[name]['r2_std']:.3f}")
            print(f"  N samples: {summary[name]['n_samples']}")
        else:
            print(f"\n{name}: No valid samples")

    if results_found:
        print("\n" + "-" * 60)
        print("Overall Metrics:")
        print(f"  Average RMSE: {summary['overall']['rmse']:.4f} Nm/kg")
        print(f"  Average R²:   {summary['overall']['r2']:.3f}")
    else:
        print("\n" + "-" * 60)
        print("No valid results found for any label")
    print("=" * 60)


def create_argument_parser():
    """Create and configure argument parser for validation"""
    parser = argparse.ArgumentParser(description='Validate TCN model for joint moment estimation')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for validation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--save_dir', type=str, default='validation_results',
                        help='Directory to save validation results')
    parser.add_argument('--data_split', type=str, default='all',
                        choices=['all', 'train', 'val', 'test'],
                        help='Which data split to validate on')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualization plots')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to validate (for debugging)')
    return parser


def apply_sensor_selection(config):
    """Apply sensor_pick filtering to config if specified"""
    if hasattr(config, 'sensor_pick') and config.sensor_pick:
        if hasattr(config, 'input_names'):
            original_input_names = config.input_names.copy()
            filtered_input_names = [config.input_names[i] for i in config.sensor_pick
                                    if i < len(config.input_names)]
            config.input_names = filtered_input_names

            print(f"Sensor selection applied:")
            print(f"  - Original number of inputs: {len(original_input_names)}")
            print(f"  - Selected sensor indices: {config.sensor_pick}")
            print(f"  - Number of selected inputs: {len(config.input_names)}")

    return config


def main():
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config_path)
    config = apply_sensor_selection(config)

    # Setup save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'validation_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, model_info = load_pretrained_model(args.model_path, device, config)
    print("Model loaded successfully!")

    # Prepare data
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    print(f"\nValidating on labels: {label_names}")

    # Load dataset
    print("\nLoading dataset...")
    datasets = []

    for data_dir in config.data_dirs:
        print(f"Loading data from: {data_dir}")
        dataset = TcnDataset(
            data_dir=data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device
        )
        datasets.append(dataset)
        print(f"  - Loaded {len(dataset)} trials")

    # Combine all datasets
    full_dataset = ConcatDataset(datasets)
    print(f"Total dataset size: {len(full_dataset)} trials")

    # Filter out trials with NaN
    print("\nFiltering trials with NaN...")
    valid_indices = []
    for i in tqdm(range(len(full_dataset)), desc="Checking trials"):
        inputs, labels, seq_lengths = full_dataset[i]
        if not torch.isnan(inputs).any() and not torch.isnan(labels).any():
            valid_indices.append(i)

    print(
        f"Valid trials: {len(valid_indices)}/{len(full_dataset)} ({100 * len(valid_indices) / len(full_dataset):.1f}%)")

    # Apply max_samples limit if specified
    if args.max_samples and len(valid_indices) > args.max_samples:
        valid_indices = valid_indices[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for validation")

    # Create filtered dataset
    filtered_dataset = Subset(full_dataset, valid_indices)

    # Create dataloader
    val_loader = DataLoader(
        filtered_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_function(x, device)
    )

    # Validate model
    print(f"\nValidating model on {len(filtered_dataset)} trials...")
    metrics = validate_model(model, val_loader, device, config, label_names)

    # Print summary
    print_validation_summary(metrics)

    # Save results
    results_file = save_validation_results(metrics, save_dir, args.config_path, args.model_path)

    # Create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        visualizer = ValidationVisualizer(save_dir)

        # Create all plots
        plots = []

        # Metrics summary bar plot
        plot_path = visualizer.plot_metrics_summary(metrics)
        plots.append(plot_path)
        print(f"  - Metrics summary saved to: {plot_path}")

        # Predictions vs actual scatter plots
        plot_path = visualizer.plot_predictions_vs_actual(metrics)
        plots.append(plot_path)
        print(f"  - Predictions vs actual saved to: {plot_path}")

        # Error distribution plots
        plot_path = visualizer.plot_error_distribution(metrics)
        plots.append(plot_path)
        print(f"  - Error distribution saved to: {plot_path}")

        # Metrics table
        table_path, csv_path = visualizer.create_metrics_table(metrics)
        plots.append(table_path)
        print(f"  - Metrics table saved to: {table_path}")
        print(f"  - Metrics CSV saved to: {csv_path}")

        print(f"\nAll visualizations saved to: {visualizer.plots_dir}")

    print(f"\n{'=' * 60}")
    print("VALIDATION COMPLETE!")
    print(f"All results saved to: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()