"""
Refactored validation script for TCN model.
"""
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

# Import custom modules
from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader
from utils.data_loader import DataManager
from utils.metrics import ValidationMetrics, MetricsComputer
from utils.visualization import ValidationVisualizer


def validate_model(model, dataloader, device, config, label_names):
    """Validate model and compute per-label metrics."""
    model.eval()
    metrics = ValidationMetrics(label_names)
    computer = MetricsComputer()
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

                    # Ignore data points corresponding to nans
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
                    rmse = computer.compute_rmse(label, estimate).item()
                    r2 = computer.compute_r2(label, estimate)

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


def save_validation_results(metrics, save_dir, config_path, model_path):
    """Save validation results to JSON file."""
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
    results_file = save_dir + '/validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nValidation results saved to: {results_file}")
    return results_file


def print_validation_summary(metrics):
    """Print validation summary to console."""
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
    """Create and configure argument parser for validation."""
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
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualization plots')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to validate (for debugging)')
    return parser


def main():
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load and process config
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_path)
    config = config_manager.apply_sensor_selection(config)

    # Setup save directory
    save_dir = config_manager.setup_validation_directory(args.save_dir)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model_loader = ModelLoader()
    model, model_info = model_loader.load_pretrained_model(
        args.model_path, device, config, load_weights=True
    )
    print("Model loaded successfully!")

    # Prepare label names
    label_names = [name.replace("*", config.side) for name in config.label_names]
    print(f"\nValidating on labels: {label_names}")

    # Load dataset
    data_manager = DataManager()
    full_dataset = data_manager.load_datasets(config, device)

    # Get valid indices
    valid_indices = data_manager.get_or_compute_valid_indices(full_dataset, config)

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
        collate_fn=lambda x: data_manager.collate_function(x, device)
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

    # Save configuration
    config_manager.save_validation_config(args, config, save_dir)

    print(f"\n{'=' * 60}")
    print("VALIDATION COMPLETE!")
    print(f"All results saved to: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()