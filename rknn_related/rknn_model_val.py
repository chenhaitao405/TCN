import subprocess
import argparse
import torch
import numpy as np
import sys
sys.path.append(".")
from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader
from val import setup_validation_directory_with_model_name, print_validation_summary, save_validation_results
from utils.data_loader import DataManager
from torch.utils.data import DataLoader, Subset
from utils.metrics import ValidationMetrics, MetricsComputer
from utils.visualization import ValidationVisualizer
from tqdm import tqdm


def create_argument_parser():
    """Create and configure argument parser for training."""
    parser = argparse.ArgumentParser(description='Train TCN for joint moment estimation')
    parser.add_argument('--config_path', type=str, default='configs.default_config.py',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training')
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                        help='Whether to use pretrained weights (default: False)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--save_dir', type=str, default='validation_results',
                        help='Directory to save validation results')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualization plots')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to validate (for debugging)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--debug', action="store_true", default=False,
                        help='show more information')
    parser.add_argument('--int16', action="store_true", default=False,
                        help='show more information')
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_path)
    config = config_manager.apply_sensor_selection(config)
    save_dir = setup_validation_directory_with_model_name(args.save_dir, config.task_name)
    print(f"Created validation directory: {save_dir}")
    
    print(f"\nLoading model from {config.model_path}")
    model_loader = ModelLoader()
    model, model_info = model_loader.load_model(
        args.model_path, device, config, load_weights=True
    )
    print("Model loaded successfully!")
    
    sides = config.side if isinstance(config.side, list) else [config.side]
    for side in sides:
        label_names = [name.replace("*", side) for name in config.label_names]
        
    print(f"\nValidating on labels: {label_names}")
    use_sliding_window = getattr(config, 'use_sliding_window', False)
    data_manager = DataManager()
    full_dataset = data_manager.load_datasets(config, device, use_sliding_window=use_sliding_window)
    
    if use_sliding_window:
        # Get valid indices
        filtered_dataset = full_dataset
        val_loader = DataLoader(
            filtered_dataset,
            batch_size=args.batch_size,
            num_workers=16,
            pin_memory=True,  # 重要！预固定内存，加速GPU传输
            persistent_workers=True,  # 保持worker进程
            shuffle=False,  #是否随机打乱
        )

    else:
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
    
    # Validate model - pass dataset for action extraction
    print(f"\nValidating model on {len(filtered_dataset)} trials...")
    
    model.eval()
    metrics = ValidationMetrics(label_names)
    computer = MetricsComputer()
    model_history = model.get_effective_history()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch_idx, batch_data in enumerate(pbar):
            inputs, labels, seq_lengths, trial_names = batch_data
            use_sliding_window = getattr(config, 'use_sliding_window', False)
            if use_sliding_window:
                trial_names = trial_names['trial_name']
                
            batch_size = inputs.shape[0]
            L = inputs.shape[2]
            # Skip batch if input contains NaN
            if torch.isnan(inputs).any():
                continue
            
            save_inputs = inputs.unsqueeze(-2).numpy().astype(np.float32)  # (1,C,1,L)
            np.save("rknn_related/tmp_slice_inputs.npy", save_inputs, allow_pickle=True)
            
            result = subprocess.run(["adb", "push", "rknn_related/tmp_slice_inputs.npy", "/tmp/"]
                                    , stdout=None if args.debug else subprocess.DEVNULL)
            assert result.returncode == 0, "send npy file failed, check out usb wire connection"
            
            result = subprocess.run(["adb", "shell", "/root/apps/npy_infer/knee_torque_npy_infer", "/root/apps/model/torque_quantcn_8_sensors_int16.rknn" if args.int16 else "/root/apps/model/torque_quantcn_8_sensors_int8.rknn",
                "/tmp/tmp_slice_inputs.npy"], stdout=None if args.debug else subprocess.DEVNULL)
            assert result.returncode == 0, "model infer error occur"

            local_adb_bin = "rknn_related/adb_data.bin"
            result = subprocess.run(
                ["adb", "pull", "/tmp/adb_data.bin", local_adb_bin],
                stdout=None if args.debug else subprocess.DEVNULL,
                stderr=None if args.debug else subprocess.DEVNULL
            )
            assert result.returncode == 0, "adb pull /tmp/adb_data.bin failed"
            
            with open(local_adb_bin, "rb") as f:
                outputs = np.fromfile(f, dtype=np.float32)
                expected_count = batch_size * L
                assert outputs.size == expected_count, (
                    f"invalid output length from adb_data.bin, got {outputs.size}, expected {expected_count}"
                )
                estimates = torch.from_numpy(outputs).float().reshape(batch_size, -1, L)
            if torch.isnan(estimates).any():
                continue
            
            action_types = []
            for trial_name in trial_names:
                # Get the actual dataset to access extract_action_type method
                # Assuming dataset is ConcatDataset, get first dataset
                if hasattr(full_dataset, 'datasets'):
                    action_type = full_dataset.datasets[0].extract_action_type(trial_name)
                else:
                    # Fallback: basic extraction
                    folder_name = trial_name.split('/')[-1] if '/' in trial_name else trial_name
                    action_type = folder_name.split('_')[0]
                action_types.append(action_type)

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

                # Add this sample's results to metrics with action type
                metrics.add_batch_results(
                    sample_estimates, sample_labels,
                    sample_rmse, sample_r2,
                    action_types=[action_types[i]]  # Pass the action type for this sample
                )

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
                
    # metrics.compute_overall_metrics()
    all_rmse = []
    all_r2 = []
    for name in metrics.label_names:
        label_metrics = metrics.metrics['per_label'][name]
        avg_rmse = computer.compute_rmse(torch.tensor(label_metrics['labels']),
                                         torch.tensor(label_metrics['estimates'])).item()
        avg_r2 = computer.compute_r2(torch.tensor(label_metrics['labels']), torch.tensor(label_metrics['estimates']))
        all_rmse.append(avg_rmse)
        all_r2.append(avg_r2)
    metrics.metrics['overall']['rmse'] = np.mean(all_rmse) if all_rmse else np.nan
    metrics.metrics['overall']['r2'] = np.mean(all_r2) if all_r2 else np.nan
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
        plot_path = visualizer.plot_metrics_summary(metrics,config)
        plots.append(plot_path)
        print(f"  - Metrics summary saved to: {plot_path}")

        # NEW: Per-action metrics plots
        plot_paths = visualizer.plot_per_action_metrics(metrics, config)
        for plot_path in plot_paths:
            plots.append(plot_path)
            print(f"  - Per-action metrics saved to: {plot_path}")

        # Predictions vs actual scatter plots
        plot_path = visualizer.plot_predictions_vs_actual(metrics,config)
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