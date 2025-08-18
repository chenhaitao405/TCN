import argparse
import os
import sys
from typing import List, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import load_config
from tcn import TCN
from dataloader import TcnDataset


def load_model(config, device: torch.device):
    '''Creates TCN and loads pretrained weights.'''
    model_info = torch.load(config.model_path, map_location=device)
    state_dict = model_info["state_dict"]
    del model_info["state_dict"]
    tcn = TCN(**model_info).to(device)
    tcn.load_state_dict(state_dict)
    return tcn


def compute_rmse(trial_names: List[str],
                 label_names: List[str],
                 estimates: torch.FloatTensor,
                 labels: torch.FloatTensor,
                 model_history: int,
                 trial_sequence_lengths: List[float],
                 model_delays: List[int]) -> Dict:
    '''Computes model RMSE relative to ground-truth and returns results.'''
    results = {}
    for i, trial_name in enumerate(trial_names):
        trial_results = {}
        for j, label_name in enumerate(label_names):
            # Extract estimates and labels
            estimate = estimates[i, j, model_history:trial_sequence_lengths[i]]
            label = labels[i, j, model_history:trial_sequence_lengths[i]]

            # Correct for any intentional delays in model estimates
            if model_delays[j] != 0:
                estimate = estimate[model_delays[j]:]
                label = label[:-model_delays[j]]

            # Ignore data points corresponding to nans
            valid_index = torch.where(~torch.isnan(estimate) & ~torch.isnan(label))
            estimate = estimate[valid_index]
            label = label[valid_index]

            # Compute RMSE
            rmse = torch.sqrt(torch.mean((estimate - label) ** 2)).item()
            trial_results[label_name] = rmse

        results[trial_name] = trial_results

    return results


def run_single_config(config_path: str, device: torch.device) -> Dict:
    '''Run a single configuration and return RMSE results.'''
    print(f"\nRunning config: {config_path}")

    # Load config
    config = load_config(config_path)

    # Load TCN
    tcn = load_model(config, device)
    tcn.train(False)

    # Load data
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]
    dataset = TcnDataset(data_dir=config.data_dir,
                         input_names=input_names,
                         label_names=label_names,
                         side=config.side,
                         participant_masses=config.participant_masses,
                         device=device)
    input_data, label_data, trial_sequence_lengths = dataset[:]

    # Compute model estimates
    with torch.no_grad():
        out = tcn(input_data)

    # Compute RMSE
    results = compute_rmse(dataset.get_trial_names(),
                           label_names,
                           out,
                           label_data,
                           tcn.get_effective_history(),
                           trial_sequence_lengths,
                           config.model_delays)

    return results


def plot_results(all_results: Dict, output_dir: str = "results"):
    '''Create visualization plots for RMSE results.'''
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    configs = list(all_results.keys())
    config_labels = [os.path.basename(c).replace('_config.py', '').replace('_', ' ').title()
                     for c in configs]

    # Get all trial names
    trial_names = list(next(iter(all_results.values())).keys())

    # Create individual plots for each trial
    for trial_name in trial_names:
        fig, ax = plt.subplots(figsize=(12, 6))

        hip_rmse = []
        knee_rmse = []

        for config in configs:
            hip_rmse.append(all_results[config][trial_name]['hip_flexion_r_moment'])
            knee_rmse.append(all_results[config][trial_name]['knee_angle_r_moment'])

        x = np.arange(len(config_labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, hip_rmse, width, label='Hip RMSE', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2, knee_rmse, width, label='Knee RMSE', color='coral', alpha=0.8)

        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('RMSE (Nm/kg)', fontsize=12)
        ax.set_title(f'RMSE Comparison - {trial_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{trial_name}_rmse.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Create average plot
    fig, ax = plt.subplots(figsize=(12, 6))

    avg_hip_rmse = []
    avg_knee_rmse = []
    std_hip_rmse = []
    std_knee_rmse = []

    for config in configs:
        hip_values = [all_results[config][trial]['hip_flexion_r_moment'] for trial in trial_names]
        knee_values = [all_results[config][trial]['knee_angle_r_moment'] for trial in trial_names]

        avg_hip_rmse.append(np.mean(hip_values))
        avg_knee_rmse.append(np.mean(knee_values))
        std_hip_rmse.append(np.std(hip_values))
        std_knee_rmse.append(np.std(knee_values))

    x = np.arange(len(config_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, avg_hip_rmse, width, yerr=std_hip_rmse,
                   label='Hip RMSE', color='steelblue', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width / 2, avg_knee_rmse, width, yerr=std_knee_rmse,
                   label='Knee RMSE', color='coral', alpha=0.8, capsize=5)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('RMSE (Nm/kg)', fontsize=12)
    ax.set_title('Average RMSE Comparison Across All Trials', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, err in zip(bars1, std_hip_rmse):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + err,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

    for bar, err in zip(bars2, std_knee_rmse):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + err,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save results to CSV
    results_df = []
    for config in configs:
        for trial in trial_names:
            results_df.append({
                'Config': os.path.basename(config).replace('_config.py', ''),
                'Trial': trial,
                'Hip_RMSE': all_results[config][trial]['hip_flexion_r_moment'],
                'Knee_RMSE': all_results[config][trial]['knee_angle_r_moment']
            })

    df = pd.DataFrame(results_df)
    df.to_csv(os.path.join(output_dir, 'rmse_results.csv'), index=False)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    for config in configs:
        config_name = os.path.basename(config).replace('_config.py', '').replace('_', ' ').title()
        hip_values = [all_results[config][trial]['hip_flexion_r_moment'] for trial in trial_names]
        knee_values = [all_results[config][trial]['knee_angle_r_moment'] for trial in trial_names]

        print(f"\n{config_name}:")
        print(f"  Hip RMSE:  Mean={np.mean(hip_values):.4f}, Std={np.std(hip_values):.4f}")
        print(f"  Knee RMSE: Mean={np.mean(knee_values):.4f}, Std={np.std(knee_values):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to host model and data.")
    parser.add_argument("--output_dir", type=str, default="rmse_results", help="Directory to save results.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Define all config files to run
    config_files = [
        "configs.default_config",
        "configs.sensor_selection.all_sensors_no_insole_config",
        "configs.sensor_selection.encoders_only_config",
        "configs.sensor_selection.imus_only_config",
        "configs.sensor_selection.insole_only_config",
        "configs.sensor_selection.no_insole_or_footimu_config"
    ]

    # Run all configurations
    all_results = {}
    for config_path in config_files:
        try:
            results = run_single_config(config_path, device)
            all_results[config_path] = results

            # Print results for this config
            print(f"\nResults for {config_path}:")
            for trial_name, trial_results in results.items():
                print(f"  {trial_name}:")
                for label_name, rmse in trial_results.items():
                    print(f"    {label_name}: {rmse:.6f} Nm/kg")
        except Exception as e:
            print(f"Error running config {config_path}: {str(e)}")
            continue

    # Create visualizations
    if all_results:
        plot_results(all_results, args.output_dir)
        print(f"\nResults saved to {args.output_dir}/")
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()