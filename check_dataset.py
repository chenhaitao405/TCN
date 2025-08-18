import argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

from config_utils import load_config
from dataloader import TcnDataset
from utils import collate_function


def check_single_trial(dataset, idx):
    """
    Detailed check of a single trial
    """
    print(f"\n{'=' * 60}")
    print(f"Checking trial {idx}: {dataset.trial_names[idx]}")
    print('=' * 60)

    try:
        # Load single trial
        input_data, label_data, seq_length = dataset[idx]

        print(f"Input shape: {input_data.shape}")
        print(f"Label shape: {label_data.shape}")
        print(f"Sequence length: {seq_length}")

        # Check for NaN in input
        input_nan_mask = torch.isnan(input_data)
        if input_nan_mask.any():
            nan_count = input_nan_mask.sum().item()
            total_elements = input_data.numel()
            print(
                f"\n⚠️  Input contains {nan_count}/{total_elements} NaN values ({100 * nan_count / total_elements:.1f}%)")

            # Find which channels have NaN
            for i in range(input_data.shape[1]):
                channel_nans = input_nan_mask[0, i, :].sum().item()
                if channel_nans > 0:
                    print(f"   Channel {i} ({dataset.input_names[i]}): {channel_nans} NaNs")
        else:
            print("✓ No NaN in input data")

        # Check for NaN in labels
        label_nan_mask = torch.isnan(label_data)
        if label_nan_mask.any():
            nan_count = label_nan_mask.sum().item()
            total_elements = label_data.numel()
            print(
                f"\n⚠️  Labels contain {nan_count}/{total_elements} NaN values ({100 * nan_count / total_elements:.1f}%)")

            # Find which labels have NaN
            for i in range(label_data.shape[1]):
                label_nans = label_nan_mask[0, i, :].sum().item()
                if label_nans > 0:
                    print(f"   Label {i} ({dataset.label_names[i]}): {label_nans} NaNs")
        else:
            print("✓ No NaN in label data")

        # Check for infinite values
        if torch.isinf(input_data).any():
            print("\n⚠️  Input contains infinite values!")
        if torch.isinf(label_data).any():
            print("\n⚠️  Labels contain infinite values!")

        # Check value ranges
        print(f"\nInput value range: [{input_data.min():.4f}, {input_data.max():.4f}]")
        print(f"Label value range: [{label_data.min():.4f}, {label_data.max():.4f}]")

        return True

    except Exception as e:
        print(f"❌ Error loading trial: {e}")
        return False


def check_raw_files(data_dir, trial_name):
    """
    Check the raw CSV files for a specific trial
    """
    print(f"\n{'=' * 60}")
    print(f"Checking raw files for: {trial_name}")
    print('=' * 60)

    trial_dir = os.path.join(data_dir, trial_name)

    # Find and check exo file
    exo_file = None
    for file in os.listdir(trial_dir):
        if file.lower().endswith("exo.csv") and not file.lower().endswith("power_exo.csv"):
            exo_file = os.path.join(trial_dir, file)
            break

    if exo_file:
        print(f"\nExo file: {os.path.basename(exo_file)}")
        try:
            df = pd.read_csv(exo_file)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.shape[1]}")

            # Check for NaN in each column
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                print(f"  ⚠️  Columns with NaN: {nan_cols}")
                for col in nan_cols:
                    nan_count = df[col].isna().sum()
                    print(f"     {col}: {nan_count}/{len(df)} NaNs")
            else:
                print("  ✓ No NaN values")

            # Check for empty columns
            empty_cols = [col for col in df.columns if df[col].isna().all()]
            if empty_cols:
                print(f"  ⚠️  Completely empty columns: {empty_cols}")

        except Exception as e:
            print(f"  ❌ Error reading file: {e}")

    # Find and check moment file
    moment_file = None
    for file in os.listdir(trial_dir):
        if file.lower().endswith("_moment_filt.csv"):
            moment_file = os.path.join(trial_dir, file)
            break

    if moment_file:
        print(f"\nMoment file: {os.path.basename(moment_file)}")
        try:
            df = pd.read_csv(moment_file)
            print(f"  Shape: {df.shape}")

            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                print(f"  ⚠️  Columns with NaN: {nan_cols}")
            else:
                print("  ✓ No NaN values")

        except Exception as e:
            print(f"  ❌ Error reading file: {e}")


def validate_dataset(config_path, device='cpu', check_raw=False):
    """
    Comprehensive dataset validation
    """
    # Load config
    config = load_config(config_path)
    device = torch.device(device)

    # Prepare data
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"Data directory: {config.data_dir}")
    print(f"Number of inputs: {len(input_names)}")
    print(f"Number of labels: {len(label_names)}")
    print(f"Side: {config.side}")

    # Check participant masses
    print(f"\nParticipant masses provided: {len(config.participant_masses)}")
    for participant, mass in config.participant_masses.items():
        if mass <= 0:
            print(f"  ⚠️  {participant}: Invalid mass {mass}")
        elif mass < 30 or mass > 200:
            print(f"  ⚠️  {participant}: Unusual mass {mass} kg")

    # Load dataset
    print("\nLoading dataset...")
    dataset = TcnDataset(
        data_dir=config.data_dir,
        input_names=input_names,
        label_names=label_names,
        side=config.side,
        participant_masses=config.participant_masses,
        device=device
    )

    print(f"Total trials: {len(dataset)}")

    # Check each trial
    print("\nChecking individual trials...")
    problematic_trials = []
    nan_trials = []

    for idx in tqdm(range(len(dataset)), desc="Validating trials"):
        try:
            input_data, label_data, seq_length = dataset[idx]

            # Check for NaN
            if torch.isnan(input_data).any():
                nan_trials.append((idx, dataset.trial_names[idx], "input"))
            if torch.isnan(label_data).any():
                nan_trials.append((idx, dataset.trial_names[idx], "label"))

            # Check for unusual values
            if input_data.abs().max() > 1000:
                problematic_trials.append((idx, dataset.trial_names[idx], "large_input"))
            if label_data.abs().max() > 1000:
                problematic_trials.append((idx, dataset.trial_names[idx], "large_label"))

        except Exception as e:
            problematic_trials.append((idx, dataset.trial_names[idx], str(e)))

    # Report results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    if nan_trials:
        print(f"\n⚠️  Trials with NaN values: {len(nan_trials)}/{len(dataset)}")
        for idx, trial_name, data_type in nan_trials[:10]:  # Show first 10
            print(f"   Trial {idx}: {trial_name} ({data_type})")
        if len(nan_trials) > 10:
            print(f"   ... and {len(nan_trials) - 10} more")
    else:
        print("\n✓ No NaN values found in dataset")

    if problematic_trials:
        print(f"\n⚠️  Problematic trials: {len(problematic_trials)}/{len(dataset)}")
        for idx, trial_name, issue in problematic_trials[:10]:
            print(f"   Trial {idx}: {trial_name} ({issue})")
    else:
        print("\n✓ No problematic trials found")

    # Test batch collation
    print("\n" + "=" * 60)
    print("TESTING BATCH COLLATION")
    print("=" * 60)

    from torch.utils.data import DataLoader

    # Test with small batch
    test_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: collate_function(x, device)
    )

    print("Testing batch loading...")
    for batch_idx, (inputs, labels, seq_lengths) in enumerate(test_loader):
        if batch_idx >= 3:  # Test first 3 batches
            break

        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Label shape: {labels.shape}")
        print(f"  Sequence lengths: {seq_lengths}")

        if torch.isnan(inputs).any():
            print(f"  ⚠️  Batch contains NaN in inputs!")
            # Find which samples have NaN
            for i in range(inputs.shape[0]):
                if torch.isnan(inputs[i]).any():
                    print(f"     Sample {i} has NaN")
        else:
            print(f"  ✓ No NaN in batch inputs")

    # Detailed check of problematic trials
    if nan_trials and check_raw:
        print("\n" + "=" * 60)
        print("DETAILED CHECK OF PROBLEMATIC TRIALS")
        print("=" * 60)

        for idx, trial_name, _ in nan_trials[:3]:  # Check first 3
            check_single_trial(dataset, idx)
            check_raw_files(config.data_dir, trial_name)

    return nan_trials, problematic_trials


def main():
    parser = argparse.ArgumentParser(description='Validate dataset for NaN and other issues')
    parser.add_argument('--config_path', type=str, default='configs.default_config.py',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--check_raw', action='store_true',
                        help='Check raw CSV files for problematic trials')
    parser.add_argument('--check_trial', type=int, default=None,
                        help='Check specific trial index in detail')

    args = parser.parse_args()

    if args.check_trial is not None:
        # Check specific trial
        config = load_config(args.config_path)
        device = torch.device(args.device)

        input_names = [name.replace("*", config.side) for name in config.input_names]
        label_names = [name.replace("*", config.side) for name in config.label_names]

        dataset = TcnDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device
        )

        check_single_trial(dataset, args.check_trial)
        if args.check_raw:
            check_raw_files(config.data_dir, dataset.trial_names[args.check_trial])
    else:
        # Full validation
        nan_trials, problematic_trials = validate_dataset(
            args.config_path,
            args.device,
            args.check_raw
        )

        # Suggest fixes
        if nan_trials:
            print("\n" + "=" * 60)
            print("SUGGESTED FIXES")
            print("=" * 60)
            print("1. Check if participant masses are correctly set (0 mass causes NaN)")
            print("2. Some trials may have missing sensor data")
            print("3. Consider filtering out trials with >50% NaN values")
            print("4. Add NaN handling in dataloader:")
            print("   - Replace NaN with 0 or interpolate")
            print("   - Skip trials with too many NaNs")


if __name__ == "__main__":
    main()