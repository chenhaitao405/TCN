import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

# Import custom modules
from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader
from utils.data_loader import DataManager
from utils.metrics import RMSELoss, MetricsComputer
from utils.visualization import TrainingVisualizer


def train_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        config,
        epoch,
        total_epochs,
        visualizer=None,
        clip_grad_norm=1.0
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    nan_count = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    total_batches = len(dataloader)
    computer = MetricsComputer()

    for batch_idx, (inputs, labels, seq_lengths,_) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"Warning: NaN detected in model output at batch {batch_idx}")
            nan_count += 1
            continue

        # Process batch to handle padding and delays
        processed_outputs, processed_labels = computer.process_batch(
            outputs, labels,
            model.get_effective_history(),
            seq_lengths,
            config.model_delays
        )

        # Skip if no valid samples
        if processed_outputs.numel() == 0:
            continue

        # Compute loss
        loss = criterion(processed_outputs, processed_labels)

        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}")
            nan_count += 1
            continue

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        # Update metrics
        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1

        # Log batch metrics
        if visualizer:
            visualizer.log_batch(epoch, batch_idx, loss_value, total_batches)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}' if num_batches > 0 else 'N/A',
            'nan_batches': nan_count
        })

    if nan_count > 0:
        print(f"Warning: {nan_count} batches with NaN encountered in epoch {epoch}")

    # Log model weights distribution
    if visualizer and epoch % 5 == 0:
        visualizer.log_model_weights(model, epoch)

    return total_loss / num_batches if num_batches > 0 else 0


def validate_epoch(
        model,
        dataloader,
        criterion,
        device,
        config,
        epoch,
        total_epochs
):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_losses = []
    computer = MetricsComputer()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Valid]')
        for inputs, labels, seq_lengths,_ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Skip batch if input contains NaN
            if torch.isnan(inputs).any():
                continue

            # Forward pass
            outputs = model(inputs)

            # Skip if output contains NaN
            if torch.isnan(outputs).any():
                continue

            # Process batch
            processed_outputs, processed_labels = computer.process_batch(
                outputs, labels,
                model.get_effective_history(),
                seq_lengths,
                config.model_delays
            )

            # Skip if no valid samples
            if processed_outputs.numel() == 0:
                continue

            # Compute loss
            loss = criterion(processed_outputs, processed_labels)

            if not torch.isnan(loss):
                loss_value = loss.item()
                total_loss += loss_value
                all_losses.append(loss_value)
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}' if num_batches > 0 else 'N/A'
                })

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    # Print validation statistics
    if all_losses:
        print(f"  Val Stats - Mean: {np.mean(all_losses):.4f}, "
              f"Std: {np.std(all_losses):.4f}, "
              f"Min: {np.min(all_losses):.4f}, "
              f"Max: {np.max(all_losses):.4f}")

    return avg_loss


def create_argument_parser():
    """Create and configure argument parser for training."""
    parser = argparse.ArgumentParser(description='Train TCN for joint moment estimation')
    parser.add_argument('--config_path', type=str, default='configs.default_config.py',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                        help='Whether to use pretrained weights (default: False)')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use tensorboard for visualization')
    parser.add_argument('--plot_interval', type=int, default=5,
                        help='Update plots every N epochs')
    return parser


def prepare_data(config, args, device):
    """Prepare data with sliding window support for training."""

    # Initialize data manager
    data_manager = DataManager()
    device_cpu = torch.device("cpu")
    # Get sliding window configuration
    use_sliding_window = getattr(config, 'use_sliding_window', False)

    if config.split_mode == 'manual':
        # Manual split mode (by directories)
        if not hasattr(config, 'train_data_dirs') or not hasattr(config, 'val_dataset'):
            raise ValueError("Manual split mode requires 'train_data_dirs' and 'val_dataset' in config")

        # Print configuration
        print("=" * 50)
        print("Data Loading Configuration:")
        print(f"  Split Mode: Manual")
        print(f"  Training Mode: {'Sliding Window' if use_sliding_window else 'Full Trial'}")
        if use_sliding_window:
            print(f"  Window Size: {config.window_size}")
            print(f"  Window Stride: {config.window_stride}")
        print("=" * 50)

        # Create manual split
        train_dataset, val_dataset = data_manager.create_manual_split(
            config=config,
            device=device_cpu,
            max_samples=args.max_samples if hasattr(args, 'max_samples') else None
        )

        # Create data loaders
        train_loader, val_loader = data_manager.create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            device=device_cpu
        )

        print(f"\nManual split completed:")
        if use_sliding_window:
            print(f"  Training: {len(train_dataset)} windows, {len(train_loader)} batches")
        else:
            print(f"  Training: {len(train_dataset)} trials, {len(train_loader)} batches")
        print(f"  Testing: {len(val_dataset)} trials, {len(val_loader)} batches")

    else:  # random split
        # Random split mode
        print("=" * 50)
        print("Data Loading Configuration:")
        print(f"  Split Mode: Random")
        print(f"  Training Mode: {'Sliding Window' if use_sliding_window else 'Full Trial'}")
        if use_sliding_window:
            print(f"  Window Size: {config.window_size}")
            print(f"  Window Stride: {config.window_stride}")
        print("=" * 50)

        # Load full dataset
        full_dataset = data_manager.load_datasets(
            config=config,
            device=device,
            use_sliding_window=use_sliding_window
        )

        # Get validation split ratio
        val_split = getattr(config, 'val_split', args.val_split if hasattr(args, 'val_split') else 0.1)

        # Create train/val split
        train_dataset, val_dataset = data_manager.create_train_val_split(
            full_dataset=full_dataset,
            config=config,
            val_split=val_split,
            max_samples=args.max_samples if hasattr(args, 'max_samples') else None,
            use_sliding_window=use_sliding_window
        )

        # Create data loaders
        train_loader, val_loader = data_manager.create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            device=device
        )

        print(f"\nRandom split completed:")
        if use_sliding_window:
            print(f"  Training: {len(train_dataset)} windows, {len(train_loader)} batches")
            print(f"  Validation: {len(val_dataset)} windows, {len(val_loader)} batches")
        else:
            print(f"  Training: {len(train_dataset)} trials, {len(train_loader)} batches")
            print(f"  Validation: {len(val_dataset)} trials, {len(val_loader)} batches")

    return train_loader, val_loader

def setup_training_directory_with_model_name(base_dir, model_path):
    """
    Create training directory with model path name and timestamp.
    
    Args:
        base_dir: Base directory (e.g., 'checkpoints')
        model_path: Model path from config (e.g., 'allsensor')
    
    Returns:
        Full path to the created directory
    """
    # Extract model name from path (remove directory and extension if present)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name with format: train_modelname_timestamp
    dir_name = f'train_{model_name}_{timestamp}'
    
    # Create full path
    full_path = os.path.join(base_dir, dir_name)
    
    # Create directory
    os.makedirs(full_path, exist_ok=True)
    
    return full_path


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

    # Setup training directory with model name
    save_dir = setup_training_directory_with_model_name(args.save_dir, config.task_name)
    print(f"Created training directory: {save_dir}")

    # Initialize visualizer
    visualizer = TrainingVisualizer(save_dir, use_tensorboard=args.use_tensorboard)

    # Save configuration
    config_manager.save_training_config(args, config, save_dir)

    # Load model
    print(f"\nLoading model from {config.model_path}")
    print(f"Using pretrained weights: {args.use_pretrained}")

    model_loader = ModelLoader()
    model, model_info = model_loader.load_pretrained_model(
        config.model_path,
        device,
        config,
        load_weights=args.use_pretrained
    )

    print("\nModel loaded successfully!")
    # print(f"Model architecture: {model}")

    # 根据配置选择划分模式
    split_mode = getattr(config, 'split_mode', 'random')
    print(f"\nUsing split mode: {split_mode}")

    # Prepare data
    train_loader, val_loader = prepare_data(config, args, device)

    # Initialize optimizer, scheduler, and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = RMSELoss()

    # Training loop
    best_val_loss = float('inf')
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}, Gradient clipping: {args.clip_grad}")
    print(f"Results saved to: {save_dir}")
    print("-" * 50)

    try:
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer,
                device, config, epoch, args.epochs,
                visualizer=visualizer,
                clip_grad_norm=args.clip_grad
            )

            # Validate
            val_loss = validate_epoch(
                model, val_loader, criterion,
                device, config, epoch, args.epochs
            )

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch}/{args.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.2e}")

            # Log metrics
            visualizer.log_epoch(epoch, train_loss, val_loss, current_lr)

            # Update learning rate
            scheduler.step(val_loss)

            # Update plots periodically
            if epoch % args.plot_interval == 0:
                plot_path = visualizer.plot_training_curves()
                print(f"  Plots updated: {plot_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, 'best_model.tar')
                model_loader.save_checkpoint(
                    model, optimizer, epoch, val_loss, save_path, model_info
                )
                print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f})")

            # Save periodic checkpoint
            if epoch % args.save_interval == 0:
                save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.tar')
                model_loader.save_checkpoint(
                    model, optimizer, epoch, val_loss, save_path, model_info
                )
                print(f"  Checkpoint saved: {save_path}")

            print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")

    finally:
        # Save final model
        save_path = os.path.join(save_dir, 'final_model.tar')
        model_loader.save_checkpoint(
            model, optimizer, epoch, val_loss, save_path, model_info
        )
        print(f"\nFinal model saved to {save_path}")

        # Generate final plots
        final_plot_path = visualizer.plot_training_curves()
        print(f"Final training curves saved to: {final_plot_path}")

        # Save metrics
        visualizer.save_metrics()

        # Close visualizer
        visualizer.close()

        # Print summary
        if visualizer.metrics['val_loss']:
            best_epoch = visualizer.metrics['epochs'][np.argmin(visualizer.metrics['val_loss'])]
            best_loss = min(visualizer.metrics['val_loss'])
            print(f"\nTraining Summary:")
            print(f"  Best validation loss: {best_loss:.4f} at epoch {best_epoch}")
            print(f"  Final train loss: {visualizer.metrics['train_loss'][-1]:.4f}")
            print(f"  Final val loss: {visualizer.metrics['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
