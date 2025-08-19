import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset

from config_utils import load_config
from dataloader import TcnDataset
from tcn import TCN
from utils import (
    compute_rmse,
    process_batch,
    save_checkpoint,
    collate_function
)


class RMSELoss(nn.Module):
    """RMSE Loss function"""

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))


class TrainingVisualizer:
    """Handle training visualization and logging"""

    def __init__(self, save_dir, use_tensorboard=True):
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
            'batch_losses': []  # Store all batch losses for distribution plot
        }

    def log_epoch(self, epoch, train_loss, val_loss, lr):
        """Log metrics for an epoch"""
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

    def log_batch(self, epoch, batch_idx, loss, total_batches):
        """Log batch-level metrics"""
        if self.writer:
            global_step = epoch * total_batches + batch_idx
            self.writer.add_scalar('Batch_Loss/Train', loss, global_step)
        self.metrics['batch_losses'].append(loss)

    def log_model_weights(self, model, epoch):
        """Log model weight distributions"""
        if self.writer:
            for name, param in model.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def plot_training_curves(self, save_path=None):
        """Create and save training curves"""
        if len(self.metrics['epochs']) == 0:
            return

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

        # Plot 4: Recent batch losses distribution (last epoch)
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
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.save_dir, 'training_metrics.json')
        # Convert to serializable format
        save_metrics = {
            'epochs': self.metrics['epochs'],
            'train_loss': self.metrics['train_loss'],
            'val_loss': self.metrics['val_loss'],
            'learning_rate': self.metrics['learning_rate'],
            'best_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None,
            'best_epoch': self.metrics['epochs'][np.argmin(self.metrics['val_loss'])] if self.metrics[
                'val_loss'] else None
        }
        with open(metrics_file, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

    def close(self):
        """Close tensorboard writer"""
        if self.writer:
            self.writer.close()


def load_pretrained_model(model_path: str, device: torch.device, load_weights: bool = True) -> tuple:
    """
    Load TCN model with optional pretrained weights.

    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        load_weights: Whether to load pretrained weights

    Returns:
        Tuple of (model, model_info_dict)
    """
    model_info = torch.load(model_path, map_location=device)
    state_dict = model_info.get("state_dict", None)

    # Get TCN initialization parameters
    tcn_signature = inspect.signature(TCN.__init__)
    tcn_param_names = [param.name for param in tcn_signature.parameters.values()
                       if param.name != 'self']

    # Only pass parameters that TCN needs
    tcn_params = {k: v for k, v in model_info.items()
                  if k in tcn_param_names}

    #修改为根据config构建模型

    # Create model
    tcn = TCN(**tcn_params).to(device)

    # Load pretrained weights if requested and available
    if load_weights and state_dict is not None:
        tcn.load_state_dict(state_dict)
        print("Loaded pretrained weights successfully!")
    else:
        print("Using random initialization for model weights.")

    # Prepare model info for saving (exclude state_dict and training info)
    save_info = {k: v for k, v in model_info.items()
                 if k not in ["state_dict", "optimizer_state_dict", "epoch", "loss"]}

    return tcn, save_info


def train_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        model_delays,
        epoch,
        total_epochs,
        visualizer=None,
        clip_grad_norm=1.0
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    nan_count = 0
    batch_losses = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    total_batches = len(dataloader)

    for batch_idx, (inputs, labels, seq_lengths) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"Warning: NaN detected in model output at batch {batch_idx}")
            nan_count += 1
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"  NaN found in parameter: {name}")
            continue

        # Process batch to handle padding and delays
        processed_outputs, processed_labels = process_batch(
            outputs, labels,
            model.get_effective_history(),
            seq_lengths,
            model_delays
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

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        # Update metrics
        loss_value = loss.item()
        total_loss += loss_value
        batch_losses.append(loss_value)
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
    if visualizer and epoch % 5 == 0:  # Log every 5 epochs to save space
        visualizer.log_model_weights(model, epoch)

    return total_loss / num_batches if num_batches > 0 else 0


def validate_epoch(
        model,
        dataloader,
        criterion,
        device,
        model_delays,
        epoch,
        total_epochs
):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_losses = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Valid]')
        for inputs, labels, seq_lengths in pbar:
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
            processed_outputs, processed_labels = process_batch(
                outputs, labels,
                model.get_effective_history(),
                seq_lengths,
                model_delays
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


def main():
    # Parse arguments
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
                        help='Whether to use pretrained weights (default: True)')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use tensorboard for visualization')
    parser.add_argument('--plot_interval', type=int, default=5,
                        help='Update plots every N epochs')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    device = torch.device(args.device)

    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'train_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize visualizer
    visualizer = TrainingVisualizer(save_dir, use_tensorboard=args.use_tensorboard)

    # Save training arguments
    args_file = os.path.join(save_dir, 'training_args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training arguments saved to: {args_file}")

    # Load model with pretrained weights
    print(f"Loading model from {config.model_path}")
    print(f"Using pretrained weights: {args.use_pretrained}")
    model, model_info = load_pretrained_model(
        config.model_path,
        device,
        load_weights=args.use_pretrained
    )

    # Verify model weights are not NaN
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"ERROR: NaN found in initial parameter: {name}")
            print("Please check the pretrained model file!")
            return

    # Prepare data
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    print("Loading dataset...")
    # 创建一个列表来存储所有数据集
    datasets = []

    # 循环读取每个路径的数据
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

    # 合并所有数据集
    full_dataset = ConcatDataset(datasets)
    print(f"Total dataset size: {len(full_dataset)} trials")

    # Split dataset
    ## 过滤含nan的数据
    print("Filtering trials with NaN...")
    valid_indices = []
    for i in tqdm(range(len(full_dataset)), desc="Checking trials"):
        inputs, labels, seq_lengths = full_dataset[i]
        # 只检查输入和标签的原始NaN，不是模型产生的
        if not torch.isnan(inputs).any() and not torch.isnan(labels).any():
            valid_indices.append(i)

    print(
        f"Valid trials: {len(valid_indices)}/{len(full_dataset)} ({100 * len(valid_indices) / len(full_dataset):.1f}%)")

    # 使用Subset只包含有效试验
    from torch.utils.data import Subset
    filtered_dataset = Subset(full_dataset, valid_indices)

    # 然后对filtered_dataset进行train/val split
    val_size = int(len(filtered_dataset) * args.val_split)
    train_size = len(filtered_dataset) - val_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    print(f"Dataset split: {train_size} train, {val_size} validation")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_function(x, device)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_function(x, device)
    )

    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
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
                device, config.model_delays, epoch, args.epochs,
                visualizer=visualizer,
                clip_grad_norm=args.clip_grad
            )

            # Validate
            val_loss = validate_epoch(
                model, val_loader, criterion,
                device, config.model_delays, epoch, args.epochs
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
                save_checkpoint(model, optimizer, epoch, val_loss, save_path, model_info)
                print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f})")

            # Save periodic checkpoint
            if epoch % args.save_interval == 0:
                save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.tar')
                save_checkpoint(model, optimizer, epoch, val_loss, save_path, model_info)
                print(f"  Checkpoint saved: {save_path}")

            print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")

    finally:
        # Save final model
        save_path = os.path.join(save_dir, 'final_model.tar')
        save_checkpoint(model, optimizer, epoch, val_loss, save_path, model_info)
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