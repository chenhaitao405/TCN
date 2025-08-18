import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config_utils import load_config
from dataloader import TcnDataset
from utils import (
    load_model_architecture,
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


def train_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        model_delays,
        epoch,
        total_epochs
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    for batch_idx, (inputs, labels, seq_lengths) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

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

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Valid]')
        for inputs, labels, seq_lengths in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

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

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches if num_batches > 0 else 0


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
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    device = torch.device(args.device)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize model with same architecture as saved model
    print(f"Loading model architecture from {config.model_path}")
    model = load_model_architecture(config.model_path, device)

    # Get model info for saving
    model_info = torch.load(config.model_path, map_location=device)
    if "state_dict" in model_info:
        del model_info["state_dict"]
    if "optimizer_state_dict" in model_info:
        del model_info["optimizer_state_dict"]
    if "epoch" in model_info:
        del model_info["epoch"]
    if "loss" in model_info:
        del model_info["loss"]

    # Prepare data
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    print("Loading dataset...")
    full_dataset = TcnDataset(
        data_dir=config.data_dir,
        input_names=input_names,
        label_names=label_names,
        side=config.side,
        participant_masses=config.participant_masses,
        device=device
    )

    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset split: {train_size} train, {val_size} validation")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_function(x, device)  # 使用自定义函数
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_function(x, device)
    )

    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RMSELoss()

    # Training loop
    best_val_loss = float('inf')
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, config.model_delays, epoch, args.epochs
        )

        # Validate
        val_loss = validate_epoch(
            model, val_loader, criterion,
            device, config.model_delays, epoch, args.epochs
        )

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, f'best_model_{timestamp}.tar')
            save_checkpoint(model, optimizer, epoch, val_loss, save_path, model_info)
            print(f"Saved best model to {save_path}")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}_{timestamp}.tar')
            save_checkpoint(model, optimizer, epoch, val_loss, save_path, model_info)
            print(f"Saved checkpoint to {save_path}")

    # Save final model
    save_path = os.path.join(args.save_dir, f'final_model_{timestamp}.tar')
    save_checkpoint(model, optimizer, args.epochs, val_loss, save_path, model_info)
    print(f"\nTraining complete! Final model saved to {save_path}")


if __name__ == "__main__":
    main()