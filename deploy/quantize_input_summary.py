max_limits = [29.639290, 44.000000, 92.000000, 9.669252, 15.112404, 1.874497, 1.378986, 126.312927]
min_limits = [-25.241596, -44.000000, -92.000000, -2.028937, 1.110079, -3.041151, -89.030258, -118.253586]

bits = 8
quan_max = 2 ** 8 - 1
quan_min = 0

import torch

scales = (torch.tensor(max_limits) - torch.tensor(min_limits)) / (quan_max - quan_min)
zeros = torch.clamp(torch.round(quan_min - torch.tensor(min_limits) / scales), min=quan_min, max=quan_max)

print(scales, '\n', zeros)

# 求量化后的归一化数据
import argparse

from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(".")

# Import custom modules
from utils.config_utils import ConfigManager

from utils.data_loader import DataManager
from utils.metrics import MetricsComputer



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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')

    return parser


def prepare_data(config, args, device):

    """
    Prepare data with sliding window support for training.

    Args:
        config: Configuration object containing data loading parameters
        args: Arguments containing batch_size and optional max_samples
        device: Device to load data onto

    Returns:
        tuple: (train_loader, val_loader) - DataLoaders for training and validation
    """

    def _print_configuration(config, split_mode, use_sliding_window):
        """Print data loading configuration."""
        print("=" * 50)
        print("Data Loading Configuration:")
        print(f"  Split Mode: {split_mode.capitalize()}")
        print(f"  Training Mode: {'Sliding Window' if use_sliding_window else 'Full Trial'}")

        if use_sliding_window:
            print(f"  Window Size: {config.window_size}")
            print(f"  Window Stride: {config.window_stride}")

        print("=" * 50)

    def _create_datasets(data_manager, config, split_mode, use_sliding_window, device, max_samples):
        """
        Create train and validation datasets based on split mode.

        Args:
            data_manager: DataManager instance
            config: Configuration object
            split_mode: 'manual' or 'random' split mode
            device: Device for data loading
            max_samples: Maximum number of samples to load

        Returns:
            tuple: (train_dataset, val_dataset)

        Raises:
            ValueError: If manual split mode is missing required configuration
        """
        if split_mode == 'manual':
            # Validate manual split configuration
            if not hasattr(config, 'train_data_dirs') or not hasattr(config, 'val_dataset'):
                raise ValueError(
                    "Manual split mode requires 'train_data_dirs' and 'val_dataset' in config"
                )

            return data_manager.create_manual_split(
                config=config,
                device=device,
                max_samples=max_samples,
            )

        else:  # random split
            return data_manager.create_random_train_val_split(
                config=config,
                device=device,
                max_samples=max_samples,
                use_sliding_window = use_sliding_window,
            )

    def _create_dataloaders(data_manager, train_dataset, val_dataset,
                            batch_size, device, use_sliding_window):
        """
        Create data loaders based on training mode.

        Args:
            data_manager: DataManager instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for data loading
            device: Device for data loading
            use_sliding_window: Whether to use sliding window mode

        Returns:
            tuple: (train_loader, val_loader)
        """
        if use_sliding_window:
            # Use sliding window data loaders
            return data_manager.create_dataloaders_windows(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                device=device
            )
        else:
            # Use full trial data loaders
            return data_manager.create_dataloaders_trail(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                device=device
            )

    def _print_split_results(split_mode, use_sliding_window,
                             train_dataset, val_dataset,
                             train_loader, val_loader):
        """Print the results of data splitting."""
        mode_name = "Manual" if split_mode == 'manual' else "Random"
        data_unit = "windows" if use_sliding_window else "trials"

        print(f"\n{mode_name} split completed:")
        print(f"  Training: {len(train_dataset)} {data_unit}, {len(train_loader)} batches")

        # Use consistent naming for validation/testing
        val_label = "Testing" if split_mode == 'manual' else "Validation"
        print(f"  {val_label}: {len(val_dataset)} {data_unit}, {len(val_loader)} batches")

    # Initialize components
    data_manager = DataManager()
    device_cpu = torch.device("cpu")

    # Extract configuration parameters
    use_sliding_window = getattr(config, 'use_sliding_window', False)
    split_mode = config.split_mode
    batch_size = args.batch_size
    max_samples = getattr(args, 'max_samples', None)

    # Print configuration header
    _print_configuration(config, split_mode, use_sliding_window)

    # Create datasets based on split mode
    train_dataset, val_dataset = _create_datasets(
        data_manager, config, split_mode, use_sliding_window, device_cpu, max_samples
    )

    # Create data loaders based on training mode
    train_loader, val_loader = _create_dataloaders(
        data_manager, train_dataset, val_dataset,
        batch_size, device_cpu, use_sliding_window
    )

    # Print split results
    _print_split_results(
        split_mode, use_sliding_window,
        train_dataset, val_dataset,
        train_loader, val_loader
    )

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



class Summary():
    def __init__(self, num_channels, channel_order, total_num, num_bins=100):
        """
        Enhanced Summary class with distribution tracking.
        
        Args:
            num_channels: Number of data channels
            channel_order: Which dimension represents channels
            total_num: Total number of data points for mean calculation
            num_bins: Number of bins for histogram (default: 100)
        """
        self.max = torch.tensor([-100000] * num_channels, dtype=torch.float64, device="cpu")
        self.min = torch.tensor([100000] * num_channels, dtype=torch.float64, device="cpu")
        self.mean = torch.tensor([0] * num_channels, dtype=torch.float64, device="cpu")
        self.std = torch.tensor([0] * num_channels, dtype=torch.float64, device="cpu")
        self.channel_order = channel_order
        self.total_num = total_num
        self.num_bins = num_bins
        
        # Distribution tracking
        self.histograms = None  # Will store histogram counts per channel
        self.bin_edges = None   # Will store bin edges per channel
        self.sample_count = 0   # Track number of samples processed
        
    def update_max(self, batch_data):
        batch_data = batch_data.to(torch.float64)
        max_dim = [i for i in range(len(batch_data.shape)) if i != self.channel_order]
        _max = batch_data
        for th, dim in enumerate(max_dim):
            _max = torch.max(_max, dim=dim-th)[0]
        self.max = torch.maximum(self.max, _max)

    def update_min(self, batch_data):
        batch_data = batch_data.to(torch.float64)
        min_dim = [i for i in range(len(batch_data.shape)) if i != self.channel_order]
        _min = batch_data
        for th, dim in enumerate(min_dim):
            _min = torch.min(_min, dim=dim-th)[0]
        self.min = torch.minimum(self.min, _min)
    
    def update_mean(self, batch_data: torch.Tensor):
        batch_data = batch_data.to(torch.float64)
        mean_dim = [i for i in range(len(batch_data.shape)) if i != self.channel_order]
        _mean_add = torch.sum(batch_data, dim=mean_dim) / self.total_num
        self.mean += _mean_add
    
    def update_std(self, batch_data: torch.Tensor):
        """
        Update standard deviation using Welford's online algorithm.
        Should be called after update_mean is finalized.
        """
        batch_data = batch_data.to(torch.float64)
        mean_dim = [i for i in range(len(batch_data.shape)) if i != self.channel_order]
        
        # Calculate squared differences from mean
        squared_diff = (batch_data - self.mean.view(*[1 if i != self.channel_order else -1 
                                                      for i in range(len(batch_data.shape))])) ** 2
        _std_add = torch.sum(squared_diff, dim=mean_dim) / self.total_num
        self.std += _std_add
    
    def initialize_histograms(self):
        """
        Initialize histogram bins based on current min/max values.
        Should be called after first pass through data to get min/max.
        """
        num_channels = len(self.max)
        self.histograms = torch.zeros((num_channels, self.num_bins), dtype=torch.float64)
        self.bin_edges = []
        
        for i in range(num_channels):
            # Create bin edges for this channel
            edges = torch.linspace(self.min[i].item(), self.max[i].item(), self.num_bins + 1)
            self.bin_edges.append(edges)
    
    def update_distribution(self, batch_data: torch.Tensor):
        """
        Update histogram counts for distribution tracking.
        
        Args:
            batch_data: Input batch tensor
        """
        if self.histograms is None:
            self.initialize_histograms()
        
        batch_data = batch_data.to(torch.float64).cpu()
        
        # Flatten all dimensions except channel dimension
        if self.channel_order == 1:  # (batch, channel, time)
            flat_data = batch_data.transpose(0, 1).reshape(len(self.max), -1)
        elif self.channel_order == 0:  # (channel, batch, time)
            flat_data = batch_data.reshape(len(self.max), -1)
        else:
            raise ValueError(f"Unsupported channel_order: {self.channel_order}")
        
        # Update histogram for each channel
        for ch_idx in range(len(self.max)):
            channel_data = flat_data[ch_idx]
            
            # Compute histogram using torch (faster than numpy for large data)
            hist = torch.histc(
                channel_data, 
                bins=self.num_bins,
                min=self.min[ch_idx].item(),
                max=self.max[ch_idx].item()
            )
            self.histograms[ch_idx] += hist
        
        self.sample_count += batch_data.numel() // len(self.max)
    
    def get_distribution_stats(self, channel_idx):
        """
        Get distribution statistics for a specific channel.
        
        Args:
            channel_idx: Index of the channel
            
        Returns:
            dict: Dictionary containing distribution statistics
        """
        if self.histograms is None:
            return None
        
        hist = self.histograms[channel_idx].numpy()
        bin_edges = self.bin_edges[channel_idx].numpy()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate percentiles
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            idx = np.searchsorted(cumsum, total * p / 100)
            percentiles[f'p{p}'] = bin_centers[min(idx, len(bin_centers)-1)]
        
        return {
            'mean': self.mean[channel_idx].item(),
            'std': np.sqrt(self.std[channel_idx].item()),
            'min': self.min[channel_idx].item(),
            'max': self.max[channel_idx].item(),
            'percentiles': percentiles,
            'histogram': hist,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges
        }
    
    def plot_distribution(self, channel_idx, channel_name=None, save_path=None, 
                         show_stats=True, figsize=(10, 6)):
        """
        Plot distribution histogram for a specific channel.
        
        Args:
            channel_idx: Index of the channel to plot
            channel_name: Name of the channel (optional)
            save_path: Path to save the figure (optional)
            show_stats: Whether to show statistics on the plot
            figsize: Figure size tuple
        """
        stats = self.get_distribution_stats(channel_idx)
        if stats is None:
            print("No distribution data available. Call update_distribution first.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.bar(stats['bin_centers'], stats['histogram'], 
               width=np.diff(stats['bin_edges'])[0],
               alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for mean and percentiles
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(stats['percentiles']['p50'], color='green', linestyle='--', 
                  linewidth=2, label='Median')
        ax.axvline(stats['percentiles']['p5'], color='orange', linestyle=':', 
                  linewidth=1.5, alpha=0.7, label='5th/95th percentile')
        ax.axvline(stats['percentiles']['p95'], color='orange', linestyle=':', 
                  linewidth=1.5, alpha=0.7)
        
        # Labels and title
        title = f"Distribution - Channel {channel_idx}"
        if channel_name:
            title += f" ({channel_name})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        if show_stats:
            stats_text = f"Mean: {stats['mean']:.4f}\n"
            stats_text += f"Std: {stats['std']:.4f}\n"
            stats_text += f"Min: {stats['min']:.4f}\n"
            stats_text += f"Max: {stats['max']:.4f}\n"
            stats_text += f"Median: {stats['percentiles']['p50']:.4f}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plot to {save_path}")
        
        plt.show()
    
    def plot_all_distributions(self, channel_names=None, save_dir=None, 
                              ncols=3, figsize_per_subplot=(5, 4)):
        """
        Plot distributions for all channels in a grid.
        
        Args:
            channel_names: List of channel names (optional)
            save_dir: Directory to save individual plots (optional)
            ncols: Number of columns in the grid
            figsize_per_subplot: Size of each subplot
        """
        if self.histograms is None:
            print("No distribution data available. Call update_distribution first.")
            return
        
        num_channels = len(self.max)
        nrows = (num_channels + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(figsize_per_subplot[0] * ncols, 
                                        figsize_per_subplot[1] * nrows))
        axes = axes.flatten() if num_channels > 1 else [axes]
        
        for ch_idx in range(num_channels):
            stats = self.get_distribution_stats(ch_idx)
            ax = axes[ch_idx]
            
            # Plot histogram
            ax.bar(stats['bin_centers'], stats['histogram'],
                  width=np.diff(stats['bin_edges'])[0],
                  alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add mean line
            ax.axvline(stats['mean'], color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7)
            
            # Title
            title = f"Ch {ch_idx}"
            if channel_names and ch_idx < len(channel_names):
                title += f"\n{channel_names[ch_idx]}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Save individual plot if directory provided
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"channel_{ch_idx}"
                individual_path = Path(save_dir) / f"dist_{name}.png"
                
                # Create individual figure
                fig_single, ax_single = plt.subplots(figsize=figsize_per_subplot)
                ax_single.bar(stats['bin_centers'], stats['histogram'],
                            width=np.diff(stats['bin_edges'])[0],
                            alpha=0.7, edgecolor='black', linewidth=0.5)
                ax_single.axvline(stats['mean'], color='red', linestyle='--', linewidth=2)
                ax_single.set_title(title, fontsize=12, fontweight='bold')
                ax_single.set_xlabel('Value', fontsize=10)
                ax_single.set_ylabel('Frequency', fontsize=10)
                ax_single.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(individual_path, dpi=300, bbox_inches='tight')
                plt.close(fig_single)
        
        # Hide unused subplots
        for idx in range(num_channels, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_statistics(self, save_path):
        """
        Save all statistics to a file.
        
        Args:
            save_path: Path to save the statistics
        """
        stats_dict = {
            'max': self.max.numpy(),
            'min': self.min.numpy(),
            'mean': self.mean.numpy(),
            'std': np.sqrt(self.std.numpy()),
            'sample_count': self.sample_count,
            'num_bins': self.num_bins,
        }
        
        if self.histograms is not None:
            stats_dict['histograms'] = self.histograms.numpy()
            stats_dict['bin_edges'] = [edges.numpy() for edges in self.bin_edges]
        
        np.savez(save_path, **stats_dict)
        print(f"Statistics saved to {save_path}")
    
    def print_summary(self, channel_names=None):
        """
        Print a summary of statistics for all channels.
        
        Args:
            channel_names: List of channel names (optional)
        """
        print("\n" + "="*80)
        print("DATA STATISTICS SUMMARY")
        print("="*80)
        
        for ch_idx in range(len(self.max)):
            name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"Channel {ch_idx}"
            print(f"\n{name}:")
            print(f"  Min:  {self.min[ch_idx].item():>12.6f}")
            print(f"  Max:  {self.max[ch_idx].item():>12.6f}")
            print(f"  Mean: {self.mean[ch_idx].item():>12.6f}")
            print(f"  Std:  {np.sqrt(self.std[ch_idx].item()):>12.6f}")
            
            if self.histograms is not None:
                stats = self.get_distribution_stats(ch_idx)
                print(f"  Median: {stats['percentiles']['p50']:>10.6f}")
                print(f"  P5-P95: [{stats['percentiles']['p5']:>10.6f}, {stats['percentiles']['p95']:>10.6f}]")
                print(f"  P10-P90: [{stats['percentiles']['p10']:>10.6f}, {stats['percentiles']['p90']:>10.6f}]")
        
        print("\n" + "="*80)

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

    # Prepare data
    train_loader, val_loader = prepare_data(config, args, device)
    num_channels = len(config.sensor_pick)
    data_summary = Summary(num_channels, 1, total_num=len(train_loader.dataset) * config.window_size)
    
    for data_batch in tqdm.tqdm(train_loader):
        quan_data = data_batch[0] / scales.view(1, num_channels, 1) + zeros.view(1, num_channels, 1)
        quan_data = torch.clamp(torch.round(quan_data), quan_min, quan_max)
        quan_data = quan_data / 255.0
        data_summary.update_max(quan_data)
        data_summary.update_min(quan_data)
        data_summary.update_mean(quan_data)
        
    for data_batch in tqdm.tqdm(train_loader):
        quan_data = data_batch[0] / scales.view(1, num_channels, 1) + zeros.view(1, num_channels, 1)
        quan_data = torch.clamp(torch.round(quan_data), quan_min, quan_max)
        quan_data = quan_data / 255.0
        data_summary.update_std(quan_data)
        data_summary.update_distribution(quan_data)
        
    data_summary.print_summary(channel_names=config.sensor_pick)

    data_summary.plot_all_distributions(
        channel_names=config.sensor_pick,
        save_dir='./distribution_plots'
    )
    
    

if __name__ == "__main__":
    main()
