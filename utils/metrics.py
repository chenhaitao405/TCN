"""
Unified metrics computation and loss functions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class RMSELoss(nn.Module):
    """RMSE Loss function"""

    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((predictions - targets) ** 2))


class MetricsComputer:
    """Handle metrics computation for training and validation."""

    @staticmethod
    def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Root Mean Square Error.

        Args:
            predictions: Model predictions
            targets: Ground truth labels

        Returns:
            RMSE value
        """
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    @staticmethod
    def compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute R-squared (coefficient of determination).

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            R² score
        """
        if len(y_true) == 0:
            return float('nan')

        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

        # Handle edge case where ss_tot is 0
        if ss_tot == 0:
            return float('nan')

        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    @staticmethod
    def process_batch(
            estimates: torch.Tensor,
            labels: torch.Tensor,
            model_history: int,
            sequence_lengths: List[int],
            model_delays: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch to handle padding, delays, and invalid values.

        Args:
            estimates: Model estimates [batch_size, num_outputs, sequence_length]
            labels: Ground truth labels [batch_size, num_outputs, sequence_length]
            model_history: Model's effective history length
            sequence_lengths: Actual sequence lengths for each sample
            model_delays: Delays for each output channel

        Returns:
            Processed estimates and labels
        """
        batch_size, num_outputs, _ = estimates.shape
        processed_estimates = []
        processed_labels = []

        for i in range(batch_size):
            for j in range(num_outputs):
                # Extract valid sequence (ignore padding)
                estimate = estimates[i, j, model_history:sequence_lengths[i]]
                label = labels[i, j, model_history:sequence_lengths[i]]

                # Correct for intentional delays
                if model_delays[j] != 0:
                    if model_delays[j] > 0:
                        estimate = estimate[model_delays[j]:]
                        label = label[:-model_delays[j]]
                    else:
                        estimate = estimate[:model_delays[j]]
                        label = label[-model_delays[j]:]

                # Filter out NaN values
                valid_mask = ~torch.isnan(estimate) & ~torch.isnan(label)
                if valid_mask.any():
                    processed_estimates.append(estimate[valid_mask])
                    processed_labels.append(label[valid_mask])

        # Concatenate all valid samples
        if processed_estimates:
            return torch.cat(processed_estimates), torch.cat(processed_labels)
        else:
            return torch.tensor([]), torch.tensor([])


class ValidationMetrics:
    """Class to compute and store validation metrics."""

    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.metrics = {
            'per_label': {name: {'rmse': [], 'r2': [], 'estimates': [], 'labels': []}
                          for name in label_names},
            'overall': {'rmse': None, 'r2': None}
        }
        self.computer = MetricsComputer()

    def add_batch_results(
            self,
            estimates: Dict[int, Optional[torch.Tensor]],
            labels: Dict[int, Optional[torch.Tensor]],
            rmse_values: List[float],
            r2_values: List[float]
    ) -> None:
        """Store batch results for later aggregation."""
        for i, name in enumerate(self.label_names):
            # Check if i is within bounds and rmse is not NaN
            if i < len(rmse_values) and not np.isnan(rmse_values[i]):
                self.metrics['per_label'][name]['rmse'].append(rmse_values[i])
                self.metrics['per_label'][name]['r2'].append(r2_values[i])
                if i in estimates and estimates[i] is not None:
                    self.metrics['per_label'][name]['estimates'].extend(estimates[i].cpu().numpy())
                if i in labels and labels[i] is not None:
                    self.metrics['per_label'][name]['labels'].extend(labels[i].cpu().numpy())

    def compute_overall_metrics(self) -> None:
        """Compute overall metrics across all labels."""
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

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
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