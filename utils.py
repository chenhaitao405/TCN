import torch
from typing import List, Tuple
from tcn import TCN


def load_model_architecture(model_path: str, device: torch.device) -> TCN:
    """
    Load TCN model architecture from saved model file.

    Args:
        model_path: Path to the saved model file
        device: Device to load the model on

    Returns:
        TCN model with architecture matching the saved model
    """
    model_info = torch.load(model_path, map_location=device)
    if "state_dict" in model_info:
        del model_info["state_dict"]
    tcn = TCN(**model_info).to(device)
    return tcn


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
                estimate = estimate[model_delays[j]:]
                label = label[:-model_delays[j]]

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


def save_checkpoint(
        model: TCN,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        save_path: str,
        model_info: dict
):
    """
    Save model checkpoint in tar format.

    Args:
        model: TCN model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save the checkpoint
        model_info: Model architecture information
    """
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **model_info  # Include all model architecture parameters
    }
    torch.save(checkpoint, save_path)

# 在train.py中添加自定义collate函数
def collate_function(batch, device):
    inputs = [item[0] for item in batch]  # 原始形状可能为 [1, 25, seq_len]
    labels = [item[1] for item in batch]
    seq_lengths = [item[2][0] for item in batch]  # 提取原始序列长度

    # 关键：删除多余的维度（通常是第0维的长度1）
    # 注意：根据实际数据维度调整squeeze的参数，确保只删除长度为1的维度
    inputs = [x.squeeze(0) for x in inputs]  # 变为 [25, seq_len]
    labels = [x.squeeze(0) for x in labels]  # 变为 [num_labels, seq_len]

    # 计算批次内最大序列长度
    max_seq_len = max([x.shape[-1] for x in inputs])

    # 填充到相同长度
    padded_inputs = []
    padded_labels = []
    for inp, lab in zip(inputs, labels):
        pad_length = max_seq_len - inp.shape[-1]
        padded_inp = torch.nn.functional.pad(inp, (0, pad_length), mode='constant', value=0)
        padded_lab = torch.nn.functional.pad(lab, (0, pad_length), mode='constant', value=0)
        padded_inputs.append(padded_inp)
        padded_labels.append(padded_lab)

    # 堆叠后形状为 [32, 25, max_seq_len]
    return torch.stack(padded_inputs), torch.stack(padded_labels), seq_lengths

