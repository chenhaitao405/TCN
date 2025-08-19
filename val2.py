import argparse
from typing import List
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from config_utils import load_config
from tcn import TCN
from dataloader import TcnDataset
from utils import collate_function, process_batch, compute_rmse


class ModelEvaluator:
    """评估模型性能并可视化结果"""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.results = {
            'rmse': [],
            'r2': [],
            'predictions': [],
            'labels': [],
            'trial_names': []
        }

    def load_model(self):
        """加载预训练模型"""
        import inspect
        model_info = torch.load(self.config.model_path, map_location=self.device)
        state_dict = model_info["state_dict"]

        # 获取TCN类的初始化参数
        tcn_signature = inspect.signature(TCN.__init__)
        tcn_param_names = [param.name for param in tcn_signature.parameters.values()
                           if param.name != 'self']

        # 只传递TCN需要的参数
        tcn_params = {k: v for k, v in model_info.items()
                      if k in tcn_param_names}

        tcn = TCN(**tcn_params).to(self.device)
        tcn.load_state_dict(state_dict)
        tcn.eval()  # 设置为评估模式
        return tcn

    def print_detailed_results(self, trial_names: List[str],
                               label_names: List[str],
                               estimates: torch.FloatTensor,
                               labels: torch.FloatTensor,
                               model_history: int,
                               trial_sequence_lengths: List[int]):
        """打印每个试验和标签的详细RMSE结果"""
        print("\n" + "=" * 80)
        print("DETAILED EVALUATION RESULTS BY TRIAL AND LABEL")
        print("=" * 80)

        overall_rmse_by_label = {label_name: [] for label_name in label_names}

        for i, trial_name in enumerate(trial_names):
            print(f"\n{trial_name} results:")
            print("-" * 50)

            for j, label_name in enumerate(label_names):
                # 提取估计值和标签，忽略使用零填充的开始或结束序列
                estimate = estimates[i, j, model_history:trial_sequence_lengths[i]]
                label = labels[i, j, model_history:trial_sequence_lengths[i]]

                # 校正模型估计中的任何故意延迟
                if self.config.model_delays[j] != 0:
                    if self.config.model_delays[j] > 0:
                        estimate = estimate[self.config.model_delays[j]:]
                        label = label[:-self.config.model_delays[j]]
                    else:
                        # 负延迟的情况
                        estimate = estimate[:self.config.model_delays[j]]
                        label = label[-self.config.model_delays[j]:]

                # 忽略输入或标签数据中对应于nan的数据点
                valid_mask = ~torch.isnan(estimate) & ~torch.isnan(label)
                valid_estimate = estimate[valid_mask]
                valid_label = label[valid_mask]

                # 计算并打印RMSE
                if len(valid_estimate) > 0:
                    rmse = torch.sqrt(torch.mean((valid_estimate - valid_label) ** 2))
                    print(f"  {label_name} RMSE: {rmse:.4f} Nm/kg (valid points: {len(valid_estimate)})")
                    overall_rmse_by_label[label_name].append(rmse.item())
                else:
                    print(f"  {label_name} RMSE: N/A (no valid points)")

        # 打印每个标签的总体统计
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS BY LABEL")
        print("=" * 80)
        for label_name, rmse_list in overall_rmse_by_label.items():
            if rmse_list:
                mean_rmse = np.mean(rmse_list)
                std_rmse = np.std(rmse_list)
                print(f"{label_name}:")
                print(f"  Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} Nm/kg")
                print(f"  Min RMSE:  {np.min(rmse_list):.4f} Nm/kg")
                print(f"  Max RMSE:  {np.max(rmse_list):.4f} Nm/kg")
            else:
                print(f"{label_name}: No valid data")

    def evaluate_dataset_detailed(self, model, datasets, label_names):
        """详细评估整个数据集，按试验和标签分别计算"""
        print("Collecting all trials for detailed evaluation...")

        all_estimates = []
        all_labels = []
        all_trial_names = []
        all_sequence_lengths = []

        # 收集所有数据集的试验
        for dataset in datasets:
            trial_names = dataset.get_trial_names()

            for trial_idx in tqdm(range(len(dataset)), desc=f"Processing trials"):
                inputs, labels, seq_length = dataset[trial_idx]

                # 添加批次维度
                inputs = inputs.unsqueeze(0).to(self.device)
                labels = labels.unsqueeze(0).to(self.device)
                seq_lengths = [seq_length]

                with torch.no_grad():
                    # 前向传播
                    outputs = model(inputs)

                # 存储结果
                all_estimates.append(outputs.squeeze(0))  # 移除批次维度
                all_labels.append(labels.squeeze(0))  # 移除批次维度
                all_trial_names.append(trial_names[trial_idx])
                all_sequence_lengths.append(seq_length)

        # 转换为张量
        estimates_tensor = torch.stack(all_estimates)  # [num_trials, num_labels, seq_len]
        labels_tensor = torch.stack(all_labels)  # [num_trials, num_labels, seq_len]

        # 打印详细结果
        self.print_detailed_results(
            trial_names=all_trial_names,
            label_names=label_names,
            estimates=estimates_tensor,
            labels=labels_tensor,
            model_history=model.get_effective_history(),
            trial_sequence_lengths=all_sequence_lengths
        )

        return {
            'estimates': estimates_tensor,
            'labels': labels_tensor,
            'trial_names': all_trial_names,
            'sequence_lengths': all_sequence_lengths
        }

    def evaluate_batch(self, model, inputs, labels, seq_lengths):
        """评估单个批次（保持原有功能用于快速评估）"""
        with torch.no_grad():
            # 前向传播
            outputs = model(inputs)

            # 处理批次数据（处理padding和delays）
            processed_outputs, processed_labels = process_batch(
                outputs, labels,
                model.get_effective_history(),
                seq_lengths,
                self.config.model_delays
            )

            # 计算RMSE
            if processed_outputs.numel() > 0:
                rmse = compute_rmse(processed_outputs, processed_labels)

                # 计算R²
                ss_res = torch.sum((processed_outputs - processed_labels) ** 2)
                ss_tot = torch.sum((processed_labels - torch.mean(processed_labels)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                return {
                    'rmse': rmse.item(),
                    'r2': r2.item() if isinstance(r2, torch.Tensor) else r2,
                    'outputs': processed_outputs.cpu().numpy(),
                    'labels': processed_labels.cpu().numpy()
                }
        return None

    def evaluate_dataset(self, model, dataloader):
        """评估整个数据集（保持原有功能用于快速评估）"""
        all_rmse = []
        all_r2 = []
        all_outputs = []
        all_labels = []

        print("Evaluating model...")
        for batch_idx, (inputs, labels, seq_lengths) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 评估批次
            batch_results = self.evaluate_batch(model, inputs, labels, seq_lengths)

            if batch_results:
                all_rmse.append(batch_results['rmse'])
                all_r2.append(batch_results['r2'])
                all_outputs.append(batch_results['outputs'])
                all_labels.append(batch_results['labels'])

        # 计算平均指标
        avg_rmse = np.mean(all_rmse) if all_rmse else 0
        avg_r2 = np.mean(all_r2) if all_r2 else 0

        # 合并所有预测和标签
        if all_outputs:
            all_outputs = np.concatenate(all_outputs)
            all_labels = np.concatenate(all_labels)

        return {
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2,
            'all_rmse': all_rmse,
            'all_r2': all_r2,
            'outputs': all_outputs,
            'labels': all_labels
        }


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs.val.allsensor_config.py",
                        help="File path to config file for loading and testing pretrained TCN model.")
    parser.add_argument("--device", type=str, default="cpu" if torch.cuda.is_available() else "cpu",
                        help="Device to host model and data.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--save_path", type=str, default="evaluation_results.png",
                        help="Path to save visualization results")
    parser.add_argument("--detailed", action="store_true",
                        help="Run detailed evaluation by trial and label (slower but more comprehensive)")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config_path)
    device = torch.device(args.device)

    # 创建评估器
    evaluator = ModelEvaluator(config, device)

    # 加载模型
    print("Loading model...")
    model = evaluator.load_model()
    print(f"Model loaded successfully from {config.model_path}")

    # 准备数据
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

    print(f"Total trials across all datasets: {sum(len(dataset) for dataset in datasets)}")

    if args.detailed:
        # 详细评估 - 按试验和标签分别评估
        print("\n" + "=" * 50)
        print("RUNNING DETAILED EVALUATION")
        print("=" * 50)
        detailed_results = evaluator.evaluate_dataset_detailed(model, datasets, label_names)

    else:
        # 快速评估 - 批次处理
        print("\n" + "=" * 50)
        print("RUNNING BATCH EVALUATION")
        print("=" * 50)

        # 合并所有数据集
        full_dataset = ConcatDataset(datasets)

        # 创建DataLoader
        dataloader = DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_function(x, device)
        )

        # 评估模型
        results = evaluator.evaluate_dataset(model, dataloader)

        # 打印结果
        print("\n" + "=" * 50)
        print("BATCH EVALUATION RESULTS:")
        print("=" * 50)
        print(f"Average RMSE: {results['avg_rmse']:.4f} Nm/kg")
        print(f"Average R²: {results['avg_r2']:.4f}")
        print(f"Total batches evaluated: {len(results['all_rmse'])}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()