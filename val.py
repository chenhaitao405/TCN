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

    def evaluate_batch(self, model, inputs, labels, seq_lengths):
        """评估单个批次"""
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
            #TODO:按dataset.get_trial_names()和label_names进行评估

            # def print_results(trial_names: List[str],
            #                   label_names: List[str],
            #                   estimates: torch.FloatTensor,
            #                   labels: torch.FloatTensor,
            #                   model_history: int,
            #                   trial_sequence_lengths: List[float]):
            #     '''Prints model RMSE relative to ground-truth.'''
            #     for i, trial_name in enumerate(trial_names):
            #         print(f"{trial_name} results:")
            #         for j, label_name in enumerate(label_names):
            #             # Extract estimates and labels. Ignore andy starting or ending sequences that used zero padding.
            #             estimate = estimates[i, j, model_history:trial_sequence_lengths[i]]
            #             label = labels[i, j, model_history:trial_sequence_lengths[i]]
            #
            #             # Correct for any intentional delays in model estimates
            #             if config.model_delays[j] != 0:
            #                 estimate = estimate[config.model_delays[j]:]
            #                 label = label[:-config.model_delays[j]]
            #
            #             # Ignore data points corresponding to nans in input or label data
            #             valid_index = torch.where(~torch.isnan(estimate) & ~torch.isnan(label))
            #             estimate = estimate[valid_index]
            #             label = label[valid_index]
            #
            #             # Compute and print RMSE
            #             rmse = torch.sqrt(torch.mean((estimate - label) ** 2))
            #             print(f"{label_name} RMSE: {rmse} Nm/kg")
        return None

    def evaluate_dataset(self, model, dataloader):
        """评估整个数据集"""
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

    # 合并所有数据集
    full_dataset = ConcatDataset(datasets)
    print(f"Total dataset size: {len(full_dataset)} trials")


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
    print("Evaluation Results:")
    print("=" * 50)
    print(f"Average RMSE: {results['avg_rmse']:.4f} Nm/kg")
    print(f"Average R²: {results['avg_r2']:.4f}")
    print(f"Total batches evaluated: {len(results['all_rmse'])}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()