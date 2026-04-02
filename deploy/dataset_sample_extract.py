import re
import argparse
import torch
import sys
import os  # 新增: 用于创建文件夹
sys.path.append(".")
from tqdm import tqdm
# Import custom modules
from utils.config_utils import ConfigManager
from utils.data_loader import DataManager
import numpy as np


class DatasetSampler:
    
    ACTION_PATTERNS = [
        r".*jump_.*",                   #1
        r".*incline_walk_.*",           #2
        r".*poses_.*",                  #3
        r".*stairs_.*",                 #4
        r".*normal_walk_.*_0-6.*",      #5
        r".*normal_walk_.*_1-2.*",      #6
        r".*normal_walk_.*_1-8.*",      #7
        r".*normal_walk_.*_2-0.*",      #8
        r".*normal_walk_.*_2-5.*",      #9
        r".*normal_walk_.*_shuffle.*",  #10
        r".*normal_walk_.*_skip.*",     #11
        r".*walk_backward_.*",          #12
        r".*weighted_walk_.*",          #13
        r".*cutting_.*",                #14
        r".*sit_to_stand_.*",           #15
        r".*start_stop_.*",             #16
        r".*turn_and_step_.*",          #17
        r".*squats_.*"                  #18
    ]
    
    ACTION_NUM = [
        27820, 39462, 32906, 8401, 8789, 10406, 8450, 8388, 9320, 
        5236, 4179, 26862, 8554, 1992, 104176, 16283, 7354, 13400
    ]
    
    SAMPLE_NUM = [
        5, #1
        10, #2 
        25, #3
        10, #4
        10,  #5
        10,  #6
        5,  #7
        0,  #8
        0,  #9
        4,  #10
        0,  #11
        15, #12
        0,  #13
        0,  #14
        15, #15
        5, #16
        5,  #17
        0   #18
    ]
    
    ACTION_NAMES = [
        "jump",
        "incline_walk",
        "poses",
        "stairs",
        "normal_walk_0-6",
        "normal_walk_1-2",
        "normal_walk_1-8",
        "normal_walk_2-0",
        "normal_walk_2-5",
        "normal_walk_shuffle",
        "normal_walk_skip",
        "walk_backward",
        "weighted_walk",
        "cutting",
        "sit_to_stand",
        "start_stop",
        "turn_and_step",
        "squats"
    ]

    def __init__(self, dataloader, save_dir, txt_path):
        self.dataloader = dataloader
        self.save_dir = save_dir
        
        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created save directory: {self.save_dir}")

        self.txt_path = txt_path
        self.txt_file = open(txt_path, "w")
        
        # 初始化计数器
        self.current_counts = {action: 0 for action in self.ACTION_PATTERNS}
        self.saved_counts = {action: 0 for action in self.ACTION_PATTERNS}
        
        # 计算采样间隔 (Interval) 和 下一个采样点 (Milestone)
        self.intervals = {}
        self.next_milestones = {}
        
        for i, action in enumerate(self.ACTION_PATTERNS):
            total = self.ACTION_NUM[i]
            sample = self.SAMPLE_NUM[i]
            
            if sample > 0:
                interval = int(total / sample)
                self.intervals[action] = interval
                self.next_milestones[action] = interval # 第一次采样发生在第 interval 个
            else:
                self.intervals[action] = float('inf')
                self.next_milestones[action] = float('inf')

    def save_sample(self, input_tensor: torch.Tensor, action_name: str, index):
        """保存单个样本到磁盘"""
        save_data: np.ndarray = input_tensor.numpy()
        filename = f"{action_name}_{index}.npy"
        save_path = os.path.join(self.save_dir, filename)
        save_data = np.expand_dims(save_data, 0)  # (1,C,T)
        np.save(save_path, np.expand_dims(save_data, -2).astype(np.float32))  # (1,C,1,T)
        self.txt_file.write(os.path.relpath(save_path, os.path.dirname(self.txt_path)) + '\n')
        

    def process_and_sample(self):
        print("Start processing and sampling...")
        
        for batch_idx, data_batch in tqdm(enumerate(self.dataloader)):
            # 假设 data_batch 结构为 (inputs, targets, weights/other, metadatas)
            # 根据你的代码: metadatas = data_batch[3]
            inputs = data_batch[0]
            
            metadatas = data_batch[3]
            trial_names = metadatas['trial_name']
            
            batch_size = len(trial_names)
            
            for i in range(batch_size):
                trial_name = trial_names[i]
                
                # 匹配动作类型
                matched_action = None
                action_name = None
                for k, action_pattern in enumerate(self.ACTION_PATTERNS):
                    if re.match(action_pattern, trial_name):
                        action_name = self.ACTION_NAMES[k]
                        matched_action = action_pattern
                        break
                
                if matched_action:
                    # 1. 增加当前动作的总计数
                    self.current_counts[matched_action] += 1
                    
                    # 2. 检查是否达到采样间隔
                    milestone = self.next_milestones[matched_action]
                    if not (milestone == float('inf')) and self.current_counts[matched_action] >= int(milestone):
                        
                        # 执行保存
                        self.save_sample(
                            inputs[i], 
                            action_name,  
                            self.saved_counts[matched_action]
                        )
                        
                        # 更新状态
                        self.saved_counts[matched_action] += 1
                        self.next_milestones[matched_action] += self.intervals[matched_action]
                        
        self.txt_file.close()

    def print_summary(self):
        print("\n" + "="*50)
        print("Sampling Summary:")
        print(f"{'Action Pattern':<40} | {'Total':<8} | {'Target':<8} | {'Saved':<8}")
        print("-" * 70)
        for i, action in enumerate(self.ACTION_PATTERNS):
            print(f"{action:<40} | {self.current_counts[action]:<8} | {self.SAMPLE_NUM[i]:<8} | {self.saved_counts[action]:<8}")
        print("="*50 + "\n")


def create_argument_parser():
    """Create and configure argument parser for training."""
    parser = argparse.ArgumentParser(description='Train TCN for joint moment estimation')
    parser.add_argument('--config_path', type=str, default='configs.default_config.py',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--save_path', type=str, help="save directory")
    parser.add_argument('--txt_path', type=str, default="deploy/quantize_dataset.txt")
    
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

    def _create_datasets(data_manager: DataManager, config, split_mode, use_sliding_window, device, max_samples):
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
    
    # 初始化采样器，保存到 'sampled_dataset' 文件夹
    sampler = DatasetSampler(train_loader, save_dir=args.save_path, txt_path=args.txt_path)
    
    # 开始处理
    sampler.process_and_sample()
    
    # 打印结果
    sampler.print_summary()

if __name__ == "__main__":
    main()
    
    
