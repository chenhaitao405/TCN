import argparse
import torch
import numpy as np
import sys
sys.path.append(".")
# Import custom modules
from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader
from rknn.api import RKNN

def create_argument_parser():
    """Create and configure argument parser for training."""
    parser = argparse.ArgumentParser(description='Train TCN for joint moment estimation')
    parser.add_argument('--config_path', type=str, default='configs.default_config.py',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training')
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                        help='Whether to use pretrained weights (default: False)')
    return parser

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_path)
    config = config_manager.apply_sensor_selection(config)
    
    print(f"\nLoading model from {config.model_path}")
    print(f"Using pretrained weights: {args.use_pretrained}")
    
    model_loader = ModelLoader()
    model, model_info = model_loader.load_model(
        config.model_path,
        device,
        config,
        load_weights=args.use_pretrained
    )
    model = model.to(device)
    data_input = np.load("deploy/datasets/normal_walk_2-5_1.npy").astype(np.float32)
    data_input = torch.from_numpy(data_input).to(device)
    model.eval()
    with torch.no_grad():
        output = model(data_input)
    print(output)
    
if __name__ == "__main__":
    main()