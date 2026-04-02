import argparse
import torch
import numpy as np
import sys
sys.path.append(".")
# Import custom modules
from utils.config_utils import ConfigManager
from utils.model_loader import ModelLoader
from rknn.api import RKNN

OUT_RKNN_PATH = "./deploy/trained_quantcn_8_sensors.rknn"
TEST_ON_DEVICE = True

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
    
    # original model output
    data_input = np.load("deploy/datasets/normal_walk_1-2_7.npy").astype(np.float32).reshape(1,8,280)
    data_input = torch.from_numpy(data_input).to(device).float()
    model.eval()
    with torch.no_grad():
        output = model(data_input)
    print(output)
    
    # quantize model output
    data_input = np.load("deploy/datasets/normal_walk_1-2_7.npy").astype(np.float32)
    data_input = data_input - np.array([ 0.7113, -0.4300, 0.7254,
        2.6529, 8.7143, -0.2818,
        -30.6663, -0.1348 ]).reshape(1,-1,1,1)
    data_input = data_input / np.array([21.3148, 45.2281, 81.4140, 
        3.9808, 4.4334, 1.8336,  
        27.8284, 107.1712    ])
    data_input = np.round(data_input / 0.079802 + 10).astype(np.int8)
    data_input_nchwc = np.empty((1, 1, 1, 280, 16), dtype=np.int8)
    
    
    rknn = RKNN()
    rknn.load_rknn(OUT_RKNN_PATH)

    if TEST_ON_DEVICE:
        ret = rknn.init_runtime(target="rv1106")
    else:
        ret = rknn.init_runtime(target=None)
    
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    outputs = rknn.inference(inputs=[data_input_nchwc], data_format="nc1hwc2", inputs_pass_through=[1])
    
    # 1,C,T
    # rknn.accuracy_analysis(inputs=[data_input], target=None)
    rknn.release()
    
    
if __name__ == "__main__":
    main()