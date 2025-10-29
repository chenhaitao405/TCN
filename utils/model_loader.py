"""
Unified model loading utilities for training and validation.
"""
import torch
import inspect
import numpy as np
from utils.tcn import TCN
from utils.TSForecasting.layers.ConvTimeNet_backbone import ConvTimeNet_backbone
from typing import Tuple, Dict, Any, Union

class ModelLoader:
    """Handle model loading with various configurations."""

    @staticmethod
    def load_TCN(
            model_path: str,
            device: torch.device,
            config: Any,
            load_weights: bool = True
    ) -> Tuple[TCN, Dict[str, Any]]:
        """
        Load TCN model with optional pretrained weights and sensor selection.

        Args:
            model_path: Path to the saved model file
            device: Device to load the model on
            config: Configuration object containing sensor_pick and label_names
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

        # Ensure center and scale are on the correct device (for both load_weights=True and False)


        # Modify parameters based on configuration when not loading weights
        if not load_weights:
            tcn_params = ModelLoader._adjust_model_params(tcn_params, config)

        if 'center' in tcn_params and tcn_params['center'] is not None:
            if torch.is_tensor(tcn_params['center']):
                tcn_params['center'] = tcn_params['center'].to(device)

        if 'scale' in tcn_params and tcn_params['scale'] is not None:
            if torch.is_tensor(tcn_params['scale']):
                tcn_params['scale'] = tcn_params['scale'].to(device)

        # Create model
        tcn = TCN(**tcn_params).to(device)

        # Load pretrained weights if requested and available
        if load_weights and state_dict is not None:
            tcn.load_state_dict(state_dict)
            print("Loaded pretrained weights successfully!")
        elif load_weights and state_dict is None:
            raise ValueError("No state_dict found in model file!")
        else:
            print("Using random initialization for model weights.")

        # Verify model weights are not NaN
        ModelLoader._verify_model_weights(tcn)

        # Prepare model info for saving (exclude state_dict and training info)
        save_info = tcn_params

        return tcn, save_info

    @staticmethod
    def load_ConvTimeNet(
            model_path: str,
            device: torch.device,
            config: Any,
            load_weights: bool = True
    ) -> Tuple[ConvTimeNet_backbone, Dict[str, Any]]:
        """
        Load ConvTimeNet model with optional pretrained weights and sensor selection.

        Args:
            model_path: Path to the saved model file
            device: Device to load the model on
            config: Configuration object containing sensor_pick and label_names
            load_weights: Whether to load pretrained weights

        Returns:
            Tuple of (model, model_info_dict)
        """
        model_info = torch.load(model_path, map_location=device)
        state_dict = model_info.get("state_dict", None)

        # Get center and scale from config or model_info
        center = None
        scale = None

        # Read center parameter
        if hasattr(config, 'center') and config.center is not None:
            center = config.center
            if isinstance(center, np.ndarray):
                center = torch.from_numpy(center).float()

            # Select sensors based on sensor_pick if provided
            if hasattr(config, 'sensor_pick') and config.sensor_pick:
                if center.dim() == 3 and center.shape[1] > len(config.sensor_pick):
                    center = center[:, config.sensor_pick, :]

            # Move to device
            center = center.to(device)
            print(f"✓ 从 config 读取 center 参数")
        elif 'center' in model_info and model_info['center'] is not None:
            center = model_info['center']
            if torch.is_tensor(center):
                center = center.to(device)
            if hasattr(config, 'sensor_pick') and config.sensor_pick:
                if center.dim() == 3 and center.shape[1] > len(config.sensor_pick):
                    center = center[:, config.sensor_pick, :]
            print(f"⚠ 从模型参数中提取 center（建议添加到 config 中）")
        else:
            center = 0.
            print(f"⚠ 使用默认 center 值: 0.")

        # Read scale parameter
        if hasattr(config, 'scale') and config.scale is not None:
            scale = config.scale
            if isinstance(scale, np.ndarray):
                scale = torch.from_numpy(scale).float()

            # Select sensors based on sensor_pick if provided
            if hasattr(config, 'sensor_pick') and config.sensor_pick:
                if scale.dim() == 3 and scale.shape[1] > len(config.sensor_pick):
                    scale = scale[:, config.sensor_pick, :]

            # Move to device
            scale = scale.to(device)
            print(f"✓ 从 config 读取 scale 参数")
        elif 'scale' in model_info and model_info['scale'] is not None:
            scale = model_info['scale']
            if torch.is_tensor(scale):
                scale = scale.to(device)
            if hasattr(config, 'sensor_pick') and config.sensor_pick:
                if scale.dim() == 3 and scale.shape[1] > len(config.sensor_pick):
                    scale = scale[:, config.sensor_pick, :]
            print(f"⚠ 从模型参数中提取 scale（建议添加到 config 中）")
        else:
            scale = 1.
            print(f"⚠ 使用默认 scale 值: 1.")

        # Determine c_in based on sensor_pick
        c_in = len(config.sensor_pick) if hasattr(config, 'sensor_pick') and config.sensor_pick else 5

        # Determine final_out based on label_names
        final_out = len(config.label_names) if hasattr(config, 'label_names') else 1

        # Fixed parameters for ConvTimeNet
        convtimenet_params = {
            'c_in': c_in,
            'seq_len': 280,
            'context_window': 280,
            'target_window': 280,
            'patch_len': 32,
            'stride': 16,
            'n_layers': 4,
            'd_model': 64,
            'd_ff': 128,
            'dropout': 0.3,
            'act': "relu",
            'enable_res_param': False,
            'dw_ks': [5, 5, 7, 7, 13, 13, 19, 19],  # Depth-wise kernel sizes for each layer
            'norm': 'batch',
            're_param': False,
            'deformable': True,
            'reduced_channels': 16,
            'revin': False,
            'final_out': final_out,
            'center': center,
            'scale': scale,
            'eff_hist': 248
        }

        # Override with model_info if loading weights and parameters exist
        if load_weights and model_info:
            # Get ConvTimeNet initialization parameters
            convtimenet_signature = inspect.signature(ConvTimeNet_backbone.__init__)
            convtimenet_param_names = [param.name for param in convtimenet_signature.parameters.values()
                                       if param.name != 'self']

            # Update with saved parameters if they exist
            for param_name in convtimenet_param_names:
                if param_name in model_info:
                    convtimenet_params[param_name] = model_info[param_name]

        # Ensure center and scale are on the correct device after all updates
        if 'center' in convtimenet_params and convtimenet_params['center'] is not None:
            if torch.is_tensor(convtimenet_params['center']):
                convtimenet_params['center'] = convtimenet_params['center'].to(device)

        if 'scale' in convtimenet_params and convtimenet_params['scale'] is not None:
            if torch.is_tensor(convtimenet_params['scale']):
                convtimenet_params['scale'] = convtimenet_params['scale'].to(device)

        # Create model
        model = ConvTimeNet_backbone(**convtimenet_params).to(device)

        # Load pretrained weights if requested and available
        if load_weights and state_dict is not None:
            model.load_state_dict(state_dict)
            print("Loaded pretrained ConvTimeNet weights successfully!")
        elif load_weights and state_dict is None:
            raise ValueError("No state_dict found in model file!")
        else:
            print("Using random initialization for ConvTimeNet weights.")

        # Verify model weights are not NaN
        ModelLoader._verify_model_weights(model)

        # Prepare model info for saving
        save_info = convtimenet_params

        return model, save_info

    @staticmethod
    def load_model(
            model_path: str,
            device: torch.device,
            config: Any,
            load_weights: bool = True
    ) -> Tuple[Union[TCN, ConvTimeNet_backbone], Dict[str, Any]]:
        """
        Load model (TCN or ConvTimeNet) based on config.model_mode.

        Args:
            model_path: Path to the saved model file
            device: Device to load the model on
            config: Configuration object containing model_mode, sensor_pick and label_names
            load_weights: Whether to load pretrained weights

        Returns:
            Tuple of (model, model_info_dict)
        """
        # Check model_mode in config
        model_mode = getattr(config, 'model_mode', 'TCN')  # Default to TCN if not specified

        if model_mode == 'TCN':
            print(f"Loading TCN model...")
            return ModelLoader.load_TCN(model_path, device, config, load_weights)
        elif model_mode == 'ConvTimeNet':
            print(f"Loading ConvTimeNet model...")
            return ModelLoader.load_ConvTimeNet(model_path, device, config, load_weights)
        else:
            raise ValueError(f"Unknown model_mode: {model_mode}. Supported modes: 'TCN', 'ConvTimeNet'")

    @staticmethod
    def _adjust_model_params(tcn_params: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """
        Adjust model parameters based on configuration.

        Args:
            tcn_params: Original TCN parameters
            config: Configuration object

        Returns:
            Modified TCN parameters
        """
        # Update output size based on label_names
        if hasattr(config, 'label_names'):
            tcn_params['output_size'] = len(config.label_names)

        # Update input size and normalization parameters based on sensor_pick
        if hasattr(config, 'sensor_pick') and config.sensor_pick:
            # Update input size to match number of selected sensors
            tcn_params['input_size'] = len(config.sensor_pick)

            # 优先从 config 读取归一化参数，如果没有则从模型参数中提取并调整
            if hasattr(config, 'center') and config.center is not None:
                # 从 config 读取 center
                import numpy as np
                import torch
                center = config.center
                if isinstance(center, np.ndarray):
                    center = torch.from_numpy(center).float()

                # 根据 sensor_pick 选择对应的传感器
                if center.dim() == 3 and center.shape[1] > len(config.sensor_pick):
                    tcn_params['center'] = center[:, config.sensor_pick, :]
                else:
                    tcn_params['center'] = center

                print(f"✓ 从 config 读取 center 参数")
            else:
                # 从模型参数中提取（原有逻辑）
                if 'center' in tcn_params and tcn_params['center'] is not None:
                    tcn_params['center'] = tcn_params['center'][:, config.sensor_pick, :]
                    print(f"⚠ 从模型参数中提取 center（建议添加到 config 中）")

            if hasattr(config, 'scale') and config.scale is not None:
                # 从 config 读取 scale
                import numpy as np
                import torch
                scale = config.scale
                if isinstance(scale, np.ndarray):
                    scale = torch.from_numpy(scale).float()

                # 根据 sensor_pick 选择对应的传感器
                if scale.dim() == 3 and scale.shape[1] > len(config.sensor_pick):
                    tcn_params['scale'] = scale[:, config.sensor_pick, :]
                else:
                    tcn_params['scale'] = scale

                print(f"✓ 从 config 读取 scale 参数")
            else:
                # 从模型参数中提取（原有逻辑）
                if 'scale' in tcn_params and tcn_params['scale'] is not None:
                    tcn_params['scale'] = tcn_params['scale'][:, config.sensor_pick, :]
                    print(f"⚠ 从模型参数中提取 scale（建议添加到 config 中）")

            print(f"Modified model parameters for sensor selection:")
            print(f"  - Input size: {tcn_params.get('input_size', 'N/A')}")
            print(f"  - Output size: {tcn_params.get('output_size', 'N/A')}")
            print(f"  - Selected sensors: {config.sensor_pick}")

        return tcn_params

    @staticmethod
    def _verify_model_weights(model: Union[TCN, ConvTimeNet_backbone]) -> None:
        """
        Verify that model weights don't contain NaN values.

        Args:
            model: TCN or ConvTimeNet model to verify

        Raises:
            ValueError: If NaN values are found in model weights
        """
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN found in initial parameter: {name}. "
                                 "Please check the pretrained model file!")

    @staticmethod
    def save_checkpoint(
            model: Union[TCN, ConvTimeNet_backbone],
            optimizer: torch.optim.Optimizer,
            epoch: int,
            loss: float,
            save_path: str,
            model_info: Dict[str, Any]
    ) -> None:
        """
        Save model checkpoint in tar format.

        Args:
            model: TCN or ConvTimeNet model
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