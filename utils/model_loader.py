"""
Unified model loading utilities for training and validation.
"""
import torch
import inspect
from utils.tcn import TCN
from typing import Tuple, Dict, Any


class ModelLoader:
    """Handle model loading with various configurations."""

    @staticmethod
    def load_pretrained_model(
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

        # Modify parameters based on configuration when not loading weights
        if not load_weights:
            tcn_params = ModelLoader._adjust_model_params(tcn_params, config)

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

            # Update center array - always [1, features, 1] shape
            if 'center' in tcn_params and tcn_params['center'] is not None:
                tcn_params['center'] = tcn_params['center'][:, config.sensor_pick, :]

            # Update scale array - always [1, features, 1] shape
            if 'scale' in tcn_params and tcn_params['scale'] is not None:
                tcn_params['scale'] = tcn_params['scale'][:, config.sensor_pick, :]

            print(f"Modified model parameters for sensor selection:")
            print(f"  - Input size: {tcn_params.get('input_size', 'N/A')}")
            print(f"  - Output size: {tcn_params.get('output_size', 'N/A')}")
            print(f"  - Selected sensors: {config.sensor_pick}")

        return tcn_params

    @staticmethod
    def _verify_model_weights(model: TCN) -> None:
        """
        Verify that model weights don't contain NaN values.

        Args:
            model: TCN model to verify

        Raises:
            ValueError: If NaN values are found in model weights
        """
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN found in initial parameter: {name}. "
                               "Please check the pretrained model file!")

    @staticmethod
    def save_checkpoint(
        model: TCN,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        save_path: str,
        model_info: Dict[str, Any]
    ) -> None:
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