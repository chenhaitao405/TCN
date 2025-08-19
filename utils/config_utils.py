"""
Configuration loading and processing utilities.
"""
import importlib
import json
import os
from typing import Any, Dict
from datetime import datetime


class ConfigManager:
    """Handle configuration loading and processing."""

    @staticmethod
    def load_config(config_path: str) -> Any:
        """
        Load config file as module.

        Args:
            config_path: Path to config file

        Returns:
            Loaded configuration module
        """
        config_path = config_path.replace("/", ".").replace("\\", ".")
        if config_path.endswith(".py"):
            config_path = config_path[:-3]
        print(f"Loading config file from {config_path}.")
        return importlib.import_module(config_path)

    @staticmethod
    def apply_sensor_selection(config: Any) -> Any:
        """
        Apply sensor_pick filtering to config if specified.

        Args:
            config: Configuration object

        Returns:
            Modified configuration object
        """
        if hasattr(config, 'sensor_pick') and config.sensor_pick:
            # Filter input_names based on sensor_pick indices
            if hasattr(config, 'input_names'):
                original_input_names = config.input_names.copy()
                filtered_input_names = [config.input_names[i] for i in config.sensor_pick
                                       if i < len(config.input_names)]
                config.input_names = filtered_input_names

                print(f"Sensor selection applied:")
                print(f"  - Original number of inputs: {len(original_input_names)}")
                print(f"  - Selected sensor indices: {config.sensor_pick}")
                print(f"  - Number of selected inputs: {len(config.input_names)}")
                print(f"  - Selected input names: {config.input_names}")

        return config

    @staticmethod
    def save_training_config(
        args: Any,
        config: Any,
        save_dir: str
    ) -> str:
        """
        Save training arguments and configuration to a JSON file.

        Args:
            args: Parsed command line arguments
            config: Configuration object
            save_dir: Directory to save the file

        Returns:
            Path to saved file
        """
        # Start with training arguments
        save_data = vars(args).copy()

        # Extract all config attributes
        config_dict = {}
        for attr_name in dir(config):
            if not attr_name.startswith('__'):
                attr_value = getattr(config, attr_name)
                # Convert non-serializable types to serializable ones
                if isinstance(attr_value, (list, tuple, str, int, float, bool, dict)):
                    config_dict[attr_name] = attr_value
                elif attr_value is None:
                    config_dict[attr_name] = None
                else:
                    # Try to convert to string for other types
                    try:
                        config_dict[attr_name] = str(attr_value)
                    except:
                        pass

        # Add config to save_data
        save_data['config'] = config_dict

        # Save combined data to single JSON file
        args_file = os.path.join(save_dir, 'training_args.json')
        with open(args_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Training arguments and configuration saved to: {args_file}")
        return args_file

    @staticmethod
    def save_validation_config(
        args: Any,
        config: Any,
        save_dir: str
    ) -> str:
        """
        Save validation arguments and configuration to a JSON file.

        Args:
            args: Parsed command line arguments
            config: Configuration object
            save_dir: Directory to save the file

        Returns:
            Path to saved file
        """
        # Similar to save_training_config but for validation
        save_data = vars(args).copy()

        # Extract config attributes
        config_dict = {}
        for attr_name in dir(config):
            if not attr_name.startswith('__'):
                attr_value = getattr(config, attr_name)
                if isinstance(attr_value, (list, tuple, str, int, float, bool, dict)):
                    config_dict[attr_name] = attr_value
                elif attr_value is None:
                    config_dict[attr_name] = None
                else:
                    try:
                        config_dict[attr_name] = str(attr_value)
                    except:
                        pass

        save_data['config'] = config_dict

        # Save to JSON file
        args_file = os.path.join(save_dir, 'validation_args.json')
        with open(args_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Validation arguments and configuration saved to: {args_file}")
        return args_file

    @staticmethod
    def setup_training_directory(base_dir: str = 'checkpoints') -> str:
        """
        Create training directory with timestamp.

        Args:
            base_dir: Base directory for checkpoints

        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(base_dir, f'train_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created training directory: {save_dir}")
        return save_dir

    @staticmethod
    def setup_validation_directory(base_dir: str = 'validation_results') -> str:
        """
        Create validation directory with timestamp.

        Args:
            base_dir: Base directory for validation results

        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(base_dir, f'validation_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created validation directory: {save_dir}")
        return save_dir