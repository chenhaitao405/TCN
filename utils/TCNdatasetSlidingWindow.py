import os
import re
import hashlib
import json
from typing import List, Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from datetime import datetime
from tqdm import tqdm


class TcnDatasetSlidingWindow(Dataset):
    '''Dataset for loading data using sliding window approach.'''

    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: str,
                 window_size: int = 280,
                 window_stride: int = 10,
                 participant_masses: Dict[str, float] = {},
                 action_patterns: Optional[List[str]] = None,
                 device: torch.device = torch.device("cpu"),
                 cache_dir: str = 'cache',
                 cache_suffix: str = '_sliding'):
        """
        Initialize sliding window dataset.

        Args:
            window_size: Size of each window (default 280)
            window_stride: Stride for sliding window (default 10)
            cache_suffix: Suffix for cache files to differentiate from regular dataset
        """
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.window_size = window_size
        self.window_stride = window_stride
        self.participant_masses = participant_masses
        self.action_patterns = action_patterns
        self.device = device
        self.cache_dir = cache_dir
        self.cache_suffix = cache_suffix

        # Get trial names
        self.trial_names = self._get_trial_names()

        if self.action_patterns:
            print(f"  - Action patterns: {self.action_patterns}")
            print(f"  - Matched trials: {len(self.trial_names)}")

        # Initialize window indices
        print(f"  - Window size: {self.window_size}, Stride: {self.window_stride}")
        self.window_indices = self._get_or_compute_window_indices()
        print(f"  - Total valid windows: {len(self.window_indices)}")

    def __len__(self):
        '''Returns number of valid windows.'''
        return len(self.window_indices)

    def __getitem__(self, idx: int):
        '''Loads data for a specific window.'''
        window_info = self.window_indices[idx]
        trial_name = window_info['trial_name']
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']

        # Load trial data
        input_data, label_data = self._load_trial_data_train(trial_name)

        # Extract window
        window_input = input_data[:, :, start_idx:end_idx]
        window_label = label_data[:, :, start_idx:end_idx]

        # Window info for tracking
        window_metadata = {
            'trial_name': trial_name,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_idx': idx
        }

        # Return format compatible with original dataset
        # Note: seq_lengths is always window_size for sliding window
        return window_input, window_label, [self.window_size], [window_metadata]

    def _get_or_compute_window_indices(self) -> List[Dict]:
        """Get or compute valid window indices with caching."""
        os.makedirs(self.cache_dir, exist_ok=True)

        # Generate cache key
        cache_key_components = [
            str(self.data_dir),
            str(self.input_names),
            str(self.label_names),
            self.side,
            str(self.window_size),
            str(self.window_stride),
            str(self.action_patterns) if self.action_patterns else '',
            self.cache_suffix
        ]
        cache_key = ''.join(cache_key_components)
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_path = os.path.join(self.cache_dir, f'window_indices_{cache_hash}.json')

        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                # Validate cache
                if (cache_data.get('window_size') == self.window_size and
                        cache_data.get('window_stride') == self.window_stride and
                        cache_data.get('num_trials') == len(self.trial_names)):
                    print(f"  - Loaded cached window indices: {len(cache_data['window_indices'])} windows")
                    return cache_data['window_indices']
            except Exception as e:
                print(f"  - Cache loading failed: {e}")

        # Compute window indices
        print("  - Computing valid window indices...")
        window_indices = []

        for trial_name in tqdm(self.trial_names, desc="Processing trials"):
            try:
                # Load trial data
                input_data, label_data = self._load_trial_data_train(trial_name)
                trial_length = input_data.shape[-1]

                # Skip if trial is too short
                if trial_length < self.window_size:
                    continue

                # Generate windows for this trial
                for start_idx in range(0, trial_length - self.window_size + 1, self.window_stride):
                    end_idx = start_idx + self.window_size

                    # Check for NaN in window
                    window_input = input_data[:, :, start_idx:end_idx]
                    window_label = label_data[:, :, start_idx:end_idx]

                    if not torch.isnan(window_input).any() and not torch.isnan(window_label).any():
                        window_indices.append({
                            'trial_name': trial_name,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'trial_length': trial_length
                        })
            except Exception as e:
                print(f"    Warning: Failed to process {trial_name}: {e}")
                continue

        # Save cache
        cache_data = {
            'window_indices': window_indices,
            'window_size': self.window_size,
            'window_stride': self.window_stride,
            'num_trials': len(self.trial_names),
            'data_dir': self.data_dir,
            'side': self.side,
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            print(f"  - Saved window indices cache to: {cache_path}")
        except Exception as e:
            print(f"  - Failed to save cache: {e}")

        return window_indices

    def extract_action_type(self, trial_name: str) -> str:
        """Extract action type from trial name."""
        if '/' in trial_name:
            folder_name = trial_name.split('/')[-1]
        elif '\\' in trial_name:
            folder_name = trial_name.split('\\')[-1]
        else:
            folder_name = trial_name

        parts = folder_name.split('_')
        compound_actions = ['normal_walk', 'dynamic_walk', 'incline_walk', 'walk_backward',
                            'weighted_walk', 'obstacle_walk', 'sit_to_stand', 'curb_down',
                            'curb_up', 'lift_weight', 'side_shuffle', 'tug_of_war',
                            'turn_and_step', 'tire_run', 'start_stop', 'step_ups']

        if len(parts) >= 2:
            potential_compound = f"{parts[0]}_{parts[1]}"
            if potential_compound == 'normal_walk':
                normal_walk_types = ['0-6', '1-2', '1-8', '2-0', '2-5', 'shuffle', 'skip']
                for part in parts[2:]:
                    if part in normal_walk_types:
                        return f"normal_walk_{part}"
                return 'normal_walk'
            elif potential_compound in compound_actions:
                return potential_compound

        return parts[0]

    def _match_action_patterns(self, trial_folder_name: str) -> bool:
        """Check if trial folder name matches any specified regex patterns."""
        if not self.action_patterns:
            return True
        for pattern in self.action_patterns:
            if re.match(pattern, trial_folder_name):
                return True
        return False

    def _get_trial_names(self):
        '''Get all trial names in data_dir, filtered by action patterns if specified.'''
        participants = [participant for participant in os.listdir(self.data_dir)
                        if "." not in participant and participant != "LICENSE"]

        trial_names = []
        action_stats = {}

        for participant in participants:
            participant_dir = os.path.join(self.data_dir, participant)
            for trial_name in os.listdir(participant_dir):
                if self._match_action_patterns(trial_name):
                    full_trial_name = os.path.join(participant, trial_name)
                    trial_names.append(full_trial_name)
                    action_type = self.extract_action_type(full_trial_name)
                    action_stats[action_type] = action_stats.get(action_type, 0) + 1

        if self.action_patterns and action_stats:
            print(f"  - Action distribution: {action_stats}")

        return trial_names

    def _load_trial_data_train(self, trial_name: str):
        '''Loads data from a single trial.'''
        trial_dir = os.path.join(self.data_dir, trial_name)
        input_file_path = None
        for file in os.listdir(trial_dir):
            file_lower = file.lower()
            if file_lower.endswith("exo.csv") and not file_lower.endswith("power_exo.csv"):
                input_file_path = os.path.join(trial_dir, file)
                break

        if input_file_path is None:
            raise FileNotFoundError(f"No file ending with '_exo.csv' found in {trial_dir}")

        participant = trial_name.split("/")[0].split("\\")[0]
        if participant not in self.participant_masses:
            print(f"Warning - {participant} mass was not provided.")
        input_data = self._load_input_data(input_file_path, body_mass=self.participant_masses.get(participant, 1.))

        label_file_path = None
        for file in os.listdir(trial_dir):
            file_lower = file.lower()
            if file_lower.endswith("_moment_filt.csv"):
                label_file_path = os.path.join(trial_dir, file)
                break

        if label_file_path is None:
            raise FileNotFoundError(f"No file ending with '_moment_filt.csv' found in {trial_dir}")

        label_data = self._load_label_data(label_file_path)

        return input_data, label_data

    def _load_input_data(self, file_path: str, body_mass: float):
        '''Loads input data from a single file and returns as a 3D torch.FloatTensor.'''
        df = pd.read_csv(file_path)

        df.loc[:, "insole_l_force_y"] /= body_mass
        df.loc[:, "insole_r_force_y"] /= body_mass

        if self.side == "l":
            df.loc[:, "foot_imu_l_gyro_x"] *= -1.
            df.loc[:, "foot_imu_l_gyro_y"] *= -1.
            df.loc[:, "foot_imu_l_accel_z"] *= -1.
            df.loc[:, "shank_imu_l_gyro_x"] *= -1.
            df.loc[:, "shank_imu_l_gyro_y"] *= -1.
            df.loc[:, "shank_imu_l_accel_z"] *= -1.
            df.loc[:, "thigh_imu_l_gyro_x"] *= -1.
            df.loc[:, "thigh_imu_l_gyro_y"] *= -1.
            df.loc[:, "thigh_imu_l_accel_z"] *= -1.
            df.loc[:, "insole_l_cop_z"] *= -1.

        input_data = torch.tensor(df[self.input_names].values, device=self.device).transpose(0, 1).unsqueeze(0).float()
        return input_data

    def _load_label_data(self, file_path: str):
        '''Loads label data from a single file and returns as a 3D torch.FloatTensor.'''
        df = pd.read_csv(file_path)
        label_data = torch.tensor(df[self.label_names].values, device=self.device).transpose(0, 1).unsqueeze(0).float()
        return label_data