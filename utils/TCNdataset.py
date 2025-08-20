import os
import re
from typing import List, Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset


class TcnDataset(Dataset):
    '''Dataset for dynamically loading input and label data based on indices.'''

    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: str,
                 participant_masses: Dict[str, float] = {},
                 action_patterns: Optional[List[str]] = None,  # 新增参数
                 device: torch.device = torch.device("cpu")):
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.participant_masses = participant_masses
        self.action_patterns = action_patterns  # 存储正则表达式模式
        self.device = device
        self.trial_names = self._get_trial_names()

        # 打印筛选信息
        if self.action_patterns:
            print(f"  - Action patterns: {self.action_patterns}")
            print(f"  - Matched trials: {len(self.trial_names)}")

    def __len__(self):
        '''Returns number of files found.'''
        return len(self.trial_names)

    def __getitem__(self, idx: int or List[int] or slice):
        '''Loads data based on provided indices. Uses zero padding to concatenate trials of different size.'''
        # Get list of desired file names based on idx
        if isinstance(idx, list):
            trial_names = [self.trial_names[i] for i in idx]
        else:
            trial_names = self.trial_names[idx]
            trial_names = [trial_names] if not isinstance(trial_names, list) else trial_names

        # Load data
        data = [list(self._load_trial_data_train(trial_name)) for trial_name in trial_names]

        # add zero padding to allow for concatenation
        data, trial_sequence_lengths = self._add_zero_padding(data)

        # concatenate tensors
        input_data, label_data = zip(*data)
        input_data = torch.cat(input_data, dim=0)
        label_data = torch.cat(label_data, dim=0)

        return input_data, label_data, trial_sequence_lengths

    def get_trial_names(self):
        return self.trial_names

    def _match_action_patterns(self, trial_folder_name: str) -> bool:
        """
		检查试验文件夹名是否匹配任何指定的正则表达式模式。

		Args:
			trial_folder_name: 试验文件夹名（不包含参与者前缀）

		Returns:
			bool: 如果匹配返回True，否则返回False
		"""
        if not self.action_patterns:  # 如果没有指定模式，接受所有
            return True

        for pattern in self.action_patterns:
            if re.match(pattern, trial_folder_name):
                return True
        return False

    def _get_trial_names(self):
        '''Get all trial names in data_dir, filtered by action patterns if specified.'''
        # extract participant directories
        participants = [participant for participant in os.listdir(self.data_dir)
                        if "." not in participant and participant != "LICENSE"]

        # iterate through participant directories and get trial names
        trial_names = []
        action_stats = {}  # 统计每种动作的数量

        for participant in participants:
            participant_dir = os.path.join(self.data_dir, participant)
            for trial_name in os.listdir(participant_dir):
                # 检查是否匹配动作模式
                if self._match_action_patterns(trial_name):
                    trial_names.append(os.path.join(participant, trial_name))

                    # 统计动作类型
                    base_action = trial_name.split('_')[0]
                    if '_' in trial_name and trial_name.split('_')[0] in ['normal', 'dynamic', 'incline', 'walk', 'sit',
                                                                          'turn', 'side', 'tug', 'lift']:
                        base_action = '_'.join(trial_name.split('_')[:2])
                    action_stats[base_action] = action_stats.get(base_action, 0) + 1

        # 打印统计信息
        if self.action_patterns and action_stats:
            print(f"  - Action distribution: {action_stats}")

        return trial_names

    def _load_trial_data_train(self, trial_name: str):
        '''Loads data from a single trial.'''
        # 搜索文件夹中结尾为"_exo.csv"但不是"_power_exo.csv"的文件（不区分大小写）
        trial_dir = os.path.join(self.data_dir, trial_name)
        input_file_path = None
        for file in os.listdir(trial_dir):
            # 将文件名转换为小写后进行比较
            file_lower = file.lower()
            if file_lower.endswith("exo.csv") and not file_lower.endswith("power_exo.csv"):
                input_file_path = os.path.join(trial_dir, file)  # 使用原始文件名构建路径
                break

        if input_file_path is None:
            raise FileNotFoundError(f"No file ending with '_exo.csv' (excluding '_power_exo.csv') found in {trial_dir}")

        participant = trial_name.split("/")[0].split("\\")[0]  # get participant name for body mass normalization
        if participant not in self.participant_masses:
            print(f"Warning - {participant} mass was not provided.")
        input_data = self._load_input_data(input_file_path, body_mass=self.participant_masses.get(participant, 1.))

        # load label data
        # 搜索文件夹中结尾为"_moment_filt.csv"的文件（不区分大小写）
        label_file_path = None
        for file in os.listdir(trial_dir):
            # 将文件名转换为小写后进行比较
            file_lower = file.lower()
            if file_lower.endswith("_moment_filt.csv"):
                label_file_path = os.path.join(trial_dir, file)  # 使用原始文件名构建路径
                break

        if label_file_path is None:
            raise FileNotFoundError(f"No file ending with '_moment_filt.csv' found in {trial_dir}")

        label_data = self._load_label_data(label_file_path)

        return input_data, label_data

    def _load_input_data(self, file_path: str, body_mass: float):
        '''Loads input data from a single file and returns as a 3D torch.FloatTensor.'''
        # load as DataFrame
        df = pd.read_csv(file_path)

        # normalize pressure insole data by body mass
        df.loc[:, "insole_l_force_y"] /= body_mass
        df.loc[:, "insole_r_force_y"] /= body_mass

        # if left leg data, mirror sensors
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

        # convert to input and label tensors
        input_data = torch.tensor(df[self.input_names].values, device=self.device).transpose(0, 1).unsqueeze(0).float()

        return input_data

    def _load_label_data(self, file_path: str):
        '''Loads label data from a single file and returns as a 3D torch.FloatTensor.'''
        # load as DataFrame
        df = pd.read_csv(file_path)

        # convert to input and label tensors
        label_data = torch.tensor(df[self.label_names].values, device=self.device).transpose(0, 1).unsqueeze(0).float()

        return label_data

    def _add_zero_padding(self, data: List[List[torch.FloatTensor]]):
        '''Adds zero padding to the end of each trial to match the sequence lengths of all trial data.'''
        trial_sequence_lengths = [trial_data[0].shape[-1] for trial_data in data]
        max_sequence_length = max(trial_sequence_lengths)

        # iterate through each trial and add zero padding as needed
        for i in range(len(data)):
            trial_sequence_length = trial_sequence_lengths[i]
            if trial_sequence_length < max_sequence_length:
                # pad input data and label data
                padding_length = max_sequence_length - trial_sequence_length
                for j in range(len(data[i])):
                    data[i][j] = torch.cat(
                        (data[i][j], torch.zeros((1, data[i][j].shape[1], padding_length), device=self.device)), dim=2)

        return data, trial_sequence_lengths