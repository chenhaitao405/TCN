"""Shared dataset loading utilities for IMU/motor motion classification."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np

IMU_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
MOTOR_COLUMNS = ['motor_angle_L_float', 'motor_angle_R_float']
SENSORS = ['imuData2', 'motorAngle']
SKIP_PATTERN = re.compile(r"skip|switch|start", re.IGNORECASE)


def detect_separator(file_path: str) -> str:
    with open(file_path, 'r') as handle:
        first_line = handle.readline()
    return '\t' if '\t' in first_line else ','


def read_sensor_csv(csv_path: Path, is_motor: bool) -> np.ndarray:
    delimiter = detect_separator(str(csv_path))
    columns = MOTOR_COLUMNS if is_motor else IMU_COLUMNS
    values: List[List[float]] = []
    with open(csv_path, 'r') as handle:
        reader = csv.DictReader(handle, delimiter=delimiter, skipinitialspace=True)
        for row in reader:
            values.append([float(row[col]) for col in columns])
    if not values:
        return np.zeros((0, len(columns)), dtype=np.float32)
    return np.array(values, dtype=np.float32)


def ensure_sensor_files(sample_dir: Path) -> bool:
    for sensor in SENSORS:
        if not (sample_dir / f"{sensor}.csv").is_file():
            print(f"[warn] Missing {sensor}.csv under {sample_dir}, skip this sample")
            return False
    return True


def interpolate_motor(data: np.ndarray, target_len: int) -> np.ndarray:
    if data.shape[0] == target_len:
        return data
    frames_in = np.linspace(0, 1, data.shape[0])
    frames_out = np.linspace(0, 1, target_len)
    interpolated = []
    for col in range(data.shape[1]):
        series = np.interp(frames_out, frames_in, data[:, col])
        interpolated.append(series)
    return np.stack(interpolated, axis=1)


def parse_dataset_list(list_path: str, dataset_root: Path) -> List[Path]:
    dataset_root = Path(dataset_root)
    result: List[Path] = []
    with open(list_path, 'r') as handle:
        for line in handle:
            path = line.strip()
            if not path or path.startswith('#'):
                continue
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = dataset_root / candidate
            result.append(candidate)
    return result


def collect_segments(dataset_dirs: Sequence[Path],
                     class_names: Sequence[str],
                     window_size: int,
                     segments_per_record: int,
                     seed: int = 42) -> Tuple[List[np.ndarray], List[int]]:
    class_names = [cls.lower() for cls in class_names]
    rng = np.random.default_rng(seed)
    segments: List[np.ndarray] = []
    labels: List[int] = []
    for person_dir in dataset_dirs:
        if not person_dir.is_dir():
            print(f"[warn] Missing directory {person_dir}, skip")
            continue
        for cls_folder in sorted(person_dir.iterdir()):
            if not cls_folder.is_dir():
                continue
            folder_name = cls_folder.name.lower()
            if SKIP_PATTERN.search(folder_name):
                continue
            matches = [idx for idx, cls in enumerate(class_names) if cls in folder_name]
            if not matches:
                continue
            label = matches[0]
            if not ensure_sensor_files(cls_folder):
                continue
            imu_np = read_sensor_csv(cls_folder / 'imuData2.csv', is_motor=False)
            motor_np = read_sensor_csv(cls_folder / 'motorAngle.csv', is_motor=True)
            motor_np = interpolate_motor(motor_np, imu_np.shape[0])
            merged = np.concatenate([imu_np, motor_np], axis=1)
            total_len = merged.shape[0]
            if total_len < window_size:
                continue
            num_segments = max(1, (total_len // window_size) * segments_per_record)
            for _ in range(num_segments):
                start = rng.integers(0, total_len - window_size + 1)
                segment = merged[start:start + window_size, :]
                segments.append(segment)
                labels.append(label)
    return segments, labels


def prepare_data(list_path: str,
                 dataset_root: Path,
                 class_names: Sequence[str],
                 window_size: int,
                 segments_per_record: int,
                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    dataset_dirs = parse_dataset_list(list_path, dataset_root)
    segments, labels = collect_segments(dataset_dirs, class_names,
                                        window_size, segments_per_record,
                                        seed=seed)
    if not segments:
        raise RuntimeError(f"No samples collected from {list_path}")
    x = np.stack(segments).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    return x, y
