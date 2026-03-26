#!/usr/bin/env python3
"""Validate the RKNN mode classification model on the target device via ADB."""
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from data_utils import prepare_data


def create_argument_parser():
    """Create and configure argument parser for validation."""
    parser = argparse.ArgumentParser(description='Validate RKNN mode classifier on device via ADB')
    parser.add_argument('--val-list', type=str,
                        default='timesnet_rknn/1208_lehiuju_val.txt',
                        help='Path to validation dataset list')
    parser.add_argument('--dataset-root', type=str, default='timesnet_rknn',
                        help='Root directory of dataset')
    parser.add_argument('--window-size', type=int, default=100,
                        help='Window size for input sequences')
    parser.add_argument('--segments-per-record', type=int, default=5,
                        help='Number of segments per record')
    parser.add_argument('--class-names', type=str,
                        default='upstair,downstair,walk,stand',
                        help='Comma separated class names')
    parser.add_argument('--stats', type=Path, required=True,
                        help='Path to dataset_stats.json generated during training')
    parser.add_argument('--rknn-path', type=str, required=True,
                        help='Path to RKNN model file on device (e.g., /root/apps/model/modes_cnn_int8.rknn)')
    parser.add_argument('--infer-cmd', type=str,
                        default='/root/apps/npy_infer/pattern_npy_infer',
                        help='Path to inference executable on device')
    parser.add_argument('--device-tmp', type=str,
                        default='/tmp/',
                        help='Temporary directory on device for npy files')
    parser.add_argument('--local-tmp', type=Path,
                        default=Path('timesnet_rknn/tmp_modes_input.npy'),
                        help='Local temporary file for npy input')
    parser.add_argument('--result-bin', type=Path,
                        default=Path('timesnet_rknn/adb_data.bin'),
                        help='Local file to receive inference results')
    parser.add_argument('--device-result-bin', type=str,
                        default='/tmp/adb_data.bin',
                        help='Result binary path on device to pull after inference')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to validate (for debugging)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (currently only 1 is supported)')
    parser.add_argument('--debug', action='store_true',
                        help='Show detailed debug information')
    parser.add_argument('--save-results', type=Path,
                        help='Path to save validation results JSON')
    return parser


def load_stats(stats_path: Path) -> Dict:
    """Load dataset statistics from JSON file."""
    stats = json.loads(stats_path.read_text())
    required = ['mean', 'std', 'classes', 'window_size', 'num_features']
    for key in required:
        if key not in stats:
            raise ValueError(f'missing "{key}" in {stats_path}')
    return stats


def prepare_input_npy(sample: np.ndarray, output_path: Path) -> None:
    """
    Prepare input sample as NPY file in NCHW format.
    
    Args:
        sample: Input array of shape (window_size, num_features)
        output_path: Path to save the NPY file
    """
    # Convert to NCHW: (window, features) -> (1, features, 1, window)
    sample = np.transpose(sample, (1, 0))  # (features, window)
    sample = sample[:, np.newaxis, :]      # (features, 1, window)
    sample = np.expand_dims(sample, axis=0)  # (1, features, 1, window)
    sample = np.ascontiguousarray(sample, dtype=np.float32)
    np.save(output_path, sample)


def push_to_device(local_path: Path, device_path: str, debug: bool = False) -> bool:
    """Push file to device via ADB."""
    result = subprocess.run(
        ["adb", "push", str(local_path), device_path],
        stdout=None if debug else subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(f"Failed to push {local_path} to device: {result.stderr.decode()}")
        return False
    return True


def run_inference(infer_cmd: str, rknn_path: str, input_path: str, debug: bool = False) -> bool:
    """Run inference on device via ADB shell."""
    result = subprocess.run(
        ["adb", "shell", infer_cmd, rknn_path, input_path],
        stdout=None if debug else subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(f"Inference failed: {result.stderr.decode()}")
        return False
    return True


def pull_from_device(device_path: str, local_path: Path, debug: bool = False) -> bool:
    """Pull inference result binary from device via ADB."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["adb", "pull", device_path, str(local_path)],
        stdout=None if debug else subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(f"Failed to pull {device_path} to {local_path}: {result.stderr.decode()}")
        return False
    return True


def get_inference_result(result_bin: Path, num_classes: int, debug: bool = False) -> np.ndarray:
    """
    Read inference results from binary file.
    
    Args:
        result_bin: Path to the result binary file
        num_classes: Number of output classes
        debug: Whether to print debug info
    
    Returns:
        Numpy array of shape (num_classes,) with class probabilities or logits
    """
    if not result_bin.exists():
        raise FileNotFoundError(f"Result file not found: {result_bin}")
    
    # Read the binary file - assuming it contains float32 values
    with open(result_bin, "rb") as f:
        # The output from pattern_npy_infer.cpp converts fp16 to float
        # and sends it as int8 bytes, but we need to reverse that conversion
        data = np.fromfile(f, dtype=np.int8)
        
        # Based on pattern_npy_infer.cpp: result[i] = (float)out_fp16[i]
        # then sent as int8, so we need to interpret it correctly
        # Assuming the output is 4 float values sent somehow
        # Let's read as raw bytes and reinterpret
        f.seek(0)
        raw_bytes = f.read()
        
    if debug:
        print(f"Read {len(raw_bytes)} bytes from result file")
    
    # Try to interpret as float32 directly (4 bytes per float, 4 classes = 16 bytes)
    if len(raw_bytes) >= num_classes * 4:
        outputs = np.frombuffer(raw_bytes[:num_classes * 4], dtype=np.float32)
    else:
        # Fallback: pad or handle error
        print(f"Warning: Expected {num_classes * 4} bytes, got {len(raw_bytes)}")
        outputs = np.zeros(num_classes, dtype=np.float32)
    
    return outputs


def compute_confusion_matrix(predictions: List[int], labels: List[int], num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(predictions, labels):
        confusion[label, pred] += 1
    return confusion


def compute_metrics(confusion: np.ndarray, class_names: List[str]) -> Dict:
    """Compute accuracy metrics from confusion matrix."""
    total = confusion.sum()
    correct = np.trace(confusion)
    overall_accuracy = float(correct / total) if total > 0 else 0.0
    
    per_class_metrics = {}
    for idx, name in enumerate(class_names):
        true_positives = confusion[idx, idx]
        false_positives = confusion[:, idx].sum() - true_positives
        false_negatives = confusion[idx, :].sum() - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': int(confusion[idx, :].sum())
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_samples': int(total),
        'correct_predictions': int(correct),
        'per_class': per_class_metrics,
        'confusion_matrix': confusion.tolist()
    }


def print_summary(metrics: Dict, class_names: List[str]) -> None:
    """Print validation summary."""
    print('\n' + '=' * 60)
    print('VALIDATION RESULTS')
    print('=' * 60)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
    print('\nPer-Class Metrics:')
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
    print('-' * 60)
    
    for name in class_names:
        m = metrics['per_class'][name]
        print(f"{name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['samples']:<10}")
    
    print('\nConfusion Matrix:')
    print(f"{'True/Pred':<15}", end='')
    for name in class_names:
        print(f"{name[:10]:<12}", end='')
    print()
    print('-' * (15 + 12 * len(class_names)))
    
    confusion = np.array(metrics['confusion_matrix'])
    for idx, name in enumerate(class_names):
        print(f"{name:<15}", end='')
        for pred_idx in range(len(class_names)):
            print(f"{confusion[idx, pred_idx]:<12}", end='')
        print()
    print('=' * 60)


def validate_model(args: argparse.Namespace) -> Dict:
    """Main validation loop."""
    # Load statistics and prepare class names
    stats = load_stats(args.stats)
    class_names = [cls.strip().lower() for cls in args.class_names.split(',') if cls.strip()]
    num_classes = len(class_names)
    
    print(f'[info] Loading validation dataset...')
    val_x_raw, val_y = prepare_data(
        args.val_list,
        Path(args.dataset_root),
        class_names,
        args.window_size,
        args.segments_per_record
    )
    
    # Apply max_samples limit if specified
    if args.max_samples and len(val_x_raw) > args.max_samples:
        val_x_raw = val_x_raw[:args.max_samples]
        val_y = val_y[:args.max_samples]
        print(f'[info] Limited to {args.max_samples} samples for validation')
    
    print(f'[info] Validating {len(val_x_raw)} samples on device...')
    print(f'[info] RKNN model: {args.rknn_path}')
    
    predictions = []
    labels = []
    
    # Ensure local tmp directory exists
    args.local_tmp.parent.mkdir(parents=True, exist_ok=True)
    
    with tqdm(total=len(val_x_raw), desc='Validating') as pbar:
        for idx in range(len(val_x_raw)):
            sample = val_x_raw[idx]
            label = val_y[idx]
            
            # Prepare input NPY file
            prepare_input_npy(sample, args.local_tmp)
            
            # Push to device
            device_npy_path = args.device_tmp + args.local_tmp.name
            if not push_to_device(args.local_tmp, device_npy_path, args.debug):
                print(f"Skipping sample {idx} due to push failure")
                continue
            
            # Run inference on device
            if not run_inference(args.infer_cmd, args.rknn_path, device_npy_path, args.debug):
                print(f"Skipping sample {idx} due to inference failure")
                continue

            # Pull latest inference result from device
            if not pull_from_device(args.device_result_bin, args.result_bin, args.debug):
                print(f"Skipping sample {idx} due to result pull failure")
                continue
            
            # Get results
            try:
                outputs = get_inference_result(args.result_bin, num_classes, args.debug)
                pred = int(np.argmax(outputs))
                predictions.append(pred)
                labels.append(int(label))
                
                if args.debug:
                    print(f"Sample {idx}: True={class_names[label]}, Pred={class_names[pred]}, Logits={outputs}")
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
            
            pbar.update(1)
            pbar.set_postfix({
                'acc': f'{sum(np.array(predictions) == np.array(labels)) / len(predictions):.4f}'
            })
    
    # Compute metrics
    confusion = compute_confusion_matrix(predictions, labels, num_classes)
    metrics = compute_metrics(confusion, class_names)
    
    return metrics


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate model
    metrics = validate_model(args)
    
    # Print summary
    class_names = [cls.strip().lower() for cls in args.class_names.split(',') if cls.strip()]
    print_summary(metrics, class_names)
    
    # Save results if requested
    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'\n[info] Results saved to {args.save_results}')


if __name__ == '__main__':
    main()
