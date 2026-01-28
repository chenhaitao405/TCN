#!/usr/bin/env python3
"""Evaluate the RKNN quantized model either on simulator or target device."""
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from rknn.api import RKNN

from data_utils import prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate RKNN model accuracy on motion dataset')
    parser.add_argument('--dataset-list', required=True, type=str,
                        help='Path to txt file listing dataset folders')
    parser.add_argument('--dataset-root', default='/home/xietao/timesnet_rknn', type=str)
    parser.add_argument('--stats', required=True, type=Path,
                        help='dataset_stats.json generated during training')
    parser.add_argument('--segments-per-record', type=int, default=5)
    parser.add_argument('--rknn', type=Path, help='Path to exported RKNN model (for on-device eval)')
    parser.add_argument('--tflite', type=Path, help='TFLite path (required for simulator evaluation)')
    parser.add_argument('--calibration', type=Path,
                        help='Calibration dataset list used during quantization (required for simulator eval)')
    parser.add_argument('--target', default='rv1106',
                        help='Target platform when running on device (e.g., rv1106, rk3588)')
    parser.add_argument('--device-id', default=None, help='Optional device id when multiple boards connected')
    parser.add_argument('--use-simulator', action='store_true',
                        help='Force simulator mode by rebuilding from TFLite (ignores --rknn)')
    parser.add_argument('--data-format', choices=['nchw', 'nhwc'], default='nchw',
                        help='Input layout supplied to rknn.inference')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of evaluation samples')
    parser.add_argument('--metrics-out', type=Path, help='Optional path to dump metrics JSON')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def load_stats(stats_path: Path) -> Dict:
    stats = json.loads(stats_path.read_text())
    required = ['mean', 'std', 'classes', 'window_size', 'num_features']
    for key in required:
        if key not in stats:
            raise ValueError(f'missing "{key}" in {stats_path}')
    return stats


def build_simulator(args: argparse.Namespace, stats: Dict) -> RKNN:
    if args.tflite is None or args.calibration is None:
        raise ValueError('--tflite and --calibration are required for simulator evaluation')
    rknn = RKNN(verbose=args.verbose)
    config_kwargs = dict(
        target_platform=args.target,
        mean_values=[stats['mean']],
        std_values=[stats['std']],
        optimization_level=3,
    )
    config_kwargs['quantized_dtype'] = 'asymmetric_quantized-8'
    ret = rknn.config(**config_kwargs)
    if ret != 0:
        raise RuntimeError(f'rknn.config failed ({ret})')
    ret = rknn.load_tflite(model=str(args.tflite))
    if ret != 0:
        raise RuntimeError(f'rknn.load_tflite failed ({ret})')
    ret = rknn.build(do_quantization=True, dataset=str(args.calibration))
    if ret != 0:
        raise RuntimeError(f'rknn.build failed ({ret})')
    ret = rknn.init_runtime(target=None)
    if ret != 0:
        raise RuntimeError(f'rknn.init_runtime failed ({ret})')
    return rknn


def build_runtime(args: argparse.Namespace) -> RKNN:
    if args.rknn is None:
        raise ValueError('--rknn is required when not using simulator mode')
    rknn = RKNN(verbose=args.verbose)
    ret = rknn.load_rknn(str(args.rknn))
    if ret != 0:
        raise RuntimeError(f'rknn.load_rknn failed ({ret})')
    ret = rknn.init_runtime(target=args.target, device_id=args.device_id)
    if ret != 0:
        raise RuntimeError(f'rknn.init_runtime failed ({ret})')
    return rknn


def sample_to_tensor(sample: np.ndarray, data_format: str) -> np.ndarray:
    if data_format == 'nchw':
        tensor = np.transpose(sample, (1, 0))  # (features, window)
        tensor = tensor[:, :, np.newaxis]
        tensor = np.expand_dims(tensor, axis=0)
    else:  # nhwc
        tensor = sample[:, np.newaxis, :]  # (window, 1, features)
        tensor = np.expand_dims(tensor, axis=0)
    return tensor.astype(np.float32)


def evaluate(args: argparse.Namespace) -> Dict:
    stats = load_stats(args.stats)
    class_names = [cls.lower() for cls in stats['classes']]
    window_size = stats['window_size']
    segments_per_record = args.segments_per_record

    print('[info] Loading evaluation dataset...')
    data, labels = prepare_data(
        list_path=args.dataset_list,
        dataset_root=Path(args.dataset_root),
        class_names=class_names,
        window_size=window_size,
        segments_per_record=segments_per_record,
    )
    if args.limit is not None:
        data = data[:args.limit]
        labels = labels[:args.limit]
    print(f'[info] Loaded {data.shape[0]} samples, window_size={window_size}, channels={data.shape[-1]}')

    if args.use_simulator or args.rknn is None:
        rknn = build_simulator(args, stats)
        mode = 'simulator'
    else:
        rknn = build_runtime(args)
        mode = 'runtime'

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    predictions = []
    for idx, (sample, label) in enumerate(zip(data, labels)):
        tensor = sample_to_tensor(sample, args.data_format)
        outputs = rknn.inference(inputs=[tensor], data_format=args.data_format)
        logits = outputs[0].reshape(-1)
        pred = int(np.argmax(logits))
        confusion[label, pred] += 1
        predictions.append(pred)
        if (idx + 1) % 100 == 0:
            print(f'[info] Processed {idx + 1} / {data.shape[0]} samples')

    rknn.release()

    total = confusion.sum()
    correct = np.trace(confusion)
    accuracy = float(correct / total)
    per_class_acc = {}
    for idx, name in enumerate(class_names):
        denom = confusion[idx].sum()
        acc = float(confusion[idx, idx] / denom) if denom > 0 else 0.0
        per_class_acc[name] = acc

    metrics = {
        'mode': mode,
        'target': args.target if mode == 'runtime' else 'simulator',
        'overall_accuracy': accuracy,
        'num_samples': int(total),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion.tolist(),
    }
    return metrics


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    print('\n=== Evaluation Summary ===')
    print(f"Mode        : {metrics['mode']}")
    print(f"Target      : {metrics['target']}")
    print(f"Samples     : {metrics['num_samples']}")
    print(f"Accuracy    : {metrics['overall_accuracy']:.4f}")
    print('Per-class accuracy:')
    for cls, acc in metrics['per_class_accuracy'].items():
        print(f"  {cls:>12s}: {acc:.4f}")
    if args.metrics_out:
        args.metrics_out.write_text(json.dumps(metrics, indent=2))
        print(f"[info] Metrics saved to {args.metrics_out}")


if __name__ == '__main__':
    main()
