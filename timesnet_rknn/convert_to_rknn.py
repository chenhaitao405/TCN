#!/usr/bin/env python3
"""Convert the TFLite model to RKNN format with per-channel normalization."""
import argparse
import json
from pathlib import Path
from typing import List

from rknn.api import RKNN


def load_stats(stats_path: Path) -> List[float]:
    stats = json.loads(stats_path.read_text())
    mean = stats['mean']
    std = stats['std']
    if len(mean) != len(std):
        raise ValueError('Mean and std length mismatch')
    return mean, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert TensorFlow Lite model to RKNN')
    parser.add_argument('--tflite', required=True, type=Path, help='Path to TFLite model file')
    parser.add_argument('--stats', required=True, type=Path, help='dataset_stats.json generated during training')
    parser.add_argument('--dataset', required=True, type=Path,
                        help='Text file listing calibration .npy samples (one path per line)')
    parser.add_argument('--output', required=True, type=Path, help='Output RKNN path')
    parser.add_argument('--target', default='rv1106', choices=['rv1103', 'rv1106', 'rv1103b', 'rv1106b'],
                        help='Target hardware platform')
    parser.add_argument('--dtype', default='i8', choices=['i8', 'fp'],
                        help='Quantization dtype (i8 for INT8, fp for float)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tflite.is_file():
        raise FileNotFoundError(args.tflite)
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)

    mean, std = load_stats(args.stats)
    print(mean, std)
    
    do_quant = args.dtype == 'i8'

    rknn = RKNN(verbose=True)

    print('[info] Configuring RKNN context...')
    config_kwargs = dict(
        target_platform=args.target,
        mean_values=[mean],
        std_values=[std],
        optimization_level=3
    )
    if do_quant:
        config_kwargs['quantized_dtype'] = 'w8a8'
        config_kwargs['quantized_algorithm'] = 'mmse'
        config_kwargs['quantized_method'] = 'channel'

    ret = rknn.config(**config_kwargs)
    if ret != 0:
        raise RuntimeError(f'rknn.config failed ({ret})')

    print('[info] Loading TFLite model...')
    ret = rknn.load_tflite(model=str(args.tflite), input_is_nchw=True)
    if ret != 0:
        raise RuntimeError(f'rknn.load_tflite failed ({ret})')

    print('[info] Building RKNN model...')
    ret = rknn.build(do_quantization=do_quant, dataset=str(args.dataset))
    if ret != 0:
        raise RuntimeError(f'rknn.build failed ({ret})')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'[info] Exporting RKNN model to {args.output} ...')
    ret = rknn.export_rknn(str(args.output))
    if ret != 0:
        raise RuntimeError(f'rknn.export_rknn failed ({ret})')

    rknn.release()
    print('[info] RKNN conversion completed successfully.')


if __name__ == '__main__':
    main()
