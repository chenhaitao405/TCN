#!/usr/bin/env python3
"""Validate original (pre-RKNN) model accuracy on local machine."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf

from data_utils import prepare_data


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate original model (TFLite/Keras) on validation dataset"
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to model file (.tflite/.h5/.keras)")
    parser.add_argument("--model-type", type=str, default="auto",
                        choices=["auto", "tflite", "keras"],
                        help="Model type, auto infers from extension")
    parser.add_argument("--val-list", type=str,
                        default="timesnet_rknn/1208_lehiuju_val.txt",
                        help="Path to validation dataset list")
    parser.add_argument("--dataset-root", type=str, default="timesnet_rknn",
                        help="Root directory of dataset")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Window size override, default uses dataset_stats.json")
    parser.add_argument("--segments-per-record", type=int, default=5,
                        help="Number of sampled segments per record")
    parser.add_argument("--stats", type=Path, required=True,
                        help="Path to dataset_stats.json")
    parser.add_argument("--class-names", type=str, default=None,
                        help="Comma separated class names, default uses stats classes")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit sample count for quick checks")
    parser.add_argument("--save-results", type=Path, default=None,
                        help="Optional path to save metrics JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-sample prediction")
    return parser


def load_stats(stats_path: Path) -> Dict:
    stats = json.loads(stats_path.read_text())
    required = ["mean", "std", "classes", "window_size", "num_features"]
    for key in required:
        if key not in stats:
            raise ValueError(f'missing "{key}" in {stats_path}')
    return stats


def preprocess_x(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # (N, W, C) -> normalize -> (N, C, 1, W)
    x = (x_raw - mean) / std
    x = np.transpose(x, (0, 2, 1))
    x = np.expand_dims(x, axis=2)
    return np.ascontiguousarray(x, dtype=np.float32)


def infer_model_type(model_path: Path, model_type: str) -> str:
    if model_type != "auto":
        return model_type
    ext = model_path.suffix.lower()
    if ext == ".tflite":
        return "tflite"
    if ext in {".h5", ".keras"}:
        return "keras"
    raise ValueError(
        f"Cannot infer model type from extension '{ext}'. Please set --model-type explicitly"
    )


def run_tflite_inference(interpreter: tf.lite.Interpreter, sample: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    input_data = np.ascontiguousarray(sample[np.newaxis, ...], dtype=np.float32)
    interpreter.set_tensor(input_detail["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    return np.ravel(output).astype(np.float32)


def run_keras_inference(model: tf.keras.Model, sample: np.ndarray) -> np.ndarray:
    input_data = sample[np.newaxis, ...]
    output = model.predict(input_data, verbose=0)
    return np.ravel(output).astype(np.float32)


def compute_confusion_matrix(predictions: List[int], labels: List[int], num_classes: int) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(predictions, labels):
        confusion[label, pred] += 1
    return confusion


def compute_metrics(confusion: np.ndarray, class_names: Sequence[str]) -> Dict:
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
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "samples": int(confusion[idx, :].sum()),
        }

    return {
        "overall_accuracy": overall_accuracy,
        "total_samples": int(total),
        "correct_predictions": int(correct),
        "per_class": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
    }


def print_summary(metrics: Dict, class_names: Sequence[str]) -> None:
    print("\n" + "=" * 60)
    print("ORIGINAL MODEL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total Samples: {metrics['total_samples']}")
    print(
        f"Overall Accuracy: {metrics['overall_accuracy']:.4f} "
        f"({metrics['correct_predictions']}/{metrics['total_samples']})"
    )
    print("\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
    print("-" * 60)

    for name in class_names:
        m = metrics["per_class"][name]
        print(f"{name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['samples']:<10}")

    print("\nConfusion Matrix:")
    print(f"{'True/Pred':<15}", end="")
    for name in class_names:
        print(f"{name[:10]:<12}", end="")
    print()
    print("-" * (15 + 12 * len(class_names)))

    confusion = np.array(metrics["confusion_matrix"])
    for idx, name in enumerate(class_names):
        print(f"{name:<15}", end="")
        for pred_idx in range(len(class_names)):
            print(f"{confusion[idx, pred_idx]:<12}", end="")
        print()
    print("=" * 60)


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    stats = load_stats(args.stats)
    class_names = [c.strip().lower() for c in args.class_names.split(",")] if args.class_names else [c.lower() for c in stats["classes"]]
    window_size = args.window_size if args.window_size is not None else int(stats["window_size"])

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std = np.clip(std, 1e-6, None)

    print("[info] Loading validation dataset...")
    val_x_raw, val_y = prepare_data(
        args.val_list,
        Path(args.dataset_root),
        class_names,
        window_size,
        args.segments_per_record,
    )

    if args.max_samples and len(val_x_raw) > args.max_samples:
        val_x_raw = val_x_raw[:args.max_samples]
        val_y = val_y[:args.max_samples]
        print(f"[info] Limited to {args.max_samples} samples")

    val_x = preprocess_x(val_x_raw, mean, std)
    model_type = infer_model_type(args.model, args.model_type)

    if model_type == "tflite":
        interpreter = tf.lite.Interpreter(model_path=str(args.model))
        interpreter.allocate_tensors()
        model_obj = interpreter
    else:
        model_obj = tf.keras.models.load_model(args.model)

    print(f"[info] Model: {args.model}")
    print(f"[info] Type: {model_type}")
    print(f"[info] Samples: {len(val_x)}")

    predictions: List[int] = []
    labels: List[int] = []
    for idx in range(len(val_x)):
        sample = val_x[idx]
        if model_type == "tflite":
            logits = run_tflite_inference(model_obj, sample)
        else:
            logits = run_keras_inference(model_obj, sample)

        pred = int(np.argmax(logits))
        label = int(val_y[idx])
        predictions.append(pred)
        labels.append(label)

        if args.debug:
            print(f"sample={idx:04d} true={class_names[label]} pred={class_names[pred]} logits={logits}")

    confusion = compute_confusion_matrix(predictions, labels, len(class_names))
    metrics = compute_metrics(confusion, class_names)
    print_summary(metrics, class_names)

    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        args.save_results.write_text(json.dumps(metrics, indent=2))
        print(f"[info] Saved metrics to {args.save_results}")


if __name__ == "__main__":
    main()
