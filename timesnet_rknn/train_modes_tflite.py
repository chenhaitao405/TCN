#!/usr/bin/env python3
"""Train a 1-D CNN for motion-mode classification and export a TFLite model."""
import argparse
import json
import os
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from data_utils import prepare_data

# -----------------------------------------------------------------------------
# Determinism and seeds
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

def compute_channel_stats(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feature_axis = (0, 1)
    channel_mean = samples.mean(axis=feature_axis)
    channel_std = samples.std(axis=feature_axis)
    channel_std = np.clip(channel_std, 1e-6, None)
    return channel_mean, channel_std

def build_model(window_size: int, num_features: int, num_classes: int, dropout: float) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(window_size, 1, num_features), name='imu_input')
    x = tf.keras.layers.Reshape((window_size, num_features))(inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name='imu_modes_cnn')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"[info] Saved TFLite model to {output_path}")

def dump_stats(stats_path: Path, mean: np.ndarray, std: np.ndarray, class_names: Sequence[str],
               window_size: int, num_features: int) -> None:
    payload = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'var': (std ** 2).tolist(),
        'classes': list(class_names),
        'window_size': window_size,
        'num_features': num_features
    }
    stats_path.write_text(json.dumps(payload, indent=2))
    print(f"[info] Wrote dataset statistics to {stats_path}")

def make_calibration_dataset(raw_samples: np.ndarray, output_dir: Path,
                             max_samples: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = np.arange(raw_samples.shape[0])
    np.random.shuffle(indices)
    limit = min(max_samples, raw_samples.shape[0])
    dataset_txt = output_dir / 'calibration_list.txt'
    with open(dataset_txt, 'w') as handle:
        for idx in indices[:limit]:
            sample = raw_samples[idx]
            # RKNN calibration expects NCHW (batch, channel, height, width)
            sample = np.transpose(sample, (1, 0))  # (features, window)
            sample = sample[:, :, np.newaxis]      # (features, window, 1)
            sample = np.expand_dims(sample, axis=0)  # (1, features, window, 1)
            sample_path = output_dir / f'sample_{idx:05d}.npy'
            np.save(sample_path, sample.astype(np.float32))
            handle.write(str(sample_path.resolve()) + '\n')
    print(f"[info] Created calibration dataset with {limit} samples -> {dataset_txt}")
    return dataset_txt

def main() -> None:
    parser = argparse.ArgumentParser(description='Train IMU mode classifier and export TFLite model')
    parser.add_argument('--train-list', type=str,
                        default='1208_lehiuju_val.txt')
    parser.add_argument('--val-list', type=str,
                        default='1208_lehiuju_val.txt')
    parser.add_argument('--dataset-root', type=str, default='/home/xietao/timesnet_rknn')
    parser.add_argument('--output-dir', type=str, default='artifacts/latest_run')
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--segments-per-record', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--calibration-samples', type=int, default=256)
    parser.add_argument('--class-names', type=str,
                        default='upstair,downstair,walk,stand',
                        help='Comma separated class names, used as substring match')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    class_names = [cls.strip().lower() for cls in args.class_names.split(',') if cls.strip()]

    print('[info] Loading training set...')
    train_x_raw, train_y = prepare_data(args.train_list, dataset_root, class_names,
                                        args.window_size, args.segments_per_record)
    print('[info] Loading validation set...')
    val_x_raw, val_y = prepare_data(args.val_list, dataset_root, class_names,
                                    args.window_size, args.segments_per_record)

    channel_mean, channel_std = compute_channel_stats(train_x_raw)
    dump_stats(output_dir / 'dataset_stats.json', channel_mean, channel_std,
               class_names, args.window_size, train_x_raw.shape[-1])

    train_x = (train_x_raw - channel_mean) / channel_std
    val_x = (val_x_raw - channel_mean) / channel_std

    train_x = np.expand_dims(train_x, axis=2)
    val_x = np.expand_dims(val_x, axis=2)

    num_classes = len(class_names)
    train_y_one_hot = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
    val_y_one_hot = tf.keras.utils.to_categorical(val_y, num_classes=num_classes)

    model = build_model(args.window_size, train_x.shape[-1], num_classes, args.dropout)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(output_dir / 'best_model.h5'),
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=16, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.CSVLogger(str(output_dir / 'training_log.csv'))
    ]

    history = model.fit(
        train_x, train_y_one_hot,
        validation_data=(val_x, val_y_one_hot),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=2,
        callbacks=callbacks
    )

    metrics = {
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    }

    eval_loss, eval_acc = model.evaluate(val_x, val_y_one_hot, verbose=0)
    metrics['best_val_accuracy'] = float(eval_acc)
    metrics['best_val_loss'] = float(eval_loss)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    best_model_path = output_dir / 'best_model.h5'
    if best_model_path.is_file():
        model = tf.keras.models.load_model(best_model_path)

    tflite_path = output_dir / 'modes_cnn_fp32.tflite'
    export_tflite(model, tflite_path)

    calibration_dir = output_dir / 'calibration_samples'
    make_calibration_dataset(train_x_raw, calibration_dir, args.calibration_samples)

    label_map = {int(idx): name for idx, name in enumerate(class_names)}
    (output_dir / 'labels.json').write_text(json.dumps(label_map, indent=2))
    print('[info] Training pipeline completed successfully.')

if __name__ == '__main__':
    main()
