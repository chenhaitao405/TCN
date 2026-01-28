# Train + export 
python /home/xietao/timesnet_rknn/train_modes_tflite.py \
  --output-dir /home/xietao/timesnet_rknn/artifacts/latest_run \
  --epochs 5 --batch-size 64

# Convert to RKNN 
python /home/xietao/timesnet_rknn/convert_to_rknn.py \
  --tflite /home/xietao/timesnet_rknn/artifacts/latest_run/modes_cnn_fp32.tflite \
  --stats /home/xietao/timesnet_rknn/artifacts/latest_run/dataset_stats.json \
  --dataset /home/xietao/timesnet_rknn/artifacts/latest_run/calibration_samples/calibration_list.txt \
  --output /home/xietao/timesnet_rknn/artifacts/latest_run/modes_cnn_int8.rknn \
  --target rv1106 --dtype i8

# Quantized accuracy via simulator
python /home/xietao/timesnet_rknn/test_rknn_runtime.py \
  --dataset-list /home/xietao/timesnet_rknn/1208_lehiuju_val.txt \
  --dataset-root /home/xietao/timesnet_rknn \
  --stats /home/xietao/timesnet_rknn/artifacts/latest_run/dataset_stats.json \
  --tflite /home/xietao/timesnet_rknn/artifacts/latest_run/modes_cnn_fp32.tflite \
  --calibration /home/xietao/timesnet_rknn/artifacts/latest_run/calibration_samples/calibration_list.txt \
  --segments-per-record 5 --use-simulator \
  --metrics-out /home/xietao/timesnet_rknn/artifacts/latest_run/rknn_eval.json


