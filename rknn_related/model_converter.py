from rknn.api import RKNN
import numpy as np
import argparse
import os

DATASET_PATH = "./deploy/quantize_dataset.txt"
MODEL_PATH = "./deploy/trained_quantcn_8_sensors.onnx"
OUT_RKNN_PATH = "./deploy"
TEST_ON_DEVICE = True

def create_argument_parser():
    """Create and configure argument parser for training."""
    parser = argparse.ArgumentParser(description='Train TCN for joint moment estimation')
    parser.add_argument('--int16', action="store_true", default=False,
                        help='int16 quantize')
    return parser

if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    rknn = RKNN(verbose=True, verbose_file="log/convert.log")
    OUT_RKNN_PATH = os.path.join(OUT_RKNN_PATH, "torque_quantcn_8_sensors_int16.rknn" if args.int16 else "torque_quantcn_8_sensors_int8.rknn")
    rknn.config(mean_values=[[0.7113, -0.4300, 0.7254, 2.6529, 8.7143, -0.2818, -30.6663, -0.1348]],
                std_values=[[21.3148, 45.2281, 81.4140, 3.9808, 4.4334, 1.8336, 27.8284, 107.1712]],
                quantized_dtype="w16a16i_dfp" if args.int16 else "w8a8",  # w8a8,w16a16i_dfp
                quantized_algorithm="mmse" if args.int16 else "mmse",  # normal，mmse，kl_divergence
                quantized_method="channel",
                target_platform="rv1106",
                )
    
    ret = rknn.load_onnx(model=MODEL_PATH)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
    
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH, auto_hybrid=True)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    print('--> Export rknn model')
    ret = rknn.export_rknn(OUT_RKNN_PATH)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print(f"rknn model has saved in {OUT_RKNN_PATH}")
    print('done')
    rknn.release()