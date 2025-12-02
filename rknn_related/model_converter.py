from rknn.api import RKNN
import numpy as np

DATASET_PATH = "./deploy/quantize_dataset.txt"
MODEL_PATH = "./deploy/trained_quantcn_8_sensors.onnx"
OUT_RKNN_PATH = "./deploy/trained_quantcn_8_sensors.rknn"
TEST_ON_DEVICE = True

if __name__ == "__main__":
    rknn = RKNN(verbose=True, verbose_file="log/convert.log")
    rknn.config(mean_values=[[0, 0, 0, 0, 0, 0, 0, 0]],
                std_values=[[255, 255, 255, 255, 255, 255, 255, 255]],
                quantized_dtype="w8a8",
                quantized_algorithm="mmse",
                quantized_method="channel",
                target_platform="rv1106"
                )
    
    ret = rknn.load_onnx(model=MODEL_PATH)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
    
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    print('--> Export rknn model')
    ret = rknn.export_rknn(OUT_RKNN_PATH)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    rknn.release()
    
    rknn = RKNN()
    rknn.load_rknn(OUT_RKNN_PATH)
    if TEST_ON_DEVICE:
        ret = rknn.init_runtime(target="rv1106")
    else:
        ret = rknn.init_runtime(target=None)
    
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    data_input = np.load("deploy/datasets/normal_walk_2-5_1.npy").astype(np.float32)
    data_input = data_input / np.array([0.2152, 0.3451, 0.7216, 0.0459, 0.0549, 0.0193, 0.3545, 0.9591]).reshape(1,-1,1) +\
        np.array([117., 128., 128.,  44.,   0., 158., 251., 123.]).reshape(1,-1,1)
    data_input = np.clip(np.round(data_input), 0, 255).astype(np.uint8)
    data_input = data_input.transpose(0,2,1)
    # data_input = data_input / 255.0
    # data_input -= np.array([0.476752, 0.495703, 0.492770, 0.430457, 0.581848, 0.579756, 0.618145, 0.492519]).reshape(1,-1,1)
    # data_input /= np.array([0.234370, 0.237314, 0.299588, 0.317835, 0.239470, 0.259621, 0.324524, 0.275043]).reshape(1,-1,1)
    
    # rknn.accuracy_analysis(inputs=[data_input], target=None)
    # if TEST_ON_DEVICE:
    outputs = rknn.inference(inputs=[data_input], data_format="nhwc")
    # if not TEST_ON_DEVICE:
    #     outputs = (outputs[0] - 69) * 0.172141
    print(outputs)
    rknn.release()