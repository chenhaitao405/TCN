from rknn.api import RKNN
import numpy as np

DATASET_PATH = "./deploy/kneedata_left.txt"
MODEL_PATH = "./deploy/model_knee_manual_windows.onnx"
OUT_RKNN_PATH = "./deploy/model_knee_manual_windows.rknn"
TEST_ON_DEVICE = True

if __name__ == "__main__":
    rknn = RKNN(verbose=True, verbose_file="log/convert.log")
    rknn.config(mean_values=[0.7113392353057861, -0.42998769879341125, 0.7253568172454834, 2.6528778076171875, 8.714276313781738, -0.28184762597084045, -30.666257858276367, -0.13483266532421112],
                std_values=[21.314773559570312, 45.228126525878906, 81.4139633178711, 3.980790376663208, 4.433416366577148, 1.8335528373718262, 27.82839012145996, 107.17118835449219],
                quantized_dtype="w8a8",
                quantized_algorithm="kl_divergence",
                quantized_method="channel",
                target_platform="rv1106"
                )
    
    ret = rknn.load_onnx(model=MODEL_PATH, inputs=["sensor_inputs"], input_size_list=[[1,8,248]])
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
    
    if TEST_ON_DEVICE:
        ret = rknn.init_runtime(target="rv1106")
    else:
        ret = rknn.init_runtime()
    
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    data_input = np.load("./deploy/dataset/247.npy").astype(np.float32)
    # if TEST_ON_DEVICE:
    data_input = np.round((data_input / 4.595620) + 1).astype(np.uint8)
    outputs = rknn.inference(inputs=[data_input])
    # if not TEST_ON_DEVICE:
    #     outputs = (outputs[0] - 69) * 0.172141
    print(outputs)
    rknn.release()