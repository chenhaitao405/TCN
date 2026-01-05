from rknn.api import RKNN
import numpy as np

DATASET_PATH = "test_check/quant_list.txt"
MODEL_PATH = "test_check/simple_model.onnx"
OUT_RKNN_PATH = "test_check/simple_model.rknn"
TEST_ON_DEVICE = True

if __name__ == "__main__":
    rknn = RKNN(verbose=True, verbose_file="log/convert.log")
    rknn.config(mean_values=[[0]*8],
                std_values=[[255]*8],
                quantized_dtype="w8a8",
                quantized_algorithm="normal",
                quantized_method="channel",
                target_platform="rv1106",
                quant_img_RGB2BGR=False
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
    
    # rknn = RKNN()
    # rknn.load_rknn(OUT_RKNN_PATH)
    # if TEST_ON_DEVICE:
    #     ret = rknn.init_runtime(target="rv1106")
    # else:
    #     ret = rknn.init_runtime(target=None)
    
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('done')
    
    # data_input = np.load("test_check/quan_data/data_0000.npy")
    
    # print(data_input.shape, data_input.dtype)
    # outputs = rknn.inference(inputs=[data_input], data_format="nchw")

    # print(outputs)
    # rknn.release()