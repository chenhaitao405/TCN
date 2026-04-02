#pragma once
#include <vector>
#include "rknn/rknn_api.h"
#include <stdio.h>
#include "data_stream.h"
namespace Hip{
    bool checkModelInfo(std::vector<rknn_tensor_attr> &input_attrs) {
        if (input_attrs[0].type != RKNN_TENSOR_INT8 && input_attrs[0].type != RKNN_TENSOR_INT16)
        {
            printf("Model Input data type should be int8 or int16!\n");
            return false;
        }

        if (input_attrs[0].fmt != RKNN_TENSOR_NC1HWC2) {
            printf("Model Input data format should be NC1HWC2");
            return false;
        }

        int batch = input_attrs[0].dims[0];
        // if (input_attrs[0].dims[1] != 8) {
        //     return false;
        // }
        int h = input_attrs[0].dims[2];
        int w = input_attrs[0].dims[3];
        int expected_frames = h * w;
        if (batch != 1) {
            printf("Model Batch Dimension should be 1!\n");
            return false;
        }
        if (expected_frames != Hip::STREAM_LENGTH) {
            printf("Hip stream length set wrong! Check model setting and data_stream.h!\n");
            return false;
        }
        if (input_attrs[0].scale <= 0.f) {
            printf("quantize scale is less than zero, wrong setting!\n");
            return false;
        }
        return true;
    }
}

namespace Knee{
    bool checkModelInfo(std::vector<rknn_tensor_attr> &input_attrs) {
        if (input_attrs[0].type != RKNN_TENSOR_INT8 && input_attrs[0].type != RKNN_TENSOR_INT16)
        {
            printf("Model Input data type should be int8 or int16!\n");
            return false;
        }

        if (input_attrs[0].fmt != RKNN_TENSOR_NC1HWC2) {
            printf("Model Input data format should be NC1HWC2");
            return false;
        }

        int batch = input_attrs[0].dims[0];
        // if (input_attrs[0].dims[1] != 8) {
        //     return false;
        // }
        int h = input_attrs[0].dims[2];
        int w = input_attrs[0].dims[3];
        int expected_frames = h * w;

        if (batch != 1) {
            printf("Model Batch Dimension should be 1!\n");
            return false;
        }

        if (expected_frames != Knee::STREAM_LENGTH) {
            printf("Knee stream length set wrong! Check model setting and data_stream.h!\n");
            return false;
        }

        if (input_attrs[0].scale <= 0.f) {
            printf("quantize scale is less than zero, wrong setting!\n");
            return false;
        }
        return true;
    }
}
