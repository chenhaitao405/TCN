#pragma once
#include "rknn/rknn_api.h"
#include <vector>

void resource_destroy(rknn_context& ctx, std::vector<rknn_tensor_mem *> input_mems, std::vector<rknn_tensor_mem *> output_mems)
{
    for (rknn_tensor_mem* & input_mem : input_mems) {
        if (input_mem)
            rknn_destroy_mem(ctx, input_mem);
    }
    input_mems.clear();
    for (rknn_tensor_mem* & output_mem : output_mems) {
        if (output_mem)
            rknn_destroy_mem(ctx, output_mem);
    }
    output_mems.clear();
    rknn_destroy(ctx);
}

int alllocate_set_io_memory(rknn_context &ctx, std::vector<rknn_tensor_attr> &input_attrs, std::vector<rknn_tensor_attr> &output_attrs,
                        std::vector<rknn_tensor_mem *> &input_mems, std::vector<rknn_tensor_mem *> &output_mems) {
    input_mems.reserve(input_attrs.size());
    input_mems.assign(input_attrs.size(), nullptr);
    output_mems.reserve(output_attrs.size());
    output_mems.assign(output_attrs.size(), nullptr);
    int ret;
    for (int i = 0; i< input_attrs.size(); ++i) {
        input_mems[i] = rknn_create_mem(ctx, input_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, input_mems[i], &input_attrs[i]);
        if (ret != RKNN_SUCC) return ret;
    }
    for (int i = 0; i < output_attrs.size(); ++i) {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret != RKNN_SUCC) return ret;
    }

    return RKNN_SUCC;
}