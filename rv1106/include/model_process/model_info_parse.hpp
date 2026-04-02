#pragma once
#include "rknn/rknn_api.h"
#include <stdio.h>
#include <string.h>
#include <vector>

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    char dims[128] = {0};
    for (int i = 0; i < attr->n_dims; ++i)
    {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride=%d, h_stride=%d, size_with_stride=%d,"
           " pass_through=%hhu, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, attr->w_stride, attr->h_stride, attr->size_with_stride,
           attr->pass_through, get_format_string(attr->fmt), get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int model_info_parse(rknn_context &ctx, char *model_path,
                     std::vector<rknn_tensor_attr> &input_attrs,
                     std::vector<rknn_tensor_attr> &output_attrs)
{
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret != RKNN_SUCC)
    {
        printf("rknn init failed! ret=%d\n", ret);
        return ret;
    }

    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    input_attrs.assign(io_num.n_input, {});
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query error! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    printf("output tensors:\n");
    output_attrs.assign(io_num.n_output, {});
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    rknn_custom_string custom_string;
    ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("custom string: %s\n", custom_string.string);

    return RKNN_SUCC;
}
