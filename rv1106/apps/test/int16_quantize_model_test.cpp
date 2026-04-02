#include "model_process/model_process.h"
#include "rknn/rknn_api.h"
#include <vector>

int main(int argc ,char **argv) {
    if (argc < 2) {
        printf("Usage: %s in16_model_path\n", argv[0]);
        return 0;
    }
    char const *model_path = argv[1];
    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, const_cast<char*>(model_path), input_attrs, output_attrs);
	if (ret != RKNN_SUCC) {
		printf("model info parse failed!\n");
		return -1;
	}
    input_attrs[0].pass_through = 1;
    std::vector<rknn_tensor_mem*> input_mems;
	std::vector<rknn_tensor_mem*> output_mems;
	ret = alllocate_set_io_memory(ctx, input_attrs, output_attrs, input_mems, output_mems);
	if (ret != RKNN_SUCC) {
		printf("alllocate_set_io_memory fail! ret=%d\n", ret);
		resource_destroy(ctx, input_mems, output_mems);
		return -1;
	}
    return 0;
}