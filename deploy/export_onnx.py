import torch

import os.path as osp
import inspect
import sys
sys.path.append(".")
from utils.tcn import TCN, QuanTCN
model_path = "./models/trained_quantcn_8_sensors.tar"
save_path = "./deploy/trained_quantcn_8_sensors.onnx"

model_info = torch.load(model_path)
model_info = torch.load(model_path, map_location="cpu")
state_dict = model_info["state_dict"]
del model_info["state_dict"]
tcn_signature = inspect.signature(QuanTCN.__init__)
tcn_param_names = [param.name for param in tcn_signature.parameters.values()
                    if param.name != 'self']

# Only pass parameters that TCN needs
tcn_params = {k: v for k, v in model_info.items()
                if k in tcn_param_names}
tcn = QuanTCN(**tcn_params)
tcn.load_state_dict(state_dict)
tcn.eval()  # 设置为评估模式
inputs = torch.randn((1,8,280), dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(tcn,
                      inputs,
                      save_path,
                      export_params=True,
                      opset_version=11,
                      dynamic_axes=None,
                      do_constant_folding=True,
                      input_names=["sensor_inputs"],
                      output_names=["torque"]
                      )


# 删除输入量化节点
if isinstance(tcn, QuanTCN):
    import onnx
    from onnx import helper, numpy_helper
    start_tensor_name = "sensor_inputs"
    model = onnx.load(save_path)
    graph = model.graph
    # for model_input in graph.input:
    #     if model_input.name == start_tensor_name:
    #         model_input.type.tensor_type.elem_type = 2
    
    chain_ops = ["Div", "Add", "Round", "Clip","Div", "Sub", "Div"]
    nodes_to_remove = []
    current_tensor = start_tensor_name
    for op_type in chain_ops:
        found = False
        for node in graph.node:
            if node.op_type == op_type and current_tensor in node.input:
                nodes_to_remove.append(node)
                current_tensor = node.output[0]
                found = True
                break
        if not found:
            print(f"错误：链路中断，未找到类型为 {op_type} 的后续节点。")
            exit()
    print(f"链路追踪完成。截断点张量名为: {current_tensor}")
    
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == current_tensor:
                node.input[i] = start_tensor_name
                
    for node in nodes_to_remove:
        graph.node.remove(node)
        
    onnx.checker.check_model(model)
    from onnxsim import simplify
    print("正在使用 onnxsim 简化模型...")
    # simplify 会自动执行：常量折叠、死代码消除、算子融合
    model, check = simplify(model)
    if not check:
        print("onnxsim 校验失败，但仍将保存模型")
    onnx.save(model, save_path)
    print(f"处理完成，模型已保存至 {save_path}")