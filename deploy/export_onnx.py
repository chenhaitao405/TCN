import torch
from onnxsim import simplify
import onnx
from onnx import helper, shape_inference, TensorProto
import os.path as osp
import inspect
import sys
sys.path.append(".")
from utils.tcn import TCN, QuanTCN

model_path = "models/trained_quantcn_8_sensors.tar"
save_path = "./deploy/trained_quantcn_8_sensors.onnx"

model_info = torch.load(model_path, map_location="cpu")
state_dict = model_info["state_dict"]
del model_info["state_dict"]
tcn_signature = inspect.signature(QuanTCN.__init__)
tcn_param_names = [param.name for param in tcn_signature.parameters.values()
                    if param.name != 'self']

tcn_params = {k: v for k, v in model_info.items()
                if k in tcn_param_names}
tcn = QuanTCN(**tcn_params)
tcn.load_state_dict(state_dict, strict=False)
tcn.eval()

inputs = torch.randn((1, 8, 280), dtype=torch.float32)

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

model = onnx.load(save_path)
model_simplified, check = simplify(
    model,
    skipped_optimizers=[
        'fuse_pad_into_conv',
        'fuse_pad_into_pool',
    ]
)

if check:
    onnx.save(model_simplified, save_path)
    print("模型已简化并保存（保留独立 Pad 算子）")
else:
    print("简化失败，保存原始模型")
    exit(0)

# 删除部分量化节点
if isinstance(tcn, QuanTCN):
    start_tensor_name = "sensor_inputs"
    model = onnx.load(save_path)
    graph = model.graph
    
    chain_ops = ["Unsqueeze", "Div", "Add", "Round", "Clip", "Div"]
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
    
    # 修复：记录需要更新的节点，避免迭代时修改
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == current_tensor:
                node.input[i] = start_tensor_name
                print(f"已将节点 {node.name} 的输入从 {current_tensor} 改为 {start_tensor_name}")
                
    for node in nodes_to_remove:
        graph.node.remove(node)
        print(f"已删除节点: {node.op_type}")
    
    # 清理不再使用的 initializer（Div、Add 的常量输入）
    used_inputs = set()
    for node in graph.node:
        used_inputs.update(node.input)
    
    initializers_to_remove = []
    for init in graph.initializer:
        if init.name not in used_inputs:
            initializers_to_remove.append(init)
    
    for init in initializers_to_remove:
        graph.initializer.remove(init)
        print(f"已删除未使用的 initializer: {init.name}")
    
    # 修改输入节点的 shape: (N,C,T) -> (N,C,T,1) 并改为 UINT8 类型
    input_tensor = graph.input[0]

    new_shape = [1, 8, 280, 1]
    
    new_input = helper.make_tensor_value_info(
        input_tensor.name,
        elem_type=TensorProto.FLOAT,  
        shape=new_shape
    )
    
    graph.input.remove(input_tensor)
    graph.input.insert(0, new_input)
    print(f"已修改输入形状为 {new_shape}，类型")
    
    # 重新进行形状推断（重要！）
    try:
        model = shape_inference.infer_shapes(model)
        print("形状推断完成")
    except Exception as e:
        print(f"形状推断失败: {e}，继续处理...")
    
    # 检查模型
    try:
        onnx.checker.check_model(model)
        print("模型检查通过")
    except Exception as e:
        print(f"模型检查警告: {e}")
    
    # 再次简化，但保留 Pad 算子
    print("正在使用 onnxsim 简化模型...")
    model, check = simplify(
        model,
        skipped_optimizers=[
            'fuse_pad_into_conv',  # 保持不融合 Pad
            'fuse_pad_into_pool',
        ]
    )
    if not check:
        print("onnxsim 校验失败，但仍将保存模型")
    
    onnx.save(model, save_path)
    print(f"处理完成，模型已保存至 {save_path}")