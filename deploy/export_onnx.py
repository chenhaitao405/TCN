import torch

import os.path as osp
import inspect
import sys
sys.path.append(".")
from utils.tcn import TCN
model_path = "./deploy/model_knee_manual_windows.tar"
save_path = "./deploy/model_knee_manual_windows.onnx"

model_info = torch.load(model_path)
model_info = torch.load(model_path, map_location="cpu")
state_dict = model_info["state_dict"]
del model_info["state_dict"]
tcn_signature = inspect.signature(TCN.__init__)
tcn_param_names = [param.name for param in tcn_signature.parameters.values()
                    if param.name != 'self']

# Only pass parameters that TCN needs
tcn_params = {k: v for k, v in model_info.items()
                if k in tcn_param_names}
tcn = TCN(**tcn_params)
tcn.load_state_dict(state_dict)
tcn.eval()  # 设置为评估模式
inputs = torch.randn((1,8,248), dtype=torch.float32)

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