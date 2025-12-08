import argparse
import torch
import torch.nn as nn
import os


class SimpleModel(nn.Module):
    def __init__(self, in_channels):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="导出简单模型为ONNX")
    parser.add_argument("--channel", type=int, default=3, help="输入通道数")
    parser.add_argument("--height", type=int, default=224, help="输入高度")
    parser.add_argument("--width", type=int, default=224, help="输入宽度")
    parser.add_argument("--output", type=str, default="simple_model.onnx", help="输出文件名")
    args = parser.parse_args()

    # 创建模型并设置为评估模式
    model = SimpleModel(args.channel)
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(1, args.channel, args.height, args.width)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 导出为ONNX
    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(base_dir, args.output),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None
    )

    print(f"模型已导出: {args.output}")
    print(f"输入形状: [1, {args.channel}, {args.height}, {args.width}]")
    print(f"输出形状: [1, 8, {args.height}, {args.width}]")


if __name__ == "__main__":
    main()
