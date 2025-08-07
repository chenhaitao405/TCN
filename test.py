import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# Chomp1d类定义，用于移除卷积后的填充
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# TemporalBlock类定义
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, dropout_type='Dropout',
                 activation='ReLU', norm='weight_norm'):
        super(TemporalBlock, self).__init__()

        self.chomp1 = Chomp1d(padding)
        self.af1 = getattr(nn, activation)()
        self.dropout1 = getattr(nn, dropout_type)(dropout)

        self.chomp2 = Chomp1d(padding)
        self.af2 = getattr(nn, activation)()
        self.dropout2 = getattr(nn, dropout_type)(dropout)

        if norm == 'weight_norm':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.net = nn.Sequential(self.conv1, self.chomp1, self.af1, self.dropout1,
                                     self.conv2, self.chomp2, self.af2, self.dropout2)
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.norm1 = getattr(nn, norm)(n_outputs)

            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.norm2 = getattr(nn, norm)(n_outputs)

            self.net = nn.Sequential(self.conv1, self.norm1, self.chomp1, self.af1, self.dropout1,
                                     self.conv2, self.norm2, self.chomp2, self.af2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.af = getattr(nn, activation)()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.af(out + res)


# 测试不同长度的输入
def test_temporal_block():
    # 初始化模型
    model = TemporalBlock(
        n_inputs=22,
        n_outputs=64,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=2  # 确保序列长度不变
    )

    # 测试不同长度的输入
    for seq_len in [100, 1000, 10000, 318570]:
        x = torch.randn(4, 22, seq_len)  # 创建随机输入
        try:
            output = model(x)
            print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
        except Exception as e:
            print(f"处理长度为 {seq_len} 的序列时出错: {e}")


# 运行测试
test_temporal_block()
