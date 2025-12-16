'''
This code implements the temporal convolutional network (TCN) class used in the study "Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation."

This code was modified from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py.
Original License: MIT License
Copyright (c) 2018 CMU Locus Lab
'''

from typing import List
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, dropout_type='Dropout', activation='ReLU', norm='weight_norm'):
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


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dropout_type='Dropout', activation='ReLU', norm='weight_norm'):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i 
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, 
				padding=(kernel_size-1) * dilation_size, dropout=dropout, dropout_type=dropout_type,
				activation=activation, norm='weight_norm')]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


class TCN(nn.Module):
	'''Implements the temporal convolutional network used in this study.'''
	def __init__(self, 
					input_size: int, 
					output_size: int, 
					num_channels: List[int], 
					ksize: int, 
					dropout: float, 
					eff_hist: int, 
					spatial_dropout: bool = False, 
					activation: str = 'ReLU', 
					norm: str = 'weight_norm', 
					center: float = 0., 
					scale: float = 1.):
		super(TCN, self).__init__()

		# create and initialize network
		self.dropout_type = 'Dropout'
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=ksize, dropout=dropout, dropout_type=self.dropout_type, activation=activation, norm=norm)
		self.linear = nn.Linear(num_channels[-1], output_size)
		self.init_weights()
		self.eff_hist = eff_hist

		# save for input feature normalization
		self.register_buffer("center", center)
		self.register_buffer("scale", scale)
		
	def init_weights(self):
		self.linear.weight.data.normal_(0, 0.01)

	def forward(self, x):
		# normalize input features
		out = (x - self.center) / self.scale

		# forward pass of conv layers
		out = self.tcn(out)

		# reshape for final linear layer
		out = torch.cat([out[i, :, :] for i in range(out.shape[0])], dim = 1).transpose(0, 1).contiguous()

		# forward pass of final linear layer
		out = self.linear(out).transpose(0, 1)

		# reshape back to original format
		out = torch.cat([out[:, i*x.shape[2]:(i+1)*x.shape[2]].unsqueeze(0) for i in range(x.shape[0])], dim = 0)

		return out

	def get_effective_history(self):
		return self.eff_hist


    
    
class CausalPad2d(nn.Module):
    def __init__(self, padding):
        super(CausalPad2d, self).__init__()
        self.padding = padding

    def forward(self, x):
        # 使用 F.pad，padding 格式: (left, right, top, bottom)
        # 对于 (N, C, 1, T)
        return torch.nn.functional.pad(x, (self.padding, 0, 0, 0), mode='constant', value=0)
    
# class CausalPad2d(nn.Module):
# 	def __init__(self, padding):
# 		super(CausalPad2d, self).__init__()
# 		# padding: (left, right, top, bottom) for last two dims
# 		# 对于 (N, C, T, 1)，我们在 T 维度左侧填充
# 		self.pad = nn.ConstantPad2d((0, 0, padding, 0), 0)

# 	def forward(self, x):
# 		return self.pad(x)

class QuanTemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, dropout_type='Dropout', activation='ReLU', norm='BatchNorm2d'):
		super(QuanTemporalBlock, self).__init__()

		self.pad1 = CausalPad2d(padding)
		self.af1 = getattr(nn, activation)()
		self.dropout1 = getattr(nn, dropout_type)(dropout)

		self.pad2 = CausalPad2d(padding)
		self.af2 = getattr(nn, activation)()
		self.dropout2 = getattr(nn, dropout_type)(dropout)

		if norm.lower().startswith("batchnorm") or norm.lower().startswith("batch_norm"):
			self.conv1 = nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
				stride=(1, stride), padding=(0, 0), dilation=(1, dilation), bias=True)
			self.norm1 = nn.BatchNorm2d(n_outputs)

			self.conv2 = nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
				stride=(1, stride), padding=(0, 0), dilation=(1, dilation), bias=True)
			self.norm2 = nn.BatchNorm2d(n_outputs)

			self.net = nn.Sequential(
				self.pad1, self.conv1, self.norm1, self.af1, self.dropout1,
				self.pad2, self.conv2, self.norm2, self.af2, self.dropout2
			)
			self.downsample = nn.Sequential(
				nn.Conv2d(n_inputs, n_outputs, (1, 1), bias=True),
				nn.BatchNorm2d(n_outputs)
			) if n_inputs != n_outputs else None
   
		elif norm.lower().startswith("weightnorm") or norm.lower().startswith("weight_norm"):
			self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
				stride=(1, stride), padding=(0, 0), dilation=(1, dilation), bias=True))

			self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
				stride=(1, stride), padding=(0, 0), dilation=(1, dilation), bias=True))
			self.net = nn.Sequential(
				self.pad1, self.conv1, self.af1, self.dropout1,
				self.pad2, self.conv2, self.af2, self.dropout2
			)
			self.downsample = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, 1), bias=True)) if n_inputs != n_outputs else None
		
		self.af = getattr(nn, activation)()
		self.init_weights()

	def init_weights(self):
		nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
		if self.downsample is not None:
			nn.init.kaiming_normal_(self.downsample[0].weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)		
		return self.af(out + res)


# class QuanTemporalBlock(nn.Module):
# 	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, dropout_type='Dropout', activation='ReLU', norm='BatchNorm2d'):
# 		super(QuanTemporalBlock, self).__init__()

# 		self.pad1 = CausalPad2d(padding)
# 		self.af1 = getattr(nn, activation)()
# 		self.dropout1 = getattr(nn, dropout_type)(dropout)

# 		self.pad2 = CausalPad2d(padding)
# 		self.af2 = getattr(nn, activation)()
# 		self.dropout2 = getattr(nn, dropout_type)(dropout)

# 		# Conv2d 不再需要 padding，因为已经通过 CausalPad2d 处理
# 		self.conv1 = nn.Conv2d(n_inputs, n_outputs, (kernel_size, 1),
# 			stride=(stride, 1), padding=(0, 0), dilation=(dilation, 1), bias=False)
# 		self.norm1 = nn.BatchNorm2d(n_outputs)

# 		self.conv2 = nn.Conv2d(n_outputs, n_outputs, (kernel_size, 1),
# 			stride=(stride, 1), padding=(0, 0), dilation=(dilation, 1), bias=False)
# 		self.norm2 = nn.BatchNorm2d(n_outputs)

# 		self.net = nn.Sequential(
# 			self.pad1, self.conv1, self.norm1, self.af1, self.dropout1,
# 			self.pad2, self.conv2, self.norm2, self.af2, self.dropout2
# 		)

# 		self.downsample = nn.Sequential(
# 			nn.Conv2d(n_inputs, n_outputs, (1, 1), bias=False),
# 			nn.BatchNorm2d(n_outputs)
# 		) if n_inputs != n_outputs else None
		
# 		self.af = getattr(nn, activation)()
# 		self.init_weights()

# 	def init_weights(self):
# 		nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
# 		nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
# 		if self.downsample is not None:
# 			nn.init.kaiming_normal_(self.downsample[0].weight, mode='fan_out', nonlinearity='relu')

# 	def forward(self, x):
# 		out = self.net(x)
# 		res = x if self.downsample is None else self.downsample(x)		
# 		return self.af(out + res)


class QuanTemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dropout_type='Dropout', activation='ReLU', norm='BatchNorm2d'):
		super(QuanTemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [QuanTemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, 
				padding=(kernel_size-1) * dilation_size, dropout=dropout, dropout_type=dropout_type,
				activation=activation, norm=norm)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


# class QuanTCN(nn.Module):
# 	'''Implements the temporal convolutional network used in this study.'''
# 	def __init__(self, 
# 					input_size: int, 
# 					output_size: int, 
# 					num_channels: List[int], 
# 					ksize: int, 
# 					dropout: float, 
# 					eff_hist: int, 
# 					spatial_dropout: bool = False, 
# 					activation: str = 'ReLU', 
# 					norm: str = 'BatchNorm2d'):
# 		super(QuanTCN, self).__init__()

# 		# create and initialize network
# 		self.dropout_type = 'Dropout2d' if spatial_dropout else 'Dropout'
# 		self.tcn = QuanTemporalConvNet(input_size, num_channels, kernel_size=ksize, dropout=dropout, dropout_type=self.dropout_type, activation=activation, norm=norm)
# 		self.linear = nn.Linear(num_channels[-1], output_size)
# 		self.init_weights()
# 		self.eff_hist = eff_hist
  
# 		# norm_means = torch.tensor([0.476752, 0.495703, 0.492770, 0.430457, 0.581848, 0.579756, 0.618145, 0.492519]).reshape(1,-1,1)
#         # norm_stds = torch.tensor([0.234370, 0.237314, 0.299588, 0.317835, 0.239470, 0.259621, 0.324524, 0.275043]).reshape(1,-1,1)
        
# 		# norm_means = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.]).reshape(1,-1,1,1)
# 		# norm_stds = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1.]).reshape(1,-1,1,1)
# 		# self.register_buffer("mean", norm_means)
# 		# self.register_buffer("std", norm_stds)
  
# 		self.quan_max = 2 ** 8 -1
# 		self.quan_min = 0
# 		scales = torch.tensor([0.2152, 0.3451, 0.7216, 0.0459, 0.0549, 0.0193, 0.3545, 0.9591],
#                               dtype=torch.float32).reshape(1,-1,1,1)
# 		self.register_buffer("scales", scales)
# 		zeros = torch.tensor([117., 128., 128.,  44.,   0., 158., 251., 123.], dtype=torch.float32).reshape(1,-1,1,1)
# 		self.register_buffer("zeros", zeros)

# 	def init_weights(self):
# 		self.linear.weight.data.normal_(0, 0.01)

# 	def quantize_input(self, x):
# 		x = x / self.scales + self.zeros
# 		x = torch.clamp(torch.round(x), self.quan_min, self.quan_max)
# 		return x / 255.0

# 	def forward(self, x):
# 		x = x.unsqueeze(-2)
# 		# x shape: (N, C, 1, T)
# 		x = self.quantize_input(x)
  
# 		# normalize input features
# 		# out = (x - self.mean) / self.std

# 		out = self.tcn(x)

# 		# out shape: (N, C', 1, T) -> (N, 1, T, C')
# 		out = out.permute(0,2,3,1)

# 		# forward pass of final linear layer
# 		out = self.linear(out)

# 		# reshape back to original format
# 		out = out.permute(0,3,1,2)

# 		return out

# 	def get_effective_history(self):
# 		return self.eff_hist


class QuanTCN(nn.Module):
	'''Implements the temporal convolutional network used in this study.'''
	def __init__(self, 
					input_size: int, 
					output_size: int, 
					num_channels: List[int], 
					ksize: int, 
					dropout: float, 
					eff_hist: int, 
					spatial_dropout: bool = False, 
					activation: str = 'ReLU', 
					norm: str = 'BatchNorm2d'):
		super(QuanTCN, self).__init__()

		# create and initialize network
		self.dropout_type = 'Dropout2d' if spatial_dropout else 'Dropout'
		self.tcn = QuanTemporalConvNet(input_size, num_channels, kernel_size=ksize, dropout=dropout, dropout_type=self.dropout_type, activation=activation, norm=norm)
		self.output_layer = nn.Conv2d(num_channels[-1], output_size, kernel_size=1, padding=0)
		self.init_weights()
		self.eff_hist = eff_hist
  
		norm_means = torch.tensor([0.912629, -0.820086, 0.197228, 3.041710, 8.390683, -0.177071, -33.347044, 2.272569]).reshape(1,-1,1,1)
		norm_stds = torch.tensor([16.687198, 29.007817, 82.961570, 4.110988, 4.230163, 1.639207, 30.426489, 113.209047]).reshape(1,-1,1,1)
		self.register_buffer("mean", norm_means)
		self.register_buffer("std", norm_stds)

	def init_weights(self):
		nn.init.normal_(self.output_layer.weight, mean=0, std=0.1)
		nn.init.zeros_(self.output_layer.bias)

	def forward(self, x):
		# x = x.clone()
		x = x.unsqueeze(-2)  # (N, C, T) -> (N, C, 1, T)
		
		# normalize input features
		out = (x - self.mean) / self.std
		x[:, -3, ...] = torch.clamp(x[:, -3, ...], -26.0, 22.5)
		out = self.tcn(out)

		# out shape: (N, C', 1, T)
		out = self.output_layer(out)

		return out

	def get_effective_history(self):
		return self.eff_hist
