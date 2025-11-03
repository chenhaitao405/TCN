'''
2D CNN model for single leg hip moment estimation
Input: 2D motor data treated as grayscale image
Output: 1D hip moment
Modified to handle input shape (batch_size, 5, seq_length) and ensure output shape (batch_size, 1, 218)
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Dict, Optional
from torchsummary import summary


class Chomp2d(nn.Module):
    """Removes the padding added by causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Only chomp along the temporal (width) dimensions
        if self.chomp_size > 0:
            return x[:, :, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock2D(nn.Module):
    """Residual block containing two 2D temporal convolutional layers"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, 
                 is_first_block=False, input_height=5):
        super(TemporalBlock2D, self).__init__()
        
        self.is_first_block = is_first_block
        self.input_height = input_height
        
        if is_first_block:
            # First block: reduce height from 5 to 4 (or another appropriate value)
            # Using kernel height of 2 to reduce height by 1
            kernel_2d_first = (2, kernel_size)  
            kernel_2d_second = (1, kernel_size)  
            padding_2d_first = (0, padding)  
            padding_2d_second = (0, padding)
            self.output_height = input_height - 1  # 5 -> 4
        else:
            # Subsequent blocks: maintain height
            kernel_2d_first = (1, kernel_size)
            kernel_2d_second = (1, kernel_size)
            padding_2d_first = (0, padding)
            padding_2d_second = (0, padding)
            self.output_height = input_height
        
        stride_2d = (1, stride)
        dilation_2d = (1, dilation)
        
        # First convolutional layer
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_2d_first,
                                          stride=stride_2d, padding=padding_2d_first, dilation=dilation_2d))
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_2d_second,
                                          stride=stride_2d, padding=padding_2d_second, dilation=dilation_2d))
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection
        if is_first_block:
            # Need to reduce height and potentially change channels
            self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=(2, 1))
        else:
            # Height stays the same, just change channels if needed
            self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights with small random values"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.downsample is not None:
            res = self.downsample(x)
        else:
            res = x
        return self.relu(out + res)


class TemporalConvNet2D(nn.Module):
    """Stack of 2D temporal blocks with exponentially increasing dilation"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, input_height=5):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = len(num_channels)
        current_height = input_height
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            is_first = (i == 0)  # Mark the first block
            
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size, stride=1,
                                      dilation=dilation_size,
                                      padding=(kernel_size-1) * dilation_size,
                                      dropout=dropout,
                                      is_first_block=is_first,
                                      input_height=current_height)]
            
            # Update height after first block
            if is_first:
                current_height = current_height - 1  # 5 -> 4
        
        self.network = nn.Sequential(*layers)
        self.output_height = current_height
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(num_levels, kernel_size)

    def _calculate_receptive_field(self, num_levels, kernel_size):
        """Calculate the receptive field of the TCN"""
        receptive_field = 1
        for i in range(num_levels):
            dilation = 2 ** i
            receptive_field += 2 * (kernel_size - 1) * dilation
        return receptive_field

    def forward(self, x):
        return self.network(x)


class HipMomentCNN2D(nn.Module):
    """
    2D CNN model for single leg hip moment estimation
    
    Architecture:
    1. Treat input as grayscale image (1 channel, height=5, width=seq_len)
    2. 2D CNN layers with exponential dilation for temporal modeling
    3. Height reduction layer before output
    4. Output projection to predict single hip moment
    """
    def __init__(self, 
                 input_size=5,  # Changed from 2 to 5 for your input
                 tcn_channels=[64, 64, 128, 128, 256, 256, 128, 64],
                 kernel_size=2,
                 dropout=0.2,
                 use_input_projection=True,
                 projection_size=32,
                 center=0., 
                 scale=1.):
        super(HipMomentCNN2D, self).__init__()
        
        self.input_size = input_size
        self.use_input_projection = use_input_projection
        
        # Input will be (batch, 1, 5, seq_len) - grayscale image
        input_channels = 1
        
        # Optional input projection to increase feature dimension
        if use_input_projection:
            self.input_projection = nn.Sequential(
                nn.Conv2d(input_channels, projection_size, kernel_size=(1, 1)),
                nn.BatchNorm2d(projection_size),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
            tcn_input_channels = projection_size
        else:
            tcn_input_channels = input_channels
        
        # Main 2D CNN - now with input_height=5
        self.tcn = TemporalConvNet2D(
            num_inputs=tcn_input_channels,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            input_height=input_size
        )
        
        # Height reduction layer: reduce from 4 to 1
        # This is the key modification to address your issue
        self.height_reduction = nn.Sequential(
            nn.Conv2d(tcn_channels[-1], tcn_channels[-1]//2, kernel_size=(4, 1)),  # Reduce height from 4 to 1
            nn.BatchNorm2d(tcn_channels[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Output layers - now operating on (batch, channels//2, 1, seq_len)
        self.output_projection = nn.Sequential(
            nn.Conv2d(tcn_channels[-1]//2, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv2d(16, 1, kernel_size=(1, 1))  # Single output channel
        )
        
        # Normalization parameters
        self.center = center
        self.scale = scale
        
        # Store receptive field
        self.receptive_field = self.tcn.receptive_field
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights for output layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, 5, sequence_length) - motor data
            
        Returns:
            hip_moment: (batch_size, sequence_length) - predicted hip moment for single leg
        """
        # Normalize input
        x = (x - self.center) / self.scale
        
        # Reshape input to 2D image format: (batch, 5, seq_len) -> (batch, 1, 5, seq_len)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Input projection (if enabled)
        if self.use_input_projection:
            x = self.input_projection(x)  # (batch, projection_size, 5, seq_len)
        
        # Pass through 2D CNN
        features = self.tcn(x)  # Output: (batch, channels, 4, seq_len)
        
        # Reduce height from 4 to 1
        features = self.height_reduction(features)  # (batch, channels//2, 1, seq_len)
        
        # Generate output
        output = self.output_projection(features)  # (batch, 1, 1, seq_len)
        
        # Squeeze to get (batch, seq_len)
        output = output.squeeze(1).squeeze(1)
        
        return output
    
    def get_effective_history(self):
        """Get the receptive field of the model"""
        return self.receptive_field


# Testing code
if __name__ == "__main__":
    # Test with your specified dimensions
    print("=== Testing Modified 2D CNN Model ===")
    model = HipMomentCNN2D(
        input_size=5,  # Changed to 5 as per your requirement
        tcn_channels=[64, 64, 64, 64, 64],
        kernel_size=2,
        dropout=0.2,
        
    )

    # Test input with your specified dimensions
    batch_size = 4
    seq_length = 218
    input_features = 5
    x = torch.randn(batch_size, input_features, seq_length)
    
    print(f"Input shape: {x.shape}")
    summary(model, (5,218),4,device="cpu")
    # Forward pass
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_length})")
    print(f"Receptive field: {model.get_effective_history()} timesteps")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # You can also use torchsummary if needed
    # summary(model, (5, 218), batch_size=4, device='cpu')



