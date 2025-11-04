'''
2D CNN model for hip moment estimation with enhanced residual connections
Modified to handle variable input heights and output 3 channels
Input: 2D motor data treated as grayscale image  
Output: 3D hip moment (3 channels)
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
            # First block: reduce height from input_height to input_height-1
            kernel_2d_first = (2, kernel_size)  
            kernel_2d_second = (1, kernel_size)  
            padding_2d_first = (0, padding)  
            padding_2d_second = (0, padding)
            self.output_height = input_height - 1
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
        
        # Residual connection within block
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


class ResidualGroup2D(nn.Module):
    """Group of temporal blocks with same channel numbers and cross-block residual connections"""
    def __init__(self, blocks, n_channels):
        super(ResidualGroup2D, self).__init__()
        self.blocks = nn.ModuleList(blocks)
        self.n_channels = n_channels
        
    def forward(self, x):
        # Process first block
        out = self.blocks[0](x)
        
        # Process remaining blocks with residual connections
        for i in range(1, len(self.blocks)):
            residual = out  # Save previous output as residual
            out = self.blocks[i](out)
            # Add residual connection (since channels are same, no need for projection)
            out = out + residual
            
        return out


class TemporalConvNet2D(nn.Module):
    """Stack of 2D temporal blocks with exponentially increasing dilation and enhanced residual connections"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, input_height=5):
        super(TemporalConvNet2D, self).__init__()
        
        layers = []
        groups = []
        current_group = []
        current_channel = None
        
        num_levels = len(num_channels)
        current_height = input_height
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            is_first = (i == 0)  # Mark the first block
            
            block = TemporalBlock2D(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout,
                                   is_first_block=is_first,
                                   input_height=current_height)
            
            # Update height after first block
            if is_first:
                current_height = current_height - 1
            
            # Group blocks with same output channels for residual connections
            if current_channel is None or current_channel != out_channels:
                # Start a new group
                if current_group:
                    # Save the previous group
                    if len(current_group) > 1:
                        # Multiple blocks with same channels - use ResidualGroup
                        groups.append(ResidualGroup2D(current_group, current_channel))
                    else:
                        # Single block - add directly
                        groups.append(current_group[0])
                current_group = [block]
                current_channel = out_channels
            else:
                # Add to current group (same channel number)
                current_group.append(block)
        
        # Don't forget the last group
        if current_group:
            if len(current_group) > 1:
                groups.append(ResidualGroup2D(current_group, current_channel))
            else:
                groups.append(current_group[0])
        
        self.network = nn.Sequential(*groups)
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


class ACCCNN2DWithRes(nn.Module):
    """
    2D CNN model for hip moment estimation with enhanced residual connections
    
    Architecture:
    1. Height expansion layer (if input height < target height)
    2. Treat input as grayscale image (1 channel, height=expanded_height, width=seq_len)
    3. 2D CNN layers with exponential dilation for temporal modeling
    4. Cross-block residual connections for blocks with same channel numbers
    5. Height reduction layer before output
    6. Output projection to predict 3-channel hip moment
    """
    def __init__(self, 
                input_size=2,  # Input height (can be 2, 5, or other values)
                output_channels=2,  # Number of output channels
                target_height=5,  # Target height after expansion
                tcn_channels=[64, 64, 128, 128, 256, 256, 128, 64],
                kernel_size=2,
                dropout=0.2,
                use_input_projection=True,
                projection_size=32,
                center=0., 
                scale=1.):
        super(ACCCNN2DWithRes, self).__init__()
        
        self.input_size = input_size
        self.output_channels = output_channels
        self.target_height = target_height
        self.use_input_projection = use_input_projection
        
        # Height expansion layer if input height is less than target height
        self.use_height_expansion = input_size < target_height
        if self.use_height_expansion:
            # Use transposed convolution to expand height
            self.height_expansion = nn.Sequential(
                nn.ConvTranspose2d(1, 16, kernel_size=(target_height - input_size + 1, 1), 
                                stride=(1, 1), padding=(0, 0)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(1, 1)),  # Keep 16 channels
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
            working_height = target_height
            # When using height expansion, output has 16 channels
            input_channels = 16
        else:
            working_height = input_size
            # When NOT using height expansion, input still has 1 channel after unsqueeze
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
        
        # Main 2D CNN with enhanced residual connections
        self.tcn = TemporalConvNet2D(
            num_inputs=tcn_input_channels,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            input_height=working_height
        )
        
        # Calculate the final height after TCN processing
        final_height = working_height - 1  # After first block reduces height by 1
        
        # Height reduction layer: reduce from final_height to 1
        if final_height > 1:
            self.height_reduction = nn.Sequential(
                nn.Conv2d(tcn_channels[-1], tcn_channels[-1]//2, 
                        kernel_size=(final_height, 1)),  # Dynamically set kernel height
                nn.BatchNorm2d(tcn_channels[-1]//2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        else:
            # If height is already 1, just change channels
            self.height_reduction = nn.Sequential(
                nn.Conv2d(tcn_channels[-1], tcn_channels[-1]//2, kernel_size=(1, 1)),
                nn.BatchNorm2d(tcn_channels[-1]//2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        
        # Output layers - now operating on (batch, channels//2, 1, seq_len)
        # Output output_channels instead of fixed 3
        self.output_projection = nn.Sequential(
            nn.Conv2d(tcn_channels[-1]//2, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv2d(16, output_channels, kernel_size=(1, 1))  # Output configurable channels
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
            elif isinstance(m, nn.ConvTranspose2d):
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
            x: (batch_size, input_features, sequence_length) - motor data
            
        Returns:
            hip_moment: (batch_size, output_channels, sequence_length) - predicted hip moments
        """
        # Normalize input
        x = (x - self.center) / self.scale
        
        # Reshape input to 2D image format: (batch, input_features, seq_len) -> (batch, 1, input_features, seq_len)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Height expansion if needed
        if self.use_height_expansion:
            x = self.height_expansion(x)  # (batch, 1, target_height, seq_len)
        
        # Input projection (if enabled)
        if self.use_input_projection:
            x = self.input_projection(x)  # (batch, projection_size, height, seq_len)
        
        # Pass through 2D CNN with enhanced residual connections
        features = self.tcn(x)  # Output: (batch, channels, height-1, seq_len)
        
        # Reduce height to 1
        features = self.height_reduction(features)  # (batch, channels//2, 1, seq_len)
        
        # Generate output
        output = self.output_projection(features)  # (batch, output_channels, 1, seq_len)
        
        # Squeeze height dimension to get (batch, output_channels, seq_len)
        output = output.squeeze(2)
        
        return output
    
    def get_effective_history(self):
        """Get the receptive field of the model"""
        return self.receptive_field


# Testing code
if __name__ == "__main__":
    # Test with input_features = 2 and output_channels = 3
    print("=== Testing Modified 2D CNN Model with Height Expansion ===")
    
    # Test with input_features = 2
    model = ACCCNN2DWithRes(
            input_size=5,  # Input height is 2
            output_channels=2,  # Output 3 channels
            target_height=5,  # Expand to height 5 internally
            tcn_channels=[64, 64, 64, 64, 64, 64, 64, ],
            kernel_size=3,
            dropout=0.2,
        )

    # Test input with specified dimensions
    batch_size = 4
    seq_length = 218
    input_features = 5
    x = torch.randn(batch_size, input_features, seq_length)
    
    print(f"Input shape: {x.shape}")
    print(f"TCN channels: [64, 64, 64, 64, 64]")
    print("Note: Cross-block residual connections are automatically added between blocks with same channel numbers")
    
    # Forward pass
    output = model(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Receptive field: {model.get_effective_history()} timesteps")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    # summary(model, (5,218), batch_size=4, device="cpu")
    
    # # Test with input_features = 5 (no height expansion needed)
    # print("\n=== Testing with input_features = 5 (no expansion) ===")
    # model2 = ACCCNN2DWithRes(
    #     input_size=5,  # Input height is 5
    #     output_channels=3,  # Output 3 channels
    #     target_height=5,  # No expansion needed
    #     tcn_channels=[64, 64, 128, 128, 256],
    #     kernel_size=2,
    #     dropout=0.2,
    # )
    
    # x2 = torch.randn(batch_size, 5, seq_length)
    # output2 = model2(x2)
    # print(f"Input shape: {x2.shape}")
    # print(f"Output shape: {output2.shape}")
    # print(f"Success! Output matches target: ({batch_size}, 3, {seq_length})")
    
    # # Verify gradient flow
    # print("\n=== Gradient Flow Test ===")
    # loss = output.mean()
    # loss.backward()
    # print("Gradient flow test passed - residual connections working properly")