'''
Pure TCN model for single leg hip moment estimation
Input: 2D motor data (angle + angular velocity)
Output: 1D hip moment
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Dict, Optional


class Chomp1d(nn.Module):
    """Removes the padding added by causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block containing two temporal convolutional layers"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks with exponentially increasing dilation"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
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


class HipMomentTCN(nn.Module):
    """
    Pure TCN model for single leg hip moment estimation
    
    Architecture:
    1. Optional input projection layer to increase feature dimension
    2. TCN layers with exponential dilation for temporal modeling
    3. Output projection to predict single hip moment
    """
    def __init__(self, 
                 input_size=2,  # angle + angular velocity
                 tcn_channels=[64, 64, 128, 128, 256, 256, 128, 64],  # More layers for better feature extraction
                 kernel_size=4,
                 dropout=0.2,
                 use_input_projection=True,
                 projection_size=32,
                 center=0., 
                 scale=1.):
        super(HipMomentTCN, self).__init__()
        
        self.input_size = input_size
        self.use_input_projection = use_input_projection
        
        # Optional input projection to increase feature dimension
        if use_input_projection:
            self.input_projection = nn.Sequential(
                nn.Conv1d(input_size, projection_size, kernel_size=1),
                nn.BatchNorm1d(projection_size),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
            tcn_input_size = projection_size
        else:
            tcn_input_size = input_size
        
        # Main TCN
        self.tcn = TemporalConvNet(
            num_inputs=tcn_input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Conv1d(tcn_channels[-1], tcn_channels[-1]//2, kernel_size=1),
            nn.BatchNorm1d(tcn_channels[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(tcn_channels[-1]//2, 1, kernel_size=1)  # Single output for one leg
        )
        
        # Normalization parameters
        self.center = center
        self.scale = scale
        
        # Store receptive field
        self.receptive_field = self.tcn.receptive_field
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights for output layers"""
        for m in self.output_projection.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, 2, sequence_length) - motor data (angle + angular velocity)
            
        Returns:
            hip_moment: (batch_size, sequence_length) - predicted hip moment for single leg
        """
        # Normalize input
        x = (x - self.center) / self.scale
        
        # Input projection (if enabled)
        if self.use_input_projection:
            x = self.input_projection(x)
        
        # Pass through TCN
        features = self.tcn(x)
        
        # Generate output
        output = self.output_projection(features)
        
        # Squeeze channel dimension since we have single output
        output = output.squeeze(1)  # (batch_size, sequence_length)
        
        return output
    
    def get_effective_history(self):
        """Get the receptive field of the model"""
        return self.receptive_field


class HipMomentTCNWithSkip(nn.Module):
    """
    Enhanced TCN with skip connections for single leg hip moment estimation
    Includes skip connections from early layers to preserve low-level features
    """
    def __init__(self, 
                 input_size=2,
                 tcn_channels=[64, 64, 128, 128, 256, 256, 128, 64],
                 kernel_size=4,
                 dropout=0.2,
                 use_skip_connections=True,
                 center=0., 
                 scale=1.):
        super(HipMomentTCNWithSkip, self).__init__()
        
        self.input_size = input_size
        self.use_skip_connections = use_skip_connections
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_size, tcn_channels[0], kernel_size=1),
            nn.BatchNorm1d(tcn_channels[0]),
            nn.ReLU()
        )
        
        # Build TCN blocks individually for skip connections
        self.tcn_blocks = nn.ModuleList()
        for i in range(len(tcn_channels)):
            dilation = 2 ** i
            in_channels = tcn_channels[i-1] if i > 0 else tcn_channels[0]
            out_channels = tcn_channels[i]
            
            block = TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation,
                padding=(kernel_size-1) * dilation,
                dropout=dropout
            )
            self.tcn_blocks.append(block)
        
        # Skip connection aggregation
        if use_skip_connections:
            # Collect features from multiple layers
            skip_indices = [1, 3, 5]  # Early, middle, late features
            skip_channels = sum([tcn_channels[i] for i in skip_indices])
            final_channels = tcn_channels[-1] + skip_channels
        else:
            final_channels = tcn_channels[-1]
        
        self.skip_indices = skip_indices if use_skip_connections else []
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Conv1d(final_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 1, kernel_size=1)
        )
        
        # Normalization
        self.center = center
        self.scale = scale
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with skip connections
        
        Args:
            x: (batch_size, 2, sequence_length)
            
        Returns:
            hip_moment: (batch_size, sequence_length)
        """
        # Normalize
        x = (x - self.center) / self.scale
        
        # Input projection
        x = self.input_projection(x)
        
        # Pass through TCN blocks with skip connections
        skip_features = []
        for i, block in enumerate(self.tcn_blocks):
            x = block(x)
            if i in self.skip_indices:
                skip_features.append(x)
        
        # Concatenate skip connections
        if self.use_skip_connections and skip_features:
            x = torch.cat([x] + skip_features, dim=1)
        
        # Output projection
        output = self.output_layers(x)
        output = output.squeeze(1)
        
        return output
    
    def get_effective_history(self):
        """Calculate receptive field"""
        receptive_field = 1
        kernel_size = 4
        for i in range(len(self.tcn_blocks)):
            dilation = 2 ** i
            receptive_field += 2 * (kernel_size - 1) * dilation
        return receptive_field


# Testing code
if __name__ == "__main__":
    from torchsummary import summary
    # Test basic model
    print("=== Testing Basic Single Leg TCN Model ===")
    model = HipMomentTCN(
        input_size=4,
        tcn_channels=[50, 50, 50, 50, 50],  # Similar to original config
        kernel_size=4,
        dropout=0.2
    )
    
    # Test input
    batch_size = 8
    seq_length = 218
    x = torch.randn(batch_size, 2, seq_length)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # summary(model, (4,218), 4, device='cpu')

    # # Forward pass
    # output = model(x)
    
    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")
    # print(f"Receptive field: {model.get_effective_history()} timesteps")
    
    # # Count parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")
    
    # # Test enhanced model with skip connections
    # print("\n=== Testing Enhanced Model with Skip Connections ===")
    # model_skip = HipMomentTCNWithSkip(
    #     input_size=2,
    #     tcn_channels=[64, 64, 128, 128, 256, 256, 128, 64],
    #     kernel_size=4,
    #     dropout=0.2,
    #     use_skip_connections=True
    # )
    
    # output_skip = model_skip(x)
    # print(f"Output shape: {output_skip.shape}")
    
    # total_params_skip = sum(p.numel() for p in model_skip.parameters())
    # print(f"Total parameters: {total_params_skip:,}")