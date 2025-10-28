import torch
import torch.nn as nn
from model.ConvTimeNet_backbone import ConvTimeNet_backbone
from torchsummary import summary

def main():
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Network parameters
    c_in = 5          # Input channels (features)
    c_out = 10        # Output classes (adjust based on your task)
    seq_len = 218     # Sequence length
    batch_size = 32   # Batch size for testing
    

    # Create model
    model = ConvTimeNet_backbone(
        c_in=5,
        seq_len=218,
        context_window = 218,
        target_window = 218,
        patch_len= 32,
        stride=16,
        n_layers=3,
        d_model=64,
        d_ff=128,
        dropout=0.3,
        act="relu",
        enable_res_param=False,
        dw_ks=[7, 7,13,13,19,19],  # Depth-wise kernel sizes for each layer
        norm='batch',
        re_param=False,
        deformable=True,
        reduced_channels=16,
        revin = False,
        final_out=1,
    ).to(device)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create sample input data
    # Shape: [batch_size, channels, sequence_length]
    sample_input = torch.randn(batch_size, c_in, seq_len).to(device)
    print(f"Input shape: {sample_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
        print(f"Output shape: {output.shape}")
        # print(f"Output sample: {output[0]}")  # First sample output
    
    # Test training mode
    model.train()
    output_train = model(sample_input)
    print(f"Training mode output shape: {output_train.shape}")
    
    # Example with different pooling strategies
    print("\n=== Testing different pooling strategies ===")
    # summary(model=model, input_size=(5,218),batch_size=1)

if __name__ == "__main__":
    main()