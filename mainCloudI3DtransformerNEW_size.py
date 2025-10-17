import torch.nn as nn
import os
import torch
from thop import profile
from ptflops import get_model_complexity_info
import sys
from i3d_transformer import I3dTransformer
import torch.nn as nn

# Specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")


# Define the I3DClassifier class (same as in your code)
class I3DClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(I3DClassifier, self).__init__()
        self.base_model = I3dTransformer(num_classes=num_classes, d_model=64,
                                         transformer_config={'d_ff': 32, 'num_heads': 8, 'dropout': 0, 'num_layers': 2})
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, rgb):
        rgb = rgb.repeat(1, 1, 2, 1, 1)  # (B, C, T, H, W) -> (B, C*2, T, H, W)
        result = self.base_model(rgb)
        logits = self.fc2(result["embds"])
        return logits


# Initialize model
num_classes = 4
model = I3DClassifier(num_classes=num_classes).to(device)


# Calculate number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate FLOPs using ptflops and MACs using thop
def calculate_flops_macs(model, input_size):
    # Calculate FLOPs using ptflops
    flops, params = get_model_complexity_info(
        model,
        input_size[1:],  # (channels, frames, height, width)
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    # Calculate MACs using thop
    input_tensor = torch.randn(input_size).to(device)
    macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, macs, params


# Get model file size
def get_model_file_size(model_path):
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)  # Convert to MB
        return size_mb
    else:
        return None


# Main execution
if __name__ == "__main__":
    # Model path
    model_path = "/home/zhangnb/videoUAV/videoWork/models/i3d_rgb_only_model.pth"

    # Load model weights if exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using initialized model.")

    # Calculate parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")

    # Calculate FLOPs and MACs
    input_size = (1, 3, 30, 224, 224)  # (batch_size, channels, frames, height, width)
    flops, macs, params = calculate_flops_macs(model, input_size)
    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"MACs: {macs / 1e9:.3f} GMACs")
    print(f"Parameters (from ptflops): {params:,}")

    # Get model file size
    file_size_mb = get_model_file_size(model_path)
    if file_size_mb:
        print(f"Model file size: {file_size_mb:.2f} MB")
    else:
        print(f"Model file {model_path} not found. Cannot calculate file size.")

# Device --> cuda
# Model loaded from /home/zhangnb/videoUAV/videoWork/models/i3d_rgb_only_model.pth
# Number of parameters: 14,597,752
# FLOPs: 104.576 GFLOPs
# MACs: 104.659 GMACs
# Parameters (from ptflops): 14,597,752
# Model file size: 55.89 MB
#
# Process finished with exit code 0