import torch
import torch.nn as nn
import os
import time
from thop import profile
from ptflops import get_model_complexity_info
from i3d_transformer import I3dTransformer

# Specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")


# Define the I3DClassifier class
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
    flops, params = get_model_complexity_info(
        model,
        input_size[1:],  # (channels, frames, height, width)
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
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


# Measure CPU inference time
def measure_cpu_inference_time(model, input_size, num_runs=100):
    model = model.to("cpu")  # Move model to CPU
    model.eval()  # Set model to evaluation mode
    input_tensor = torch.randn(input_size).to("cpu")

    # Warm-up run
    with torch.no_grad():
        _ = model(input_tensor)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time


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

    # Input size
    input_size = (1, 3, 30, 224, 224)  # (batch_size, channels, frames, height, width)
    print(f"Input image size: {input_size[3]}x{input_size[4]} pixels, {input_size[2]} frames")

    # Calculate FLOPs and MACs
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

    # Measure CPU inference time
    avg_cpu_inference_time = measure_cpu_inference_time(model, input_size, num_runs=100)
    print(f"Average CPU inference time: {avg_cpu_inference_time * 1000:.2f} ms per sample")