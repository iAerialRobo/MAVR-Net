import os
import time
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from C3D_model import C3D  # Assuming this is your C3D model definition


# Define the C3DClassifier class as in your code
class C3DClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(C3DClassifier, self).__init__()
        self.c3d = C3D(num_classes=num_classes, pretrained=pretrained)

    def forward(self, rgb):
        return self.c3d(rgb)


# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

# Initialize model
num_classes = 4  # As defined in your code
model = C3DClassifier(num_classes=num_classes, pretrained=False).to(device)

# Compute number of parameters and FLOPs
input_shape = (3, 16, 112, 112)  # C3D input: (channels, frames, height, width)
flops, params = get_model_complexity_info(
    model,
    input_shape,
    as_strings=False,
    print_per_layer_stat=True
)

# Calculate MACs (approximately half of FLOPs for convolutional networks)
macs = flops / 2

# Get saved model size
model_path = "/home/zhangnb/videoUAV/videoWork/models/c3d_rgb_only_model.pth"
model_size_mb = 0
if os.path.exists(model_path):
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
else:
    print(f"Model file {model_path} not found. Please ensure the model is saved.")

# Measure CPU inference time
model.eval()  # Set model to evaluation mode
model = model.to("cpu")  # Move model to CPU for inference timing
dummy_input = torch.randn(1, *input_shape)  # Create a batch of 1 sample
with torch.no_grad():  # Disable gradient computation
    # Warm-up run to stabilize measurements
    for _ in range(5):
        _ = model(dummy_input)

    # Measure inference time over multiple runs for accuracy
    num_runs = 10
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = model(dummy_input)
    end_time = time.perf_counter()
    avg_inference_time = (end_time - start_time) / num_runs

# Print results
print("\nModel Metrics:")
print(f"Input Size (C, T, H, W): {input_shape}")
print(f"Total Parameters: {params:,}")
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"MACs: {macs / 1e9:.2f} GMACs")
print(f"Saved Model Size: {model_size_mb:.2f} MB")
print(f"Average CPU Inference Time (per sample): {avg_inference_time * 1000:.2f} ms")