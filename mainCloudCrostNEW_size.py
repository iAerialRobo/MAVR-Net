import os
import torch
from thop import profile  # Import thop for FLOPs and parameter calculation
from Cost_model import Cost, costblock  # Import your Cost and costblock

# Specify GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

# Define the model
class CostClassifier(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(CostClassifier, self).__init__()
        self.cost = Cost(num_classes, costblock, [2, 3, 4, 2])

    def forward(self, rgb):
        return self.cost(rgb)

# Initialize model
num_classes = 4
model = CostClassifier(num_classes=num_classes).to(device)

# Model input shape (batch_size, channels, frames, height, width)
input_shape = (16, 3, 16, 224, 224)
input_tensor = torch.randn(input_shape).to(device)

# Calculate model parameters and FLOPs using thop
macs, params = profile(model, inputs=(input_tensor,), verbose=False)

# Get the saved model size
model_path = "/home/zhangnb/videoUAV/videoWork/models/cost_rgb_only_model.pth"
if os.path.exists(model_path):
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)  # Convert to MB
else:
    model_size_mb = "Model file not found"

# Print results
print("=" * 50)
print("Model Metrics")
print("=" * 50)
print(f"Total Parameters: {params:,}")
print(f"FLOPs: {macs * 2 / 1e9:.2f} GFLOPs")  # Multiply by 2 for FLOPs (thop gives MACs)
print(f"MACs: {macs / 1e9:.2f} GMACs")
print(f"Model Size: {model_size_mb:.2f} MB" if isinstance(model_size_mb, float) else f"Model Size: {model_size_mb}")
print("=" * 50)

# import os
# import torch
# from torchinfo import summary
# from Cost_model import Cost, costblock  # Import your Cost and costblock
#
# # Specify GPU or CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device --> {device}")
#
# # Define the model
# class CostClassifier(torch.nn.Module):
#     def __init__(self, num_classes=4):
#         super(CostClassifier, self).__init__()
#         self.cost = Cost(num_classes, costblock, [2, 3, 4, 2])
#
#     def forward(self, rgb):
#         return self.cost(rgb)
#
# # Initialize model
# num_classes = 4
# model = CostClassifier(num_classes=num_classes).to(device)
#
# # Model input shape (batch_size, channels, frames, height, width)
# input_shape = (16, 3, 16, 224, 224)
#
# # Calculate model parameters and FLOPs using torchinfo
# model_stats = summary(model, input_size=input_shape, verbose=0)
#
# # Get the saved model size
# model_path = "/home/zhangnb/videoUAV/videoWork/models/cost_rgb_only_model.pth"
# if os.path.exists(model_path):
#     model_size_bytes = os.path.getsize(model_path)
#     model_size_mb = model_size_bytes / (1024 * 1024)  # Convert to MB
# else:
#     model_size_mb = "Model file not found"
#
# # Print results
# print("=" * 50)
# print("Model Metrics")
# print("=" * 50)
# print(f"Total Parameters: {model_stats.total_params:,}")
# print(f"FLOPs: {model_stats.total_mult_adds * 2 / 1e9:.2f} GFLOPs")  # Multiply by 2 for FLOPs (torchinfo gives MACs)
# print(f"MACs: {model_stats.total_mult_adds / 1e9:.2f} GMACs")
# print(f"Model Size: {model_size_mb:.2f} MB" if isinstance(model_size_mb, float) else f"Model Size: {model_size_mb}")
# print("=" * 50)
#
# # /home/zhangnb/miniconda3/bin/conda run -n videoWork --no-capture-output python /home/zhangnb/videoUAV/videoWork/mainCloudCrostNEW_size.py
# # Device --> cuda
# # ==================================================
# # Model Metrics
# # ==================================================
# # Total Parameters: 28,094,660
# # FLOPs: 2961.23 GFLOPs
# # MACs: 1480.61 GMACs
# # Model Size: 107.43 MB
# # ==================================================