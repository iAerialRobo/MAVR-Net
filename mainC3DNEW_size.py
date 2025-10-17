import os
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

# Print results
print("\nModel Metrics:")
print(f"Total Parameters: {params:,}")
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"MACs: {macs / 1e9:.2f} GMACs")
print(f"Saved Model Size: {model_size_mb:.2f} MB")

# /home/zhangnb/miniconda3/bin/conda run -n videoWork --no-capture-output python /home/zhangnb/videoUAV/videoWork/mainC3DNEW_size.py
# Device --> cuda
# C3DClassifier(
#  78.01 M, 100.000% Params, 38.62 GMac, 99.884% MACs,
#  (c3d): C3D(
#    78.01 M, 100.000% Params, 38.62 GMac, 99.884% MACs,
#    (conv1): Conv3d(5.25 k, 0.007% Params, 1.05 GMac, 2.725% MACs, 3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (pool1): MaxPool3d(0, 0.000% Params, 12.85 MMac, 0.033% MACs, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, dilation=1, ceil_mode=False)
#    (conv2): Conv3d(221.31 k, 0.284% Params, 11.1 GMac, 28.724% MACs, 64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (pool2): MaxPool3d(0, 0.000% Params, 6.42 MMac, 0.017% MACs, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False)
#    (conv3a): Conv3d(884.99 k, 1.134% Params, 5.55 GMac, 14.358% MACs, 128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (conv3b): Conv3d(1.77 M, 2.269% Params, 11.1 GMac, 28.711% MACs, 256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (pool3): MaxPool3d(0, 0.000% Params, 1.61 MMac, 0.004% MACs, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False)
#    (conv4a): Conv3d(3.54 M, 4.537% Params, 2.77 GMac, 7.178% MACs, 256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (conv4b): Conv3d(7.08 M, 9.073% Params, 5.55 GMac, 14.355% MACs, 512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (pool4): MaxPool3d(0, 0.000% Params, 401.41 KMac, 0.001% MACs, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False)
#    (conv5a): Conv3d(7.08 M, 9.073% Params, 693.68 MMac, 1.794% MACs, 512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (conv5b): Conv3d(7.08 M, 9.073% Params, 693.68 MMac, 1.794% MACs, 512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#    (pool5): MaxPool3d(0, 0.000% Params, 50.18 KMac, 0.000% MACs, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)
#    (fc6): Linear(33.56 M, 43.017% Params, 33.56 MMac, 0.087% MACs, in_features=8192, out_features=4096, bias=True)
#    (fc7): Linear(16.78 M, 21.511% Params, 16.78 MMac, 0.043% MACs, in_features=4096, out_features=4096, bias=True)
#     (fc8): Linear(16.39 k, 0.021% Params, 16.39 KMac, 0.000% MACs, in_features=4096, out_features=4, bias=True)
#     (dropout): Dropout(0, 0.000% Params, 0.0 Mac, 0.000% MACs, p=0.5, inplace=False)
#     (relu): ReLU(0, 0.000% Params, 23.39 MMac, 0.061% MACs, )
#   )
# )
#
# Model Metrics:
# Total Parameters: 78,012,164
# FLOPs: 38.66 GFLOPs
# MACs: 19.33 GMACs
# Saved Model Size: 297.60 MB
#
# Process finished with exit code 0
