import os
import torch
import torch.nn as nn
from thop import profile


class attentionblock(nn.Module):
    def __init__(self, in_channel):
        super(attentionblock, self).__init__()
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.in_channel = in_channel
        self.conv = nn.Conv3d(self.in_channel, self.in_channel, kernel_size=(1, 1, 1))
        self.fc = nn.Linear(self.in_channel, self.in_channel)

    def forward(self, input1, input2, input3):
        # Global pooling for each input
        input1 = self.pool(input1)
        input2 = self.pool(input2)
        input3 = self.pool(input3)

        # 1x1x1 convolution
        input1 = self.conv(input1)
        input2 = self.conv(input2)
        input3 = self.conv(input3)

        # Flatten to [B, C]
        input1 = input1.view(input1.size(0), -1)
        input2 = input2.view(input2.size(0), -1)
        input3 = input3.view(input3.size(0), -1)

        # Apply FC layer to each
        input1 = self.fc(input1)
        input2 = self.fc(input2)
        input3 = self.fc(input3)

        # Concatenate and apply softmax
        a = torch.cat((input1, input2, input3), 1)
        a = nn.functional.softmax(a, dim=1)
        return a


class costblock(nn.Module):
    def __init__(self, in_channel, channel, stride=1):
        super(costblock, self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.channel = channel
        self.bn1 = nn.BatchNorm3d(self.channel)
        self.bn2 = nn.BatchNorm3d(self.channel * 4)
        self.conv1 = nn.Conv3d(self.in_channel, self.channel, kernel_size=(1, 1, 1))
        self.conv = nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), padding=(1, 1),
                              stride=(self.stride, self.stride))
        self.conv2 = nn.Conv3d(self.channel, self.channel * 4, kernel_size=(1, 1, 1))
        self.attenblock = attentionblock(self.channel)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(self.channel)
        self.downsample = nn.Sequential()
        if self.stride != 1 or self.in_channel != self.channel * 4:
            self.downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, self.channel * 4, kernel_size=1, stride=(1, self.stride, self.stride),
                          bias=False),
                nn.BatchNorm3d(self.channel * 4)
            )

    def forward(self, input):
        shortcut = self.downsample(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)

        # Store original shape
        B, C, T, H, W = input.shape

        # Create three different 2D views and process them
        # View 1: Treat as (B, C, T, H*W) -> 2D conv on H*W
        x1 = input.view(B, C, T, H * W)
        out1 = self.conv(x1)
        out1 = self.batchnorm(out1)
        out1 = self.relu(out1)
        out1 = out1.view(B, C, T, H // self.stride, W // self.stride)

        # View 2: Treat as (B, C, H, T*W) -> 2D conv on T*W
        x2 = input.transpose(2, 3).contiguous()  # [B, C, H, T, W]
        x2 = x2.view(B, C, H, T * W)
        out2 = self.conv(x2)
        out2 = self.batchnorm(out2)
        out2 = self.relu(out2)
        out2 = out2.view(B, C, H // self.stride, T, W // self.stride)
        out2 = out2.transpose(2, 3).contiguous()  # Back to [B, C, T, H//stride, W//stride]

        # View 3: Treat as (B, C, W, T*H) -> 2D conv on T*H
        x3 = input.transpose(2, 4).contiguous()  # [B, C, W, H, T]
        x3 = x3.view(B, C, W, T * H)
        out3 = self.conv(x3)
        out3 = self.batchnorm(out3)
        out3 = self.relu(out3)
        out3 = out3.view(B, C, W // self.stride, T, H // self.stride)
        out3 = out3.transpose(2, 4).contiguous()  # Back to [B, C, T, H//stride, W//stride]

        # Debug: Print shapes to understand the mismatch
        print(f"out1 shape: {out1.shape}")
        print(f"out2 shape: {out2.shape}")
        print(f"out3 shape: {out3.shape}")

        # Ensure all outputs have the same shape by resizing if necessary
        target_shape = out1.shape
        if out2.shape != target_shape:
            out2 = torch.nn.functional.interpolate(out2, size=target_shape[2:], mode='trilinear', align_corners=False)
        if out3.shape != target_shape:
            out3 = torch.nn.functional.interpolate(out3, size=target_shape[2:], mode='trilinear', align_corners=False)

        # Get attention weights
        a = self.attenblock(out1, out2, out3)
        a1, a2, a3 = a.chunk(3, dim=1)

        # Reshape attention weights to match feature dimensions
        a1 = a1.view(B, C, 1, 1, 1)  # [B, C] -> [B, C, 1, 1, 1]
        a2 = a2.view(B, C, 1, 1, 1)
        a3 = a3.view(B, C, 1, 1, 1)

        # Apply attention weights - broadcast across spatial and temporal dimensions
        output1 = out1 * a1 + out2 * a2 + out3 * a3

        # Final convolution and residual connection
        output1 = self.conv2(output1)
        output1 = self.bn2(output1)
        output1 = output1 + shortcut
        output1 = self.relu(output1)
        return output1


class Cost(nn.Module):
    def __init__(self, num_classes, block, layers, pretrained=False):
        super(Cost, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.__init_weight()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * 4
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CostClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CostClassifier, self).__init__()
        self.cost = Cost(num_classes, costblock, [2, 3, 4, 2])

    def forward(self, rgb):
        return self.cost(rgb)


# Specify GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

# Initialize model
num_classes = 4
model = CostClassifier(num_classes=num_classes).to(device)

# Model input shape (batch_size, channels, frames, height, width)
input_shape = (1, 3, 16, 224, 224)
input_tensor = torch.randn(input_shape).to(device)

# Calculate model parameters and FLOPs using thop
try:
    macs, params = profile(model, inputs=(input_tensor,), verbose=True)

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
    print(f"FLOPs: {macs * 2 / 1e9:.2f} GFLOPs")  # Multiply by 2 for FLOPs
    print(f"MACs: {macs / 1e9:.2f} GMACs")
    print(f"Model Size: {model_size_mb:.2f} MB" if isinstance(model_size_mb, float) else f"Model Size: {model_size_mb}")
    print("=" * 50)

except Exception as e:
    print(f"Error during profiling: {e}")
    print("Running a simple forward pass to check model functionality...")

    # Test with a simple forward pass
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Model output shape: {output.shape}")
        print("Model forward pass successful!")