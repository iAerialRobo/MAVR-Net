import random
import os
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from glob import glob
from IPython.display import Image
import matplotlib.pyplot as plt
from collections import Counter
import termcolor
from termcolor import colored
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import uuid

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

# Data paths and class-modality mappings
dataset_path = '/home/zhangnb/videoUAV/data/hdf5/videoDataIntegrate/middle'
class_modality_folders = {
    '0': {
        'RGB': 'MAV_inv_vShapeRGB',
        'FLOW': 'MAV_inv_vShapeFLOW',
        'MASK': 'MAV_inv_vShapeMASK'
    },
    '1': {
        'RGB': 'MAV_left_rightRGB',
        'FLOW': 'MAV_left_rightFLOW',
        'MASK': 'MAV_left_rightMASK'
    },
    '2': {
        'RGB': 'MAV_up_downRGB',
        'FLOW': 'MAV_up_downFLOW',
        'MASK': 'MAV_up_downMASK'
    },
    '3': {
        'RGB': 'MAV_vShapeRGB',
        'FLOW': 'MAV_vShapeFLOW',
        'MASK': 'MAV_vShapeMASK'
    }
}

# Utility functions
def denormalize_img(img, mean=0.5, std=0.5):
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

def create_gif(video_tensor, filename="sample.gif"):
    frames = []
    for video_frame in video_tensor:
        frame_denormalized = denormalize_img(video_frame.permute(1, 2, 0).cpu().numpy())
        frames.append(frame_denormalized)
    imageio.mimsave(filename, frames, "GIF", duration=0.25)
    return filename

def display_gif(video_tensor, gif_name="sample.gif"):
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)

def show_gifs(inputs_rgb, inputs_flow, inputs_mask, labels, predicted, index_to_class):
    for j, (rgb_tensor, flow_tensor, mask_tensor) in enumerate(zip(inputs_rgb, inputs_flow, inputs_mask)):
        if j % 3 == 0:
            category_idx = labels[j].item()
            pred_category_idx = predicted[j].item()
            color = "green" if category_idx == pred_category_idx else "red"
            category_text = colored(f"True Category   -> {index_to_class[category_idx]}", color, attrs=['bold'])
            prediction_text = colored(f"Model Prediction -> {index_to_class[pred_category_idx]}", color, attrs=['bold'])
            separator = colored("=" * 50, 'light_grey', attrs=['bold'])
            print(separator)
            print(category_text)
            print(prediction_text)
            print("RGB Modality:")
            display_gif(rgb_tensor.cpu())
            print("FLOW Modality:")
            display_gif(flow_tensor.cpu())
            print("MASK Modality:")
            display_gif(mask_tensor.cpu())

def plot_results(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2, color='#03346E')
    plt.plot(val_losses, label='Validation Loss', linewidth=2, color='#399918')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2, color='#399918')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('training_results.png', format='png', dpi=300)
    plt.close()

def plot_class_distribution(labels, title):
    class_counts = Counter(labels)
    classes = [str(k) for k in class_counts.keys()]
    counts = list(class_counts.values())
    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_classes, sorted_counts, color='#03346E')
    plt.xlabel('Number of Clips')
    plt.ylabel('Classes')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig('class_distribution.png', format='png', dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, figsize=(10, 8), cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.0%', cmap=cmap, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Video Classification - Confusion Matrix', fontsize=15)
    plt.savefig('confusion_matrix.png', format='png', dpi=300)
    plt.close()

def get_metrics(test_losses, test_accuracies, actual_labels, model_predictions):
    test_mean_loss = np.mean(test_losses)
    test_accuracy = np.mean(test_accuracies)
    f1 = f1_score(actual_labels, model_predictions, average='macro')
    color = 'blue'
    print(colored("=" * 75, 'dark_grey', attrs=['bold']))
    print(colored("Test Results:", color, attrs=['bold', 'underline']))
    print(colored(f"\tF1 Score: {f1:.4f}", color, attrs=['bold']))
    print(colored(f"\tMean Accuracy: {test_accuracy:.3f}%", color, attrs=['bold']))
    print(colored(f"\tMean Loss: {test_mean_loss:.3f}", color, attrs=['bold']))
    print(colored("=" * 75, 'dark_grey', attrs=['bold']))

# Function to visualize and save attention weights
def visualize_attention_weights(attn_weights, batch_idx, sample_idx, save_dir="/home/zhangnb/videoUAV/videoWork/outputs/attention_maps"):
    os.makedirs(save_dir, exist_ok=True)
    attn_weights = attn_weights.cpu().detach().numpy()  # Shape: [batch_size, num_frames, num_frames]
    sample_attn = attn_weights[sample_idx]  # Shape: [num_frames, num_frames]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_attn, cmap='viridis', annot=False, cbar=True)
    plt.title(f'Attention Weights - Batch {batch_idx}, Sample {sample_idx}')
    plt.xlabel('Frames (Key)')
    plt.ylabel('Frames (Query)')
    plt.savefig(os.path.join(save_dir, f'attention_batch_{batch_idx}_sample_{sample_idx}.png'), format='png', dpi=300)
    plt.close()

# New function to create composite GIF with attention overlay
def create_composite_gif(rgb_tensor, attn_weights, batch_idx, sample_idx, save_dir="/home/zhangnb/videoUAV/videoWork/outputs/composite_videos"):
    os.makedirs(save_dir, exist_ok=True)
    frames = []
    rgb_frames = rgb_tensor.cpu().numpy()  # Shape: [num_frames, c, h, w]
    attn_weights = attn_weights.cpu().detach().numpy()[sample_idx]  # Shape: [num_frames, num_frames]
    num_frames = rgb_frames.shape[0]

    for t in range(num_frames):
        # Denormalize RGB frame
        rgb_frame = denormalize_img(rgb_frames[t].transpose(1, 2, 0))  # Shape: [h, w, c]

        # Get attention weights for this frame
        attn = attn_weights[t]  # Shape: [num_frames]
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)  # Normalize to [0, 1]

        # Create heatmap
        heatmap = plt.cm.viridis(attn[t])[:3]  # Get RGB color for this frame's attention
        heatmap = np.ones_like(rgb_frame) * 255 * heatmap  # Scale to 0-255
        heatmap = heatmap.astype(np.uint8)

        # Overlay heatmap on RGB frame with transparency
        alpha = 0.4
        composite = (alpha * heatmap + (1 - alpha) * rgb_frame).astype(np.uint8)

        frames.append(composite)

    # Save composite GIF
    filename = os.path.join(save_dir, f'composite_batch_{batch_idx}_sample_{sample_idx}.gif')
    imageio.mimsave(filename, frames, "GIF", duration=0.25)
    return filename

# Data loading
val_split_size = 0.2
test_split_size = 0.3
height = 128
width = 128
img_size = (height, width)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# Modified load_video_frames to enforce consistent frame count
def load_video_frames(video_path, label, modalities=['RGB', 'FLOW', 'MASK'], num_frames=30):
    frames = {mod: [] for mod in modalities}
    label_str = str(label)
    for mod in modalities:
        mod_folder = class_modality_folders[label_str][mod]
        mod_path = video_path.replace(class_modality_folders[label_str]['RGB'], mod_folder)
        mod_path = mod_path.replace('_RGB.', f'_{mod}.')
        try:
            with h5py.File(mod_path, 'r') as hdf:
                loaded_frames = []
                for i in range(num_frames):
                    key = f'array_{i}'
                    if key in hdf:
                        array = np.array(hdf[key])
                        loaded_frames.append(array)
                    else:
                        break
                # Pad or truncate to ensure exactly num_frames
                if len(loaded_frames) < num_frames:
                    print(f"Warning: {mod} video {mod_path} has {len(loaded_frames)} frames, padding to {num_frames}")
                    loaded_frames.extend([np.zeros((height, width, 3))] * (num_frames - len(loaded_frames)))
                elif len(loaded_frames) > num_frames:
                    print(
                        f"Warning: {mod} video {mod_path} has {len(loaded_frames)} frames, truncating to {num_frames}")
                    loaded_frames = loaded_frames[:num_frames]
                frames[mod] = loaded_frames
        except Exception as e:
            print(f"Error loading {mod} from {mod_path}: {e}")
            frames[mod] = [np.zeros((height, width, 3))] * num_frames
    return frames['RGB'], frames['FLOW'], frames['MASK']


# ... (Dataset, data loading, and model code unchanged)
class MultimodalWorkoutDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_videos = len(video_paths)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label_name = str(self.labels[idx])
        label_index = class_to_index[label_name]
        try:
            rgb_frames, flow_frames, mask_frames = load_video_frames(video_path, label_name)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            rgb_frames = flow_frames = mask_frames = [np.zeros((height, width, 3))] * 30

        rgb_frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in rgb_frames]
        flow_frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in flow_frames]
        mask_frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in mask_frames]

        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]
            flow_frames = [self.transform(frame) for frame in flow_frames]
            mask_frames = [self.transform(frame) for frame in mask_frames]

        rgb_tensor = torch.stack(rgb_frames)
        flow_tensor = torch.stack(flow_frames)
        mask_tensor = torch.stack(mask_frames)
        label_tensor = torch.tensor(label_index, dtype=torch.long).to(device)

        return rgb_tensor, flow_tensor, mask_tensor, label_tensor

def create_csv(videos, labels, file_name=""):
    df = pd.DataFrame({'path': videos, 'label': labels})
    df.to_csv(f"{file_name}.csv", index=False)

# Load and split data
all_video_paths = []
all_labels = []
for class_id in class_modality_folders:
    rgb_folder = class_modality_folders[class_id]['RGB']
    class_paths = glob(os.path.join(dataset_path, rgb_folder, '*.h5'), recursive=True)
    class_labels = [class_id] * len(class_paths)
    all_video_paths.extend(class_paths)
    all_labels.extend(class_labels)

num_classes = len(set(all_labels))
class_to_index = {str(l): idx for idx, l in enumerate(set(all_labels))}
index_to_class = {idx: str(l) for idx, l in enumerate(set(all_labels))}

trainval_videos, test_videos, trainval_labels, test_labels = train_test_split(
    all_video_paths, all_labels, test_size=test_split_size, random_state=42)
train_videos, val_videos, train_labels, val_labels = train_test_split(
    trainval_videos, trainval_labels, test_size=val_split_size, random_state=42)

create_csv_files = True
csv_dir = "/home/zhangnb/videoUAV/videoWork/myModelCSV"
if create_csv_files:
    os.makedirs(csv_dir, exist_ok=True)
    create_csv(train_videos, train_labels, os.path.join(csv_dir, "Multimodal_train"))
    create_csv(val_videos, val_labels, os.path.join(csv_dir, "Multimodal_val"))
    create_csv(test_videos, test_labels, os.path.join(csv_dir, "Multimodal_test"))

train_csv_path = os.path.join(csv_dir, "Multimodal_train.csv")
val_csv_path = os.path.join(csv_dir, "Multimodal_val.csv")
test_csv_path = os.path.join(csv_dir, "Multimodal_test.csv")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

df = pd.concat([train_df, val_df, test_df], axis=0)
num_samples = df.shape[0]
class_counts = df.label.value_counts().to_dict()
plot_class_distribution(class_counts, 'Multimodal Workout Video Dataset - Classes Distribution')

class_weights = {cat: num_samples / count for cat, count in class_counts.items()}
all_weights = [class_weights[label] for label in train_df['label'].values]
n_samples = len(all_weights)
sampler = WeightedRandomSampler(weights=all_weights, num_samples=n_samples, replacement=False)

train_dataset = MultimodalWorkoutDataset(train_df['path'].values, train_df['label'].values, transform=transform)
val_dataset = MultimodalWorkoutDataset(val_df['path'].values, val_df['label'].values, transform=transform)
test_dataset = MultimodalWorkoutDataset(test_df['path'].values, test_df['label'].values, transform=transform)

balanced_train = False
if balanced_train:
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=50, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Model definition
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512], out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features):
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]
        out = laterals[-1]
        outputs = [out]
        for lateral in reversed(laterals[:-1]):
            out = F.interpolate(out, scale_factor=2, mode='nearest') + lateral
            outputs.append(out)
        outputs = reversed(outputs)
        outputs = [self.pool(conv(out)).view(out.size(0), -1) for conv, out in zip(self.output_convs, outputs)]
        return torch.cat(outputs, dim=-1)

class MultimodalWorkoutClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(MultimodalWorkoutClassifier, self).__init__()
        self.rgb_model = models.resnet18(pretrained=True)
        self.flow_model = models.resnet18(pretrained=True)
        self.mask_model = models.resnet18(pretrained=True)

        self.rgb_model.fc = nn.Identity()
        self.flow_model.fc = nn.Identity()
        self.mask_model.fc = nn.Identity()

        self.rgb_fpn = FeaturePyramidNetwork()
        self.flow_fpn = FeaturePyramidNetwork()
        self.mask_fpn = FeaturePyramidNetwork()

        self.cross_attention = nn.MultiheadAttention(embed_dim=3072, num_heads=12)
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def extract_features(self, model, x):
        features = []
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        features.append(x)
        x = model.layer2(x)
        features.append(x)
        x = model.layer3(x)
        features.append(x)
        x = model.layer4(x)
        features.append(x)
        return features

    def forward(self, rgb, flow, mask):
        batch_size, num_frames, c, h, w = rgb.shape

        rgb = rgb.view(-1, c, h, w)
        flow = flow.view(-1, c, h, w)
        mask = mask.view(-1, c, h, w)

        rgb_features = self.extract_features(self.rgb_model, rgb)
        flow_features = self.extract_features(self.flow_model, flow)
        mask_features = self.extract_features(self.mask_model, mask)

        rgb_fpn = self.rgb_fpn(rgb_features).view(batch_size, num_frames, -1)
        flow_fpn = self.flow_fpn(flow_features).view(batch_size, num_frames, -1)
        mask_fpn = self.mask_fpn(mask_features).view(batch_size, num_frames, -1)

        all_features = torch.stack([rgb_fpn, flow_fpn, mask_fpn], dim=0)
        all_features = all_features.permute(1, 2, 0, 3)
        all_features = all_features.reshape(batch_size, num_frames, -1)

        attn_output, attn_weights = self.cross_attention(all_features, all_features, all_features)
        combined = torch.mean(attn_output, dim=1)

        out = self.fc1(combined)
        out = F.relu(out)
        out = self.fc2(out)
        return out, (rgb_fpn, flow_fpn, mask_fpn), attn_weights

# Alignment loss
def multi_view_alignment_loss(rgb_features, flow_features, mask_features, temperature=0.07):
    batch_size = rgb_features.size(0)
    rgb_features = rgb_features.mean(dim=1)
    flow_features = flow_features.mean(dim=1)
    mask_features = mask_features.mean(dim=1)

    rgb_features = F.normalize(rgb_features, dim=-1)
    flow_features = F.normalize(flow_features, dim=-1)
    mask_features = F.normalize(mask_features, dim=-1)

    sim_rgb_flow = torch.matmul(rgb_features, flow_features.T) / temperature
    sim_rgb_mask = torch.matmul(rgb_features, mask_features.T) / temperature
    sim_flow_mask = torch.matmul(flow_features, mask_features.T) / temperature

    labels = torch.arange(batch_size).to(rgb_features.device)

    loss_rgb_flow = F.cross_entropy(sim_rgb_flow, labels)
    loss_rgb_mask = F.cross_entropy(sim_rgb_mask, labels)
    loss_flow_mask = F.cross_entropy(sim_flow_mask, labels)

    loss = (loss_rgb_flow + loss_rgb_mask + loss_flow_mask) / 3
    return loss

# Initialize model
model = MultimodalWorkoutClassifier(num_classes=num_classes).to(device)

# Training setup
lr = 1e-4
weight_decay = 5e-4
lambda_align = 0.5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_losses = []
train_align_losses = []
val_losses = []
val_accuracies = []
num_epochs = 10
max_gained_acc = 0
epoch_of_max_acc = 0
early_stopping = False
epochs_tolerance = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_align_loss = 0.0
    for rgb, flow, mask, labels in tqdm(train_loader):
        rgb, flow, mask, labels = rgb.to(device), flow.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, (rgb_fpn, flow_fpn, mask_fpn), _ = model(rgb, flow, mask)
        cls_loss = criterion(outputs, labels)
        align_loss = multi_view_alignment_loss(rgb_fpn, flow_fpn, mask_fpn)
        total_loss = cls_loss + lambda_align * align_loss
        total_loss.backward()
        optimizer.step()
        running_loss += cls_loss.item()
        running_align_loss += align_loss.item()

    train_loss = running_loss / len(train_loader)
    train_align_loss = running_align_loss / len(train_loader)
    train_losses.append(train_loss)
    train_align_losses.append(train_align_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Cls Loss: {train_loss:.4f}, Train Align Loss: {train_align_loss:.4f}')

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for rgb, flow, mask, labels in val_loader:
            rgb, flow, mask, labels = rgb.to(device), flow.to(device), mask.to(device), labels.to(device)
            outputs, _, _ = model(rgb, flow, mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if val_accuracy >= max_gained_acc:
        max_gained_acc = val_accuracy
        epoch_of_max_acc = epoch

    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}%')

    if early_stopping and epoch - epoch_of_max_acc > epochs_tolerance:
        print("Early stopping triggered")
        break

    scheduler.step()

plot_results(train_losses, val_losses, val_accuracies)

# Modified create_composite_gif to handle frame count mismatch
def create_composite_gif(rgb_tensor, attn_weights, batch_idx, sample_idx,
                         save_dir="/home/zhangnb/videoUAV/videoWork/outputs/composite_videos"):
    os.makedirs(save_dir, exist_ok=True)
    frames = []
    rgb_frames = rgb_tensor.cpu().numpy()  # Shape: [num_frames, c, h, w]
    attn_weights = attn_weights.cpu().detach().numpy()[sample_idx]  # Shape: [num_frames, num_frames]

    rgb_num_frames = rgb_frames.shape[0]
    attn_num_frames = attn_weights.shape[0]
    num_frames = min(rgb_num_frames, attn_num_frames)  # Use minimum to avoid index errors
    if rgb_num_frames != attn_num_frames:
        print(f"Warning: Mismatch in frame counts for batch {batch_idx}, sample {sample_idx}. "
              f"RGB: {rgb_num_frames}, Attention: {attn_num_frames}. Using {num_frames} frames.")

    for t in range(num_frames):
        # Denormalize RGB frame
        rgb_frame = denormalize_img(rgb_frames[t].transpose(1, 2, 0))  # Shape: [h, w, c]

        # Get attention weights for this frame
        attn = attn_weights[t]  # Shape: [num_frames]
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)  # Normalize to [0, 1]

        # Create heatmap
        heatmap = plt.cm.viridis(attn[t])[:3]  # Get RGB color for this frame's attention
        heatmap = np.ones_like(rgb_frame) * 255 * heatmap  # Scale to 0-255
        heatmap = heatmap.astype(np.uint8)

        # Overlay heatmap on RGB frame with transparency
        alpha = 0.4
        composite = (alpha * heatmap + (1 - alpha) * rgb_frame).astype(np.uint8)

        frames.append(composite)

    # Save composite GIF
    filename = os.path.join(save_dir, f'composite_batch_{batch_idx}_sample_{sample_idx}.gif')
    imageio.mimsave(filename, frames, "GIF", duration=0.25)
    print(f"Saved composite GIF: {filename}")
    return filename


# ... (Rest of the code, including test function, unchanged)

# Testing with attention, video, and composite visualization
def test():
    test_losses = []
    test_accuracies = []
    cum_labels = []
    cum_predicted = []
    show_gifs_results = True
    show_gifs_at = 5
    save_outputs = True
    output_base_dir = "/home/zhangnb/videoUAV/videoWork/outputs"
    video_dir = os.path.join(output_base_dir, "test_videos")

    model.eval()
    with torch.no_grad():
        for i, (rgb, flow, mask, labels) in enumerate(test_loader):
            rgb, flow, mask, labels = rgb.to(device), flow.to(device), mask.to(device), labels.to(device)
            outputs, _, attn_weights = model(rgb, flow, mask)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            test_loss = loss.item() / len(test_loader)
            test_accuracy = 100 * (correct / total)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            cum_labels.extend(labels.cpu().numpy())
            cum_predicted.extend(predicted.cpu().numpy())

            if show_gifs_results and i % show_gifs_at == 0:
                show_gifs(rgb, flow, mask, labels, predicted, index_to_class)

            if save_outputs:
                os.makedirs(video_dir, exist_ok=True)
                for sample_idx in range(min(3, rgb.size(0))):  # Save for first 3 samples per batch
                    # Save attention map
                    visualize_attention_weights(attn_weights, i, sample_idx)

                    # Save modality videos
                    create_gif(rgb[sample_idx], os.path.join(video_dir, f'rgb_batch_{i}_sample_{sample_idx}.gif'))
                    create_gif(flow[sample_idx], os.path.join(video_dir, f'flow_batch_{i}_sample_{sample_idx}.gif'))
                    create_gif(mask[sample_idx], os.path.join(video_dir, f'mask_batch_{i}_sample_{sample_idx}.gif'))

                    # Save composite video with error handling
                    try:
                        create_composite_gif(rgb[sample_idx], attn_weights, i, sample_idx)
                    except Exception as e:
                        print(f"Error creating composite GIF for batch {i}, sample {sample_idx}: {e}")

    print('Test accuracy:', np.mean(test_accuracies))
    get_metrics(test_losses, test_accuracies, cum_labels, cum_predicted)
    plot_confusion_matrix(cum_labels, cum_predicted, list(class_to_index.keys()))

test()

# Save model
save_model = True
model_path = "/home/zhangnb/videoUAV/videoWork/models/corrected_multimodal_final_model.pth"
if save_model:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)