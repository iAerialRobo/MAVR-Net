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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from i3d_transformer import I3dTransformer

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")


# I3D Transformer Model (from original code)



# Data paths and class mappings
dataset_path = '/home/zhangnb/videoUAV/data/hdf5/videoDataIntegrate/remote'
class_folders = {
    '0': 'MAV_inv_vShapeRGB',
    '1': 'MAV_left_rightRGB',
    '2': 'MAV_up_downRGB',
    '3': 'MAV_vShapeRGB'
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


def show_gifs(inputs_rgb, labels, predicted, index_to_class):
    for j, rgb_tensor in enumerate(inputs_rgb):
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


def plot_results(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2, color='#03346E')
    plt.plot(val_losses, label='Validation Loss', linewidth=2, color='#399918')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2, color='#399918')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(val_f1_scores, label='Validation F1 Score', linewidth=2, color='#FF6F61')
    plt.title('Validation F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(val_precisions, label='Validation Precision', linewidth=2, color='#6B728E')
    plt.plot(val_recalls, label='Validation Recall', linewidth=2, color='#FFD93D')
    plt.title('Validation Precision and Recall Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_results_i3d.png', format='png', dpi=300)
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
    plt.savefig('class_distribution_i3d.png', format='png', dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, figsize=(10, 8), cmap='Blues', suffix=''):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.0%' if normalize else 'd', cmap=cmap, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Video Classification - Confusion Matrix {suffix}', fontsize=15)
    plt.savefig(f'confusion_matrix_i3d{suffix}.png', format='png', dpi=300)
    plt.close()


def plot_per_class_f1_scores(f1_scores, class_names, title, suffix=''):
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, f1_scores, color='#03346E')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'per_class_f1_scores_i3d{suffix}.png', format='png', dpi=300)
    plt.close()


def get_metrics(test_losses, test_accuracies, actual_labels, model_predictions, class_names):
    test_mean_loss = np.mean(test_losses)
    test_accuracy = np.mean(test_accuracies)
    f1_macro = f1_score(actual_labels, model_predictions, average='macro')
    precision = precision_score(actual_labels, model_predictions, average='macro')
    recall = recall_score(actual_labels, model_predictions, average='macro')
    per_class_f1 = f1_score(actual_labels, model_predictions, average=None)
    color = 'blue'
    print(colored("=" * 75, 'dark_grey', attrs=['bold']))
    print(colored("Test Results:", color, attrs=['bold', 'underline']))
    print(colored(f"\tF1 Score (Macro): {f1_macro:.4f}", color, attrs=['bold']))
    print(colored(f"\tPrecision (Macro): {precision:.4f}", color, attrs=['bold']))
    print(colored(f"\tRecall (Macro): {recall:.4f}", color, attrs=['bold']))
    print(colored(f"\tMean Accuracy: {test_accuracy:.3f}%", color, attrs=['bold']))
    print(colored(f"\tMean Loss: {test_mean_loss:.3f}", color, attrs=['bold']))
    print(colored("Per-Class F1 Scores:", color, attrs=['bold']))
    for class_name, f1_score_value in zip(class_names, per_class_f1):
        print(colored(f"\tClass {class_name}: {f1_score_value:.4f}", color, attrs=['bold']))
    print(colored("=" * 75, 'dark_grey', attrs=['bold']))
    plot_per_class_f1_scores(per_class_f1, class_names, 'Per-Class F1 Scores - Test', suffix='_test')


# Data loading
val_split_size = 0.3
test_split_size = 0.3
height = 224  # Adjusted for I3D input requirements
width = 224
img_size = (height, width)
num_frames = 30  # I3D expects 30 frames as in original code

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
])


def load_video_frames(video_path, label, num_frames=30):
    frames = []
    label_str = str(label)
    try:
        with h5py.File(video_path, 'r') as hdf:
            total_frames = len([key for key in hdf.keys() if key.startswith('array_')])
            if total_frames < num_frames:
                indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            for i in indices:
                array = np.array(hdf[f'array_{i}'])
                frames.append(array)
    except Exception as e:
        print(f"Error loading RGB from {video_path}: {e}")
        frames = [np.zeros((height, width, 3))] * num_frames
    return frames


class RGBWorkoutDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=30):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label_name = str(self.labels[idx])
        label_index = class_to_index[label_name]
        try:
            rgb_frames = load_video_frames(video_path, label_name, num_frames=self.num_frames)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            rgb_frames = [np.zeros((height, width, 3))] * self.num_frames

        rgb_frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in rgb_frames]
        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]

        rgb_tensor = torch.stack(rgb_frames).permute(1, 0, 2, 3)  # (C, T, H, W)
        label_tensor = torch.tensor(label_index, dtype=torch.long).to(device)

        return rgb_tensor, label_tensor


def create_csv(videos, labels, file_name=""):
    df = pd.DataFrame({'path': videos, 'label': labels})
    df.to_csv(f"{file_name}.csv", index=False)


# Load and split data
all_video_paths = []
all_labels = []
for class_id, rgb_folder in class_folders.items():
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
    create_csv(train_videos, train_labels, os.path.join(csv_dir, "RGB_train_i3d"))
    create_csv(val_videos, val_labels, os.path.join(csv_dir, "RGB_val_i3d"))
    create_csv(test_videos, test_labels, os.path.join(csv_dir, "RGB_test_i3d"))

train_csv_path = os.path.join(csv_dir, "RGB_train_i3d.csv")
val_csv_path = os.path.join(csv_dir, "RGB_val_i3d.csv")
test_csv_path = os.path.join(csv_dir, "RGB_test_i3d.csv")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

df = pd.concat([train_df, val_df, test_df], axis=0)
num_samples = df.shape[0]
class_counts = df.label.value_counts().to_dict()
plot_class_distribution(class_counts, 'RGB Workout Video Dataset - Classes Distribution (I3D)')

class_weights = {cat: num_samples / count for cat, count in class_counts.items()}
all_weights = [class_weights[label] for label in train_df['label'].values]
n_samples = len(all_weights)
sampler = WeightedRandomSampler(weights=all_weights, num_samples=n_samples, replacement=False)

train_dataset = RGBWorkoutDataset(train_df['path'].values, train_df['label'].values, transform=transform,
                                  num_frames=num_frames)
val_dataset = RGBWorkoutDataset(val_df['path'].values, val_df['label'].values, transform=transform,
                                num_frames=num_frames)
test_dataset = RGBWorkoutDataset(test_df['path'].values, test_df['label'].values, transform=transform,
                                 num_frames=num_frames)

balanced_train = False
if balanced_train:
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=8, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)


# I3D Classifier
class I3DClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(I3DClassifier, self).__init__()
        self.base_model = I3dTransformer(num_classes=num_classes, d_model=64,
                                         transformer_config={'d_ff': 32, 'num_heads': 8, 'dropout': 0, 'num_layers': 2})
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, rgb):
        # Repeat channels to match original I3D input (3 channels repeated to 6 for compatibility)
        rgb = rgb.repeat(1, 1, 2, 1, 1)  # (B, C, T, H, W) -> (B, C*2, T, H, W)
        result = self.base_model(rgb)
        logits = self.fc2(result["embds"])
        return logits


# Initialize model
model = I3DClassifier(num_classes=num_classes).to(device)

# Optimizer setup
lr = 1e-4
weight_decay = 5e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training setup
train_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []
val_precisions = []
val_recalls = []
num_epochs = 30
max_gained_acc = 0
epoch_of_max_acc = 0
early_stopping = False
epochs_tolerance = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for rgb, labels in tqdm(train_loader):
        rgb, labels = rgb.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_labels = []
    val_predictions = []
    with torch.no_grad():
        for rgb, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            outputs = model(rgb)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_labels.extend(labels.cpu().numpy())
            val_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    f1_macro = f1_score(val_labels, val_predictions, average='macro')
    precision = precision_score(val_labels, val_predictions, average='macro')
    recall = recall_score(val_labels, val_predictions, average='macro')
    val_per_class_f1 = f1_score(val_labels, val_predictions, average=None)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_f1_scores.append(f1_macro)
    val_precisions.append(precision)
    val_recalls.append(recall)

    plot_confusion_matrix(val_labels, val_predictions, list(class_to_index.keys()), suffix=f'_epoch_{epoch + 1}')
    plot_per_class_f1_scores(val_per_class_f1, list(class_to_index.keys()), f'Per-Class F1 Scores - Epoch {epoch + 1}',
                             suffix=f'_epoch_{epoch + 1}')

    if val_accuracy >= max_gained_acc:
        max_gained_acc = val_accuracy
        epoch_of_max_acc = epoch

    print(
        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}%, Val F1: {f1_macro:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}')
    print(f'Per-Class F1 Scores for Epoch {epoch + 1}:')
    for class_name, f1_score_value in zip(list(class_to_index.keys()), val_per_class_f1):
        print(f'  Class {class_name}: {f1_score_value:.4f}')

    if early_stopping and epoch - epoch_of_max_acc > epochs_tolerance:
        print("Early stopping triggered")
        break

    scheduler.step()

plot_results(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls)


# Testing
def test():
    test_losses = []
    test_accuracies = []
    cum_labels = []
    cum_predicted = []
    show_gifs_results = True
    show_gifs_at = 5

    model.eval()
    with torch.no_grad():
        for i, (rgb, labels) in enumerate(test_loader):
            rgb, labels = rgb.to(device), labels.to(device)
            outputs = model(rgb)
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
                show_gifs(rgb.permute(0, 2, 1, 3, 4), labels, predicted, index_to_class)

    print('Test accuracy:', np.mean(test_accuracies))
    get_metrics(test_losses, test_accuracies, cum_labels, cum_predicted, list(class_to_index.keys()))
    plot_confusion_matrix(cum_labels, cum_predicted, list(class_to_index.keys()), suffix='_test')


test()

# Save model
save_model = True
model_path = "/home/zhangnb/videoUAV/videoWork/models/i3d_rgb_only_model.pth"
if save_model:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)