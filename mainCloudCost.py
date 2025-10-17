import random
import os
# import cv2
import imageio
import av
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import models, transforms
from torchvision.io import read_video
from torchvision.datasets import VisionDataset
from C3D_model import C3D
from Cost_model import Cost, costblock
import sys
import os
import h5py
# 指定使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

def denormalize_img(img,mean=0.5,std=0.5):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
    video.shape: (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_denormalized = denormalize_img(video_frame.permute(1, 2, 0).cpu().numpy())
        frames.append(frame_denormalized)
    kwargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kwargs)
    return filename


def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)


def show_gifs(inputs, labels, predicted):
    for j, video_tensor in enumerate(inputs):
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
            gif_image = display_gif(video_tensor.cpu())
            #display(gif_image)


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

    plt.show()


def plot_class_distribution(labels, title):
    class_counts = Counter(labels)
    classes = list(class_counts.keys())
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
    plt.savefig('classDistribution.png', format='png', dpi=300)
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, normalize=True, figsize=(10, 8), cmap='Blues'):
    class_names = list(class_to_index.keys())
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.0%', cmap=cmap, cbar=False,
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Video Classification - Confusion Matrix', fontsize=15)
    plt.savefig('confusionMatrix.png', format='png', dpi=300)

    # plt.show()


def get_metrics(test_losses,test_accuracies, actual_labels, model_predictions):
    test_mean_loss = np.mean(test_losses)
    test_accuracy = np.mean(test_accuracies)
    f1 = f1_score(actual_labels, model_predictions, average='macro')

    color = 'blue'
    results_text = colored("Test Results:", color, attrs=['bold', 'underline'])
    f1_text = colored(f"\tF1 Score: {f1:.4f}", color, attrs=['bold'])
    acc_text = colored(f"\tMean Accuracy: {test_accuracy:.3f}%", color, attrs=['bold'])
    loss_text = colored(f"\tMean Loss: {test_mean_loss:.3f}", color, attrs=['bold'])
    separator = colored("=" * 75, 'dark_grey', attrs=['bold'])

    print(separator)
    print(results_text)
    print(f1_text)
    print(acc_text)
    print(loss_text)
    print(separator)

    plot_confusion_matrix(actual_labels, model_predictions)

val_split_size = 0.1
test_split_size = 0.2

height = 224
width = 224
img_size = (height, width)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

'''
def extract_key_frames(video_path, num_key_frames=5, diff_threshold=30):
    """
    Key frames are frames which demonstrated significant l1 difference between the previous frame.
    This function return the frames in array for later tensor stack.
    :param video_path: video path
    :param num_key_frames: number of key frames to create
    :param diff_threshold: threshold set to define a significant "change" in video
    :return: Key frames
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        raise ValueError(f"Error reading video {video_path}")

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    key_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
        diff_sum = np.sum(frame_diff)

        if diff_sum > diff_threshold:
            key_frames.append(frame)
            frame_count += 1
            if frame_count >= num_key_frames:
                break

        prev_frame_gray = frame_gray

    cap.release()

    if len(key_frames) < num_key_frames:
        key_frames += [key_frames[-1]] * (num_key_frames - len(key_frames))

    return key_frames
'''

def load_video_frames(video_path, num_frames=30):
    """
    Converting each frame to numpy array and append to list
    :param video_path: Video to create frames for
    :param num_frames: number of frames to create
    :return: a list of frames
    """
    

    RGB = []
    with h5py.File(video_path, 'r') as hdf:
        for i in range(30):
            array = np.array(hdf[f'array_{i}'])
            RGB.append(array)
        
    flow_path = video_path.replace('RGB', 'FLOW')
   
    FLOW = []
    with h5py.File(flow_path, 'r') as hdf:
        for i in range(30):
            array = np.array(hdf[f'array_{i}'])
            FLOW.append(array)
            
    # print(video_path)
    # print(flow_path)
    # print(len(FLOW), '=',len(RGB))
    # sys.exit()
    return RGB, FLOW
   


class WorkoutDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_videos = len(video_paths)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Convert frames to tensors - each frame is converted from (Height, Width, Channels) to (Channels, Height, Width) for matching with ResNet model dimensions, and apply transformations.
        :param idx: Video idx
        :return: Video as stacked-framed tensor, and label tensors (tuple)
        """
        video_path = self.video_paths[idx]
        label_name = self.labels[idx]
        label_index = class_to_index[label_name]

        frames, flows = load_video_frames(video_path, num_frames=30)
        

        frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in frames]
        flows = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in flows]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            flows = [self.transform(frame) for frame in flows]
        frames_tensor = torch.stack(frames)
        flows_tensor = torch.stack(frames)
        label_tensor = torch.tensor(label_index, dtype=torch.long).to(device)
        
        frames_tensor = frames_tensor.permute(1,0,2,3)
        flows_tensor = flows_tensor.permute(1,0,2,3)
        return frames_tensor, flows_tensor, label_tensor

def create_csv(videos, labels, file_name=""):
    df = pd.DataFrame.from_records(zip(videos, labels), columns=["path", "label"])
    df.to_csv(f"{file_name}.csv", index=False)


# dataset_path = '/home/dwqdx3/znb/data/videoMAV/videoData'
#dataset_path = '/home/dwqdx3/my_drive_2/public_data/videoData_shuffle'
# dataset_path = '/public/home/cit_fluo/znb/data/videoAction/RGB'
# dataset_path = '/public/home/cit_fluo/znb/data/videoAction/small/RGB'  # small 
# dataset_path = '/public/home/cit_fluo/znb/data/videoAction/middle/RGB'  # middle  
dataset_path = '/public/home/cit_fluo/znb/data/videoAction/remote/RGB'  # remote
midPath = os.path.join(dataset_path, '**', '*.*')
video_paths = glob(os.path.join(dataset_path, '**', '*.*'), recursive=True)
labels = [os.path.basename(os.path.dirname(path)) for path in video_paths]

num_classes = len(set(labels))
class_to_index = {l: idx for idx, l in enumerate(set(labels))}
index_to_class = {idx: l for idx, l in enumerate(set(labels))}

trainval_videos, test_videos, trainval_labels, test_labels = train_test_split(video_paths, labels,
                                                                              test_size=test_split_size,
                                                                              random_state=42)
train_videos, val_videos, train_labels, val_labels = train_test_split(trainval_videos, trainval_labels,
                                                                      test_size=val_split_size,
                                                                      random_state=42)

create_csv_files = True

if create_csv_files:
    create_csv(train_videos, train_labels, "Small_trainCost")
    create_csv(val_videos, val_labels, "Small_valCost")
    create_csv(test_videos, test_labels, "Small_testCost")

train_csv_path = "/public/home/cit_fluo/znb/source/videoWork/Small_trainCost.csv"
val_csv_path = "/public/home/cit_fluo/znb/source/videoWork/Small_valCost.csv"
test_csv_path = "/public/home/cit_fluo/znb/source/videoWork/Small_testCost.csv"

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

df = pd.concat([train_df, val_df], axis=0)
df = pd.concat([df, test_df], axis=0)
num_samples = df.shape[0]
class_counts = df.label.value_counts().to_dict()

plot_class_distribution(class_counts, 'Workout Video Dataset - Classes Distribution')


class_weights = {cat: num_samples / count for cat, count in class_counts.items()}
all_weights = [class_weights[label] for label in train_df['label'].values]

n_samples = len(all_weights)
sampler = WeightedRandomSampler(weights=all_weights, num_samples=n_samples, replacement=False)

train_dataset = WorkoutDataset(train_df['path'].values, train_df['label'].values, transform=transform)
val_dataset = WorkoutDataset(val_df['path'].values, val_df['label'].values, transform=transform)
test_dataset = WorkoutDataset(test_df['path'].values, test_df['label'].values, transform=transform)

balanced_train = False

if balanced_train:
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=8, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)



class WorkoutClassifier(nn.Module):
    def __init__(self, num_classes=22):
        super(WorkoutClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Identity()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, flows):
        batch_size, num_frames, c, h, w = x.shape
        # x shape torch.Size([8, 30, 3, 128, 128])
        
        x = x.view(-1, c, h, w)
        x = self.base_model(x)
        # 240 x 512
        #print(flows.shape)
        x = x.view(batch_size, num_frames, -1)
       # print(x.shape) # torch.Size([8, 30, 512])
        x = torch.mean(x, dim=1)  # Temporal average
       # print(x.shape) # torch.Size([8, 512])
        x = self.fc1(x)
        x = self.fc2(x)
        return x


load_balanced_model = False
model_path = "models/final_model.pth"
balanced_model_path = "models/balanced_model.pth"

print('The number of classes is : ', num_classes)
# model = WorkoutClassifier(num_classes=num_classes)
# model = C3D(num_classes=num_classes, pretrained=False)
model = Cost(num_classes, costblock, [2,3, 4, 2])
if not torch.cuda.is_available():
    if load_balanced_model:
        model.load_state_dict(torch.load(balanced_model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

else:
    print("Started GPU Acceleration")
    if load_balanced_model:
       # model.load_state_dict(torch.load(balanced_model_path))
        pass
    else:
       # model.load_state_dict(torch.load(model_path))
        pass
    model.to(device)


lr = 1e-4
momentum = 0.8
weight_decay = 5e-4
gamma = 0.1
step_size_decay = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_decay, gamma=gamma)

train_losses = []
val_losses = []
val_accuracies = []

num_epochs = 15
max_gained_acc = 0
epoch_of_max_acc = 0

early_stopping = False
epochs_tolerance = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, flows, labels in tqdm(train_loader):
        inputs, flows, labels = inputs.to(device), flows.to(device), labels.to(device)
       # print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs, flows)
        # print(outputs.shape)
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
    with torch.no_grad():
        for inputs, flows, labels in val_loader:
            inputs, flows, labels = inputs.to(device), flows.to(device), labels.to(device)
            outputs = model(inputs, flows)
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
        max_gained_acc = round(val_accuracy, 4)

    print(f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}%')

    if val_accuracy >= max_gained_acc:
        max_gained_acc = val_accuracy
        epoch_of_max_acc = epoch
    if early_stopping:
        if epoch - epoch_max_acc > epochs_tolerance:
            break

# plot_results(train_losses, val_losses, val_accuracies)

def test():
    test_losses = []
    test_accuracies = []
    test_loss = 0.0
    correct = 0
    total = 0
    cum_labels = []
    cum_predicted = []

    show_gifs_results = True
    show_gifs_at = 5

    model.eval()
    with torch.no_grad():
        for i, (inputs, flows, labels) in enumerate(test_loader):
            inputs, flows, labels = inputs.to(device), flows.to(device), labels.to(device)
            outputs = model(inputs, flows)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_loss /= len(test_loader)
            test_accuracy = 100 * ((predicted == labels).sum().item() / labels.size(0))
            test_losses.append(loss)
            test_accuracies.append(test_accuracy)
            cum_labels.extend(labels)
            cum_predicted.extend(predicted)
            if show_gifs_results and i % show_gifs_at == 0:
                show_gifs(inputs, labels, predicted)


test_losses = []
test_accuracies = []
test_loss = 0.0
correct = 0
total = 0
cum_labels = []
cum_predicted = []

show_gifs_results = True
show_gifs_at = 5

model.eval()
with torch.no_grad():
    for i, (inputs, flows, labels) in enumerate(test_loader):
        inputs, flows, labels = inputs.to(device), flows.to(device), labels.to(device)
        outputs = model(inputs, flows)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * ((predicted == labels).sum().item() / labels.size(0))
        test_losses.append(loss)
        test_accuracies.append(test_accuracy)
        cum_labels.extend(labels)
        cum_predicted.extend(predicted)
        if show_gifs_results and i % show_gifs_at == 0:
            pass
            # show_gifs(inputs, labels, predicted)

print('test accuracy:', np.mean(test_accuracies))

test_losses_cpu = [loss.cpu().numpy() for loss in test_losses]
actual_labels = [l.cpu().numpy() for l in cum_labels]
model_predictions = [p.cpu().numpy() for p in cum_predicted]

get_metrics(test_losses_cpu,test_accuracies, actual_labels, model_predictions)

save_model = False
last_model_state_dict = model.state_dict()
model_path = "models1/"
if save_model:
    torch.save(model.state_dict(), model_path + 'final_model.pth')
