# !pip install torch torchvision tensorboard torchmetrics

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import glob
import torch.onnx
import onnx
import os

# Define data transformations
transform = transforms.Compose([
    # Todo add other types of augmentation
    transforms.RandomHorizontalFlip(),  # Add horizontal flipping
    transforms.RandomRotation(10),     # Add random rotations (up to 10 degrees)
    transforms.RandomCrop(32, padding=4),  # Add random cropping with padding
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 normalization values
])

# Load CIFAR-100 dataset
batch_size = 64
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Define the model
model = resnet18(pretrained=True)  # Set num_classes to 100 for CIFAR-100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) # set the model on cuda
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.fc.parameters(), lr=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)



# Function to save a checkpoint in the TensorBoard log directory
def save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric, checkpoint_dir='runs/cifar100_resnet18'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"

    # Use .compute() to get the values of the metrics
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss_metric.compute(),  # Get the computed loss
        'accuracy': accuracy_metric.compute()  # Get the computed accuracy
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

# Function to load a checkpoint from the TensorBoard log directory
def load_checkpoint(model, optimizer, checkpoint_dir='runs/cifar100_resnet18'):
    # Find the latest checkpoint (e.g., based on the highest epoch number)
    checkpoint_paths = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pth")
    checkpoint_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoint_paths[-1]

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f'Checkpoint loaded from {latest_checkpoint}. Resuming training from epoch {start_epoch}')

    return start_epoch, loss, accuracy  # You can return these for reference but don't update accuracy_metric with them




# Updated training function with torchmetrics and resuming capability
def train_model(model, train_loader, criterion, optimizer, num_epochs=1, start_epoch=0, resume=False, checkpoint_dir='runs/cifar100_resnet18'):
    # Initialize the metrics
    loss_metric = torchmetrics.MeanMetric().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)

    # Load checkpoint if resuming
    if resume:
        start_epoch, loss, accuracy = load_checkpoint(model, optimizer, checkpoint_dir='runs/cifar100_resnet18')
        loss_metric.update(torch.tensor(loss))

    model.train()
    for epoch in range(start_epoch, num_epochs):
        loss_metric.reset()
        accuracy_metric.reset()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_metric.update(loss.item())
            accuracy_metric.update(outputs, labels)
            if i%100 == 0:
              avg_loss = loss_metric.compute().item()
              avg_acc = accuracy_metric.compute().item()
              avg_acc = avg_acc * 100
              print(f"Epoch [{epoch}]: Loss={avg_loss}, Accuracy={avg_acc:.2f}")

        save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric, checkpoint_dir='runs/cifar100_resnet18')


def evaluate_model(model, test_loader, criterion, optimizer, checkpoint_dir='runs/cifar100_resnet18'):
    # Load the model from the latest checkpoint if needed
    load_checkpoint(model, optimizer, checkpoint_dir)

    loss_metric = torchmetrics.MeanMetric().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_metric.update(loss.item())
            accuracy_metric.update(outputs, labels)

        avg_loss = loss_metric.compute()
        avg_acc = accuracy_metric.compute()

        return avg_loss, avg_acc*100
    

#train_model(model, train_loader, criterion, optimizer, num_epochs=100, resume=True, checkpoint_dir='runs/cifar100_resnet18')
train_model(model, train_loader, criterion, optimizer, num_epochs=50, resume=False, checkpoint_dir='runs/cifar100_resnet18')

evaluate_model(model, test_loader, criterion, optimizer, checkpoint_dir='runs/cifar100_resnet18')