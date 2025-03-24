import torch
import torchvision
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchmetrics
from google.cloud import storage
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GCS_BUCKET_NAME = "cmpt756-model-bucket"
GCS_CHECKPOINT_PATH = "checkpoints/latest_checkpoint.pth"
LOCAL_CHECKPOINT_PATH = "/tmp/latest_checkpoint.pth"

def get_model():
    model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model.to(device)

def get_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='/tmp', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

def save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss_metric.compute().item(),
        'accuracy': accuracy_metric.compute().item()
    }, LOCAL_CHECKPOINT_PATH)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_CHECKPOINT_PATH)
    blob.upload_from_filename(LOCAL_CHECKPOINT_PATH)

def load_checkpoint(model, optimizer=None):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_CHECKPOINT_PATH)
    if blob.exists():
        blob.download_to_filename(LOCAL_CHECKPOINT_PATH)
        ckpt = torch.load(LOCAL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['epoch']
    return 0

def train(request):
    model = get_model()
    train_loader = get_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_metric = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)

    start_epoch = load_checkpoint(model, optimizer)
    model.train()

    for epoch in range(start_epoch, start_epoch + 1):  # Only train 1 epoch per trigger
        loss_metric.reset()
        acc_metric.reset()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_metric.update(loss)
            acc_metric.update(outputs, labels)

    save_checkpoint(model, optimizer, epoch, loss_metric, acc_metric)

    return {
        "message": "Training completed.",
        "epoch": epoch + 1,
        "loss": loss_metric.compute().item(),
        "accuracy": acc_metric.compute().item() * 100
    }
