import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import argparse
import os
from google.cloud import storage

def train():
    # Parse args from Vertex AI
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gcs-bucket', type=str, required=True)
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)
    
    # Data
    transform = transforms.Compose([...])  # Your existing transforms
    train_data = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Save checkpoint to GCS
        checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        
        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(args.gcs_bucket)
        blob = bucket.blob(f'models/{checkpoint_path}')
        blob.upload_from_filename(checkpoint_path)