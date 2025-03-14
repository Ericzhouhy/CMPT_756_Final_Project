import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchmetrics
from flask import Flask, request, jsonify
import io
from io import BytesIO
import requests
from PIL import Image
import threading
from google.cloud import storage

# Initialize Flask application
app = Flask(__name__)

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Google Cloud Storage Configuration
GCS_BUCKET_NAME = "cmpt756-model-bucket"  # Replace with your actual GCS bucket name
GCS_CHECKPOINT_PATH = "checkpoints/latest_checkpoint.pth"  # Always overwrite the latest checkpoint
TMP_CHECKPOINT_DIR = "/tmp"  # Temporary storage directory
LOCAL_CHECKPOINT_PATH = f"{TMP_CHECKPOINT_DIR}/latest_checkpoint.pth"

# CIFAR-100 class label mapping
CIFAR100_LABELS = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
    85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
    90: "train", 91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe",
    95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm"
}

# Data preprocessing transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

# Load CIFAR-100 dataset
batch_size = 64
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define ResNet18 model
model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)

# Function to save checkpoint and overwrite the previous one
def save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss_metric.compute().item(),
        'accuracy': accuracy_metric.compute().item()
    }

    # Save locally
    torch.save(checkpoint, LOCAL_CHECKPOINT_PATH)
    print(f'Checkpoint saved locally at {LOCAL_CHECKPOINT_PATH}')

    # Upload to Google Cloud Storage and overwrite the previous checkpoint
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_CHECKPOINT_PATH)
    blob.upload_from_filename(LOCAL_CHECKPOINT_PATH)
    print(f"Checkpoint uploaded to GCS: gs://{GCS_BUCKET_NAME}/{GCS_CHECKPOINT_PATH}")

# Function to load the latest model checkpoint from GCS
def load_latest_checkpoint(model, optimizer=None):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_CHECKPOINT_PATH)

    if not blob.exists():
        print("No checkpoint found in GCS. Starting from scratch.")
        return 0

    # Download latest checkpoint
    blob.download_to_filename(LOCAL_CHECKPOINT_PATH)
    print(f"Checkpoint downloaded from GCS: gs://{GCS_BUCKET_NAME}/{GCS_CHECKPOINT_PATH}")

    checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5, resume=True):
    loss_metric = torchmetrics.MeanMetric().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)

    start_epoch = 0
    if resume:
        start_epoch = load_latest_checkpoint(model, optimizer)
        print(f"Resuming training from epoch {start_epoch+1}")

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

            loss_metric.update(loss)
            accuracy_metric.update(outputs, labels)

            if i % 100 == 0:
                avg_loss = loss_metric.compute().item()
                avg_acc = accuracy_metric.compute().item() * 100
                print(f"Epoch [{epoch+1}/{num_epochs}]: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")

        save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric)

    print("Training completed!")

# API to start training
@app.route("/start_train", methods=["POST"])
def start_training():
    training_thread = threading.Thread(target=train_model, args=(model, train_loader, criterion, optimizer, 5))
    training_thread.start()
    return jsonify({"message": "Training started"}), 200

# API to evaluate model
@app.route("/evaluate", methods=["GET"])
def evaluate():
    load_latest_checkpoint(model)
    loss_metric = torchmetrics.MeanMetric().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_metric.update(loss)
            accuracy_metric.update(outputs, labels)

    avg_loss = loss_metric.compute()
    avg_acc = accuracy_metric.compute() * 100
    return jsonify({"loss": avg_loss.item(), "accuracy": avg_acc.item()}), 200

# API to predict an image
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "image_url" in data:
        try:
            response = requests.get(data["image_url"])
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException:
            return jsonify({"error": "Invalid image URL"}), 400
    elif "file" in request.files:
        image = Image.open(request.files["file"]).convert("RGB")
    else:
        return jsonify({"error": "No image uploaded"}), 400

    image = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = CIFAR100_LABELS[predicted.item()]

    return jsonify({"prediction": label})

# Load latest checkpoint on startup
load_latest_checkpoint(model)

# Start Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
