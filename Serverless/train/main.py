import os
import threading
import logging
from flask import Flask, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchmetrics
from google.cloud import storage
import logging
import sys

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_status = {
    "in_progress": False,
    "current_epoch": 0,
    "total_epochs": 50,
    "last_log": ""
}

# ---------- Config ----------
LOCAL_CHECKPOINT_PATH = "checkpoint.pth"
GCS_BUCKET_NAME = "cmpt756-model-bucket"
GCS_CHECKPOINT_PATH = "checkpoints/resnet18_cifar100.pth"
GCS_DATA_PREFIX = "cifar-100-python/"
LOCAL_DATA_PATH = "/tmp/cifar-100-python/"
batch_size = 64

# ---------- Data Transform ----------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

# ---------- GCS: Download Dataset ----------
def download_dir_from_gcs(bucket_name, gcs_dir, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_dir)

    for blob in blobs:
        relative_path = blob.name[len(gcs_dir):].lstrip('/')
        if not relative_path:
            continue
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded {blob.name} to {local_path}")

def maybe_download_dataset():
    if not os.path.exists(os.path.join(LOCAL_DATA_PATH, "train")):
        logging.info("Dataset not found locally. Downloading from GCS...")
        download_dir_from_gcs(GCS_BUCKET_NAME, GCS_DATA_PREFIX, LOCAL_DATA_PATH)
        logging.info(f"Dataset downloaded to {LOCAL_DATA_PATH}")
    else:
        logging.info("Dataset already exists locally. Skipping download.")

# ---------- Model ----------
model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# ---------- Optimizer & Loss ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)

# ---------- Save & Load Checkpoints ----------
def save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss_metric.compute().item(),
        'accuracy': accuracy_metric.compute().item()
    }
    torch.save(checkpoint, LOCAL_CHECKPOINT_PATH)
    logging.info(f"Checkpoint saved locally at {LOCAL_CHECKPOINT_PATH}")

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_CHECKPOINT_PATH)
        blob.upload_from_filename(LOCAL_CHECKPOINT_PATH)
        logging.info(f"Checkpoint uploaded to GCS: gs://{GCS_BUCKET_NAME}/{GCS_CHECKPOINT_PATH}")
    except Exception as e:
        logging.error(f"Failed to upload checkpoint to GCS: {e}")

def load_latest_checkpoint(model, optimizer=None):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_CHECKPOINT_PATH)

    if not blob.exists():
        logging.warning("No checkpoint found in GCS. Starting from scratch.")
        return 0

    blob.download_to_filename(LOCAL_CHECKPOINT_PATH)
    logging.info(f"Checkpoint downloaded from GCS: gs://{GCS_BUCKET_NAME}/{GCS_CHECKPOINT_PATH}")

    checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']

# ---------- Training ----------
def train_model(model, train_loader, criterion, optimizer, num_epochs=50, resume=True):
    global training_status
    training_status["in_progress"] = True
    training_status["total_epochs"] = num_epochs

    loss_metric = torchmetrics.MeanMetric().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)

    try:
        start_epoch = 0
        if resume:
            start_epoch = load_latest_checkpoint(model, optimizer)
            logging.info(f"Resuming training from epoch {start_epoch + 1}")

        model.train()
        for epoch in range(start_epoch, num_epochs):
            training_status["current_epoch"] = epoch + 1
            logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
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

                if i % 10 == 0:
                    log_msg = (f"Batch {i}: Loss={loss_metric.compute().item():.4f}, "
                               f"Accuracy={accuracy_metric.compute().item() * 100:.2f}%")
                    logging.info(log_msg)
                    training_status["last_log"] = log_msg

            # save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric)

        logging.info("All epochs completed. Training finished.")
        training_status["in_progress"] = False
        training_status["last_log"] = "Training completed."

    except Exception as e:
        logging.exception("Training crashed.")
        training_status["in_progress"] = False
        training_status["last_log"] = f"Training failed: {e}"


# ---------- Flask API ----------
@app.route("/start_train", methods=["POST"])
def start_training():
    maybe_download_dataset()
    train_dataset = CIFAR100(root='/tmp', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    training_thread = threading.Thread(target=train_model, args=(model, train_loader, criterion, optimizer, 50))
    training_thread.start()
    return jsonify({"message": "Training started"}), 200

@app.route("/evaluate", methods=["GET"])
def evaluate():
    maybe_download_dataset()
    test_dataset = CIFAR100(root='/tmp', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    avg_loss = loss_metric.compute().item()
    avg_acc = accuracy_metric.compute().item() * 100
    logging.info(f"Evaluation completed - Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")
    return jsonify({"loss": avg_loss, "accuracy": avg_acc}), 200

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(training_status)

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
