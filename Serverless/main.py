import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from google.cloud import storage
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# **Download the model from Google Cloud Storage (GCS)**
def download_model_from_gcs(bucket_name, model_path, local_path="/tmp/model.pth"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename(local_path)
    print("Model downloaded from GCS!")

# **GCS bucket information**
BUCKET_NAME = "cmpt756-model-bucket"
MODEL_PATH = "model.pth"

# **Download and load the model**
download_model_from_gcs(BUCKET_NAME, MODEL_PATH)

# **Create the model instance**
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for CIFAR-100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
checkpoint = torch.load("/tmp/model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(checkpoint['model_state_dict'])  # Load state dict from the checkpoint
model.eval()

transform = transforms.Compose([
    # Todo add other types of augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 normalization values
])

batch_size = 64
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def get_correct_predictions(model, loader, max_samples=10):
    model.eval()
    correct_samples = []
    correct_labels = []
    correct_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct_indices = (preds == labels).nonzero(as_tuple=True)[0]

            for idx in correct_indices:
                if len(correct_samples) < max_samples:
                    correct_samples.append(inputs[idx].cpu())
                    correct_labels.append(labels[idx].cpu().item())
                    correct_preds.append(preds[idx].cpu().item())
                else:
                    return correct_samples, correct_labels, correct_preds

    return correct_samples, correct_labels, correct_preds


def visualize_correct_predictions(model, loader, class_names, max_samples=10):
    correct_samples, correct_labels, correct_preds = get_correct_predictions(
        model, loader, max_samples=max_samples
    )

    fig, axes = plt.subplots(1, len(correct_samples), figsize=(15, 5))
    if len(correct_samples) == 1:
        axes = [axes]

    for idx, (img, true_label, pred_label) in enumerate(
        zip(correct_samples, correct_labels, correct_preds)
    ):
        img = img.permute(1, 2, 0)
        img = img * 0.2673 + 0.5071  # Unnormalize for CIFAR100
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            fontsize=10,
        )
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

class_names=train_dataset.classes
visualize_correct_predictions(model, test_loader, class_names, max_samples=10)
