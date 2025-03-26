from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from google.cloud import storage
import requests
import os
import traceback

app = Flask(__name__)

# Label map
CIFAR100_LABELS = {i: name for i, name in enumerate([
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
])}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

# Global model object
model = None

def load_model():
    global model
    try:
        model_path = "/tmp/model.pth"
        bucket_name = "cmpt756-model-bucket"
        model_blob_name = "checkpoint_epoch_100.pth"

        if not os.path.exists(model_path):
            print("Downloading model from GCS...")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(model_blob_name)
            blob.download_to_filename(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading model...")
        model_loaded = resnet18(pretrained=False)
        model_loaded.fc = nn.Linear(model_loaded.fc.in_features, 100)

        checkpoint = torch.load(model_path, map_location=device)
        model_loaded.load_state_dict(checkpoint['model_state_dict'])
        model_loaded.to(device)
        model_loaded.eval()

        model = model_loaded
        print("Model loaded on", device)
    except Exception as e:
        print("Model load failed:", e)
        traceback.print_exc()

# Immediately load the model at startup
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
        elif 'image_url' in request.form:
            image_url = request.form['image_url']
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            return jsonify({'error': 'No image or image_url provided'}), 400

        device = next(model.parameters()).device
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = CIFAR100_LABELS[predicted.item()]

        return jsonify({'prediction': label})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
