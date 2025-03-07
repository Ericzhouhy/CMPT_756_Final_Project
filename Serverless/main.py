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

'''
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for CIFAR-100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
checkpoint = torch.load("/tmp/model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(checkpoint['model_state_dict'])  # Load state dict from the checkpoint
model.eval()'''



# CIFAR-100 Labels Dictionary
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

# Load ONNX model from GCS
def load_onnx_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)
    
    # Download ONNX model
    model_bytes = blob.download_as_bytes()
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_bytes)
    
    return ort.InferenceSession("/tmp/model.onnx")

# Global ONNX model instance
ONNX_MODEL = load_onnx_model()

# Preprocess image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))  # CIFAR-100 input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Postprocess output
def postprocess_output(output):
    predicted_class = np.argmax(output, axis=1)[0]  # Get index with highest probability
    class_name = CIFAR100_LABELS.get(predicted_class, "Unknown")  # Map index to label
    return class_name

# Cloud Function entry point
@functions_framework.http
def predict(request):
    try:
        request_json = request.get_json()

        # Read image from URL
        image_url = request_json.get("image_url")
        if not image_url:
            return json.dumps({"error": "Missing image_url"}), 400, {"Content-Type": "application/json"}
        
        response = requests.get(image_url)
        image_bytes = response.content

        # Preprocess and predict
        input_tensor = preprocess_image(image_bytes)
        output = ONNX_MODEL.run(None, {"input": input_tensor})[0]  # Adjust input key if needed
        
        # Postprocess result
        result = postprocess_output(output)

        return json.dumps({"prediction": result}), 200, {"Content-Type": "application/json"}
    
    except Exception as e:
        return json.dumps({"error": str(e)}), 400, {"Content-Type": "application/json"}

