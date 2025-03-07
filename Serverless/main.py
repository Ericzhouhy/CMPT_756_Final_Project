import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response
from google.cloud import storage
from torchvision.models import resnet18
import numpy as np
from PIL import Image
import io
import requests
import onnxruntime as ort
import functions_framework
import json

# Google Cloud Storage (GCS) Information
BUCKET_NAME = "cmpt756-model-bucket"
MODEL_PATH = "cifar100_resnet18_opset12.onnx"  # Ensure the correct ONNX model filename in GCS

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

# Function to download the ONNX model from GCS
def load_onnx_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)

    # Download ONNX model
    local_path = "/tmp/modified_model.onnx"
    blob.download_to_filename(local_path)
    print("Model downloaded from GCS:", local_path)

    return ort.InferenceSession(local_path)

# Load ONNX model globally
ONNX_MODEL = load_onnx_model()

# Preprocess image before inference
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))  # CIFAR-100 input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Postprocess ONNX output
def postprocess_output(output):
    predicted_class = np.argmax(output, axis=1)[0]  # Get index with highest probability
    class_name = CIFAR100_LABELS.get(predicted_class, "Unknown")  # Map index to label
    return class_name

# Cloud Function entry point
@functions_framework.http
def predict(request):
    try:
        request_json = request.get_json()

        # Validate input
        if "image_url" not in request_json:
            return jsonify({"error": "Missing 'image_url'"}), 400

        image_url = request_json["image_url"]
        response = requests.get(image_url)

        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        image_bytes = response.content
        print("Image downloaded successfully")

        # Preprocess and predict
        input_tensor = preprocess_image(image_bytes)
        input_name = ONNX_MODEL.get_inputs()[0].name  # Dynamically get the correct input name
        output = ONNX_MODEL.run(None, {input_name: input_tensor})[0]

        
        # Postprocess result
        result = postprocess_output(output)

        return jsonify({"prediction": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Local Testing
if __name__ == "__main__":
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def test_predict():
        test_request_data = json.dumps({
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/36/United_Airlines_Boeing_777-200_Meulemans.jpg"
        })

        test_request = app.test_request_context(
            path="/predict", method="POST", data=test_request_data, content_type="application/json"
        )

        with test_request:
            response, status_code = predict(request)  # Extract tuple correctly

        print("🔍 Prediction Response:", response)  # Debugging: prints full response tuple

        if isinstance(response, Response):  # If Flask response, use .get_json()
            response_json = response.get_json()
        else:
            response_json = json.loads(response.data.decode("utf-8"))  # Extract manually if not Flask Response

        print(f"✅ Status Code: {status_code}")
        print(f"✅ JSON Response: {response_json}")

    # Run the test
    test_predict()


