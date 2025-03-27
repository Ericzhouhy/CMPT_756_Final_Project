import torch
import torch.nn as nn
from torchvision.models import resnet18
from flask import jsonify
from PIL import Image
import torchvision.transforms as transforms
import io
import werkzeug
werkzeug.werkzeug_url_quote = werkzeug.urls.url_quote  # Workaround for compatibility
from flask import Flask, jsonify

# CIFAR-100 labels (same as your local version)
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

# Initialize model (loaded once at cold start)
model = None
device = torch.device('cpu')

def load_model():
    global model  # Important: Declare we're using the global variable
    if model is None:
        print("Loading model...")
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 100)
        checkpoint = torch.load('checkpoint_epoch_100.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    return model

# Image transformations
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

def cifar100_predict(request):  # << MUST match deployment name exactly
    """HTTP Cloud Function entry point"""
    global model
    if model is None:
        model = load_model()
    
    if request.method != 'POST':
        return jsonify({'error': 'Only POST requests accepted'}), 405
    
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image provided'}), 400
        
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return jsonify({
            'prediction': CIFAR100_LABELS[predicted.item()],
            'class_id': predicted.item()
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
def predict(request):
    """HTTP Cloud Function for predictions"""
    global model
    if model is None:
        model = load_model()
    
    if request.method != 'POST':
        return jsonify({'error': 'Only POST requests accepted'}), 405
    
    try:
        # Get image file from request
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return jsonify({
            'prediction': CIFAR100_LABELS[predicted.item()],
            'class_id': predicted.item()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    