from fastapi import FastAPI, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from google.cloud import storage
import io

app = FastAPI()

# Google Cloud Storage Information
BUCKET_NAME = "cmpt756-model-bucket"
MODEL_PATH = "cifar100_resnet18_opset12.onnx"

# CIFAR-100 Labels Dictionary (as defined in your friend's code)
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

def load_onnx_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)

    local_path = "/tmp/modified_model.onnx"
    blob.download_to_filename(local_path)
    print("Model downloaded from GCS:", local_path)

    return ort.InferenceSession(local_path)

ONNX_MODEL = load_onnx_model()

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_output(output):
    predicted_class = np.argmax(output, axis=1)[0]
    class_name = CIFAR100_LABELS.get(predicted_class, "Unknown")
    return class_name

@app.post("/predict/")
async def predict(image_url: str):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")

        image_bytes = response.content
        input_tensor = preprocess_image(image_bytes)
        input_name = ONNX_MODEL.get_inputs()[0].name
        output = ONNX_MODEL.run(None, {input_name: input_tensor})[0]

        result = postprocess_output(output)

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
