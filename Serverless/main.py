import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from google.cloud import storage

# **Define SimpleModel (must be the same as the one used during training)**
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3072, 100)  # Ensure this matches the training model shape

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x

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
state_dict = torch.load("/tmp/model.pth", map_location=torch.device("cpu"), weights_only=True)

# **Create the model instance**
model = SimpleModel()  # ✅ Ensure this matches the training model
model.load_state_dict(state_dict)
model.eval()

# **Flask API for prediction**
def predict(request):
    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features'"}), 400

        # Convert input data to Tensor
        features = torch.tensor([data["features"]], dtype=torch.float32)
        prediction = model(features).tolist()

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
