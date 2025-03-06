import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from google.cloud import storage

# **定义模型架构（必须和训练时一致）**
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# **从 GCS 下载模型**
def download_model_from_gcs(bucket_name, model_path, local_path="/tmp/model.pth"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename(local_path)
    print("Model downloaded from GCS!")

# **GCS 存储桶信息**
BUCKET_NAME = "cmpt756-model-bucket"  # ✅ 确保名称正确
MODEL_PATH = "model.pth"

# **下载并加载模型**
download_model_from_gcs(BUCKET_NAME, MODEL_PATH)
state_dict = torch.load("/tmp/model.pth", map_location=torch.device("cpu"), weights_only=True)
model = MyModel()
model.load_state_dict(state_dict)
model.eval()

def predict(request):
    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features'"}), 400

        # 转换输入数据为 Tensor
        features = torch.tensor([data["features"]], dtype=torch.float32)
        prediction = model(features).tolist()

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
