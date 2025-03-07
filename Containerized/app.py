from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# Load the ONNX model
session = ort.InferenceSession("cifar100_resnet18_opset12.onnx")

# Define a data model for the input
class DataModel(BaseModel):
    data: list

@app.get("/")
async def root():
    return {"message": "Hello, this is the ONNX model API"}

@app.post("/predict/")
async def predict(data: DataModel):
    # Convert input data to numpy array
    input_data = np.array(data.data, dtype=np.float32).reshape(1, 3, 32, 32)  # Assuming CIFAR-100 shape
    
    # Run inference
    inputs = {session.get_inputs()[0].name: input_data}
    output = session.run(None, inputs)
    
    # Return the prediction
    return {"prediction": output[0].tolist()}
