# Use the official PyTorch image as a base (adjust version as needed)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install required Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask for serving the training process
RUN pip install flask

# Copy your code into the container
COPY . /app

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Create a simple Flask app to run the training in the background and handle health checks
RUN echo 'from flask import Flask\nimport threading\nimport time\nimport torch\nimport torchvision\nimport torch.optim as optim\nimport torch.nn as nn\nfrom torchvision.models import resnet18\nfrom torch.utils.data import DataLoader\nfrom torch.utils.tensorboard import SummaryWriter\n\napp = Flask(__name__)\n\n@app.route("/")\ndef index():\n    return "Training is running..."\n\ndef run_training():\n    # Insert your training logic here\n    print("Training started...")\n    # Simulate a training loop (replace with your actual training function)\n    # training function would be your train_model function\n    # train_model(model, train_loader, criterion, optimizer, num_epochs=50, resume=False, checkpoint_dir="runs/cifar100_resnet18")\n    time.sleep(5)\n    print("Training completed...")\n\nif __name__ == "__main__":\n    threading.Thread(target=run_training).start()\n    app.run(host="0.0.0.0", port=8080, threaded=True, debug=True)' > app.py

# Set the command to run the Flask app
CMD ["python", "app.py"]
