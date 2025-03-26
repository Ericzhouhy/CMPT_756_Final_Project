# Use the official PyTorch image as a base (adjust version as needed)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install google-cloud-logging

# Copy your code into the container
COPY . /app

# Expose port 8080 for any potential communication
EXPOSE 8080

# Set the command to run the training script
CMD ["python", "train.py"]
