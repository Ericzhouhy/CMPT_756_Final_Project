# Use the official PyTorch image as a base (adjust version as needed)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any additional dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run your training script
CMD ["python", "train.py"]
