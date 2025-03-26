# Use the official Python image as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install torch torchvision flask google-cloud-storage onnxruntime requests pillow

# Expose port 8080 for the Flask app
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
