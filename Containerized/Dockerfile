FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
