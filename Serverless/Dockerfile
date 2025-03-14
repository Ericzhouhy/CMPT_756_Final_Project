# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --upgrade "typing-extensions>=4.8.0" \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application files
COPY . .

# Expose port 8080 for the Flask app
ENV PORT=8080

# Start the Gunicorn server
CMD exec gunicorn --bind 0.0.0.0:8080 --workers=1 --threads=8 main:app
    
    # The Dockerfile is a text file that contains all the commands a user could call on the command line to assemble an image. Using the Dockerfile, we can build a Docker image that contains all the dependencies and code needed to run the application. 
    # The Dockerfile starts with the  FROM  command, which specifies the base image to use. In this case, we are using the official PyTorch image as the base image. 
    # Next, the  WORKDIR  command sets the working directory in the container to  /app . 
    # The  COPY  command copies the current directory contents into the container at  /app . 
    # The  RUN  command installs the dependencies specified in the  requirements.txt  file. 
    # The  ENV  command sets the environment variable  PORT  to  8080 . 
    # Finally, the  CMD  command specifies the command to run when the container starts. In this case, we are using  gunicorn  to run the Flask application. 
    # Step 3: Build the Docker Image 
    # To build the Docker image, run the following command in the terminal: 
    # docker build -t serverless-app . 
    # This command builds a Docker image with the tag  serverless-app  using the Dockerfile in the current directory. 
    # Step 4: Run the Docker Container 
    # To run the Docker container, use the following command: 
    # docker run -p 8080:8080 serverless-app 
    # This command starts a Docker container from the  serverless-app  image and maps port 8080 on the host to port 8080 on the container. 
    # You should see output similar to the following: 
    # [2022-02-22 15:00:00 +0000] [1] [INFO] Starting gunicorn 20.1.0
    # [2022-02-22 15:00:00 +0000] [1] [INFO] Listening at: http://

