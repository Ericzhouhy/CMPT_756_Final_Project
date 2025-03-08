import requests
from PIL import Image
from io import BytesIO

# URL of the deployed service
service_url = "https://cifar100-resnet18-656944472679.us-central1.run.app/predict"

# URL of the image to test
image_url = "https://www.worldanimalprotection.ca/cdn-cgi/image/width=1280,format=auto/siteassets/shutterstock_2311286833.jpg"

# Download the image
response = requests.get(image_url)
image_bytes = response.content

# Display the image
image = Image.open(BytesIO(image_bytes))
image.show()

# Send the image to the service for prediction
files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
response = requests.post(service_url, files=files)

# Print the response from the server
print(response.json())
