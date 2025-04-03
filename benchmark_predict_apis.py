import time
import requests
import csv

# Two API endpoints
serverless_url = "https://us-central1-cmpt756-deploy-project.cloudfunctions.net/cifar100_predict"
containerized_url = "https://cifar100-api-4pxvf6xeea-uc.a.run.app/predict"

# Read image URLs from the file
with open("image_urls.txt", "r") as f:
    image_urls = [line.strip() for line in f if line.strip()]

# Prepare CSV output for logging results
with open("results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "type", "image_url", "status_code", "response_time_sec"])

    for idx, url in enumerate(image_urls):
        # Test the serverless API
        try:
            start = time.time()
            res = requests.post(
                serverless_url,
                json={"url": url},
                timeout=20
            )
            duration = time.time() - start
            writer.writerow([idx, "serverless", url, res.status_code, round(duration, 3)])
        except Exception as e:
            writer.writerow([idx, "serverless", url, "ERROR", "N/A"])

        # Test the containerized API
        try:
            start = time.time()
            res = requests.post(
                containerized_url,
                data={"image_url": url},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=20
            )
            duration = time.time() - start
            writer.writerow([idx, "containerized", url, res.status_code, round(duration, 3)])
        except Exception as e:
            writer.writerow([idx, "containerized", url, "ERROR", "N/A"])
