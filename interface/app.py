from flask import Flask, request, render_template
import requests
import time

app = Flask(__name__)

# Define the endpoints for serverless and containerized predictions
CONTAINERIZED_URL = 'https://cifar100-api-4pxvf6xeea-uc.a.run.app/predict'
SERVERLESS_URL = 'https://us-central1-erics-first-project-450223.cloudfunctions.net/cifar100_predict'

@app.route('/', methods=['GET', 'POST'])
def home():
    serverless_prediction = None
    containerized_prediction = None
    serverless_time = None
    containerized_time = None
    image_url = None

    if request.method == 'POST':
        image_url = request.form['image_url']
        
        # Get containerized prediction
        containerized_start_time = time.time()
        containerized_prediction = get_containerized_prediction(image_url)
        containerized_time = time.time() - containerized_start_time
        
        # Get serverless prediction
        serverless_start_time = time.time()
        serverless_prediction = get_serverless_prediction(image_url)
        serverless_time = time.time() - serverless_start_time

    return render_template('index.html', image_url=image_url, 
                           serverless_prediction=serverless_prediction,
                           containerized_prediction=containerized_prediction,
                           serverless_time=serverless_time,
                           containerized_time=containerized_time)

def get_containerized_prediction(image_url):
    try:
        response = requests.post(CONTAINERIZED_URL, data={'image_url': image_url})
        if response.status_code == 200:
            return response.json().get('prediction', 'No prediction found')
    except requests.exceptions.RequestException as e:
        return f'Error: {str(e)}'
    return 'Error in prediction'

def get_serverless_prediction(image_url):
    try:
        # Use the correct JSON key "url" for the serverless function
        response = requests.post(SERVERLESS_URL, json={'url': image_url},
                                 headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            return response.json().get('prediction', 'No prediction found')
    except requests.exceptions.RequestException as e:
        return f'Error: {str(e)}'
    return 'Error in prediction'

if __name__ == '__main__':
    app.run(debug=True)
