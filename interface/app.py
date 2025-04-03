from flask import Flask, request, render_template, redirect, url_for
import requests

app = Flask(__name__)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_url = None

    if request.method == 'POST':
        image_url = request.form['image_url']
        prediction = get_prediction(image_url)

    return render_template('index.html', image_url=image_url, prediction=prediction)

# Function to call the model API
def get_prediction(image_url):
    response = requests.post('https://cifar100-api-4pxvf6xeea-uc.a.run.app/predict', data={'image_url': image_url})
    if response.status_code == 200:
        return response.json().get('prediction', 'No prediction found')
    return 'Error in prediction'

if __name__ == '__main__':
    app.run(debug=True)
