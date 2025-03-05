from flask import Flask, request, jsonify
import torch
import torch.nn as nn

# Define your model class
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3072, 100)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = SimpleModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    # Assume input is a JSON with a key 'data' containing a list of numbers
    data = request.json['data']
    input_tensor = torch.tensor(data, dtype=torch.float32)

    # Run the model
    with torch.no_grad():
        output = model(input_tensor)

    # Return the output as JSON
    return jsonify({'prediction': output.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
