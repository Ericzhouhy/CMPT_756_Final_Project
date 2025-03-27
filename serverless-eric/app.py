import torch
import torchvision.models as models

# Define the model architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 100)

# Load the model checkpoint, mapping it to the CPU
checkpoint = torch.load('checkpoint_epoch_100.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

# Save the model in the required format
torch.save(model.state_dict(), 'deployed_model.pth')
