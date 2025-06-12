from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# Load the trained CNN model (ResNet18)
model = models.resnet18(pretrained=True)  # Load pre-trained weights
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Modify the final layer for binary classification (ASD vs Non-ASD)
model.load_state_dict(torch.load('model/asd_classifier_cnn.pth'))  # Load your trained model weights
model.eval()  # Set the model to evaluation mode (not training)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 (ResNet input size)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ResNet's pre-trained values
])

@app.route('/')
def index():
    return render_template('index.html')  
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:  # Check if the file part is in the request
        return jsonify({'error': 'No file part'})

    file = request.files['file']  # Get the uploaded file
    
    if file.filename == '':  # If no file is selected
        return jsonify({'error': 'No selected file'})

    # Open the uploaded image
    img = Image.open(file).convert('RGB')  # Ensure the image is in RGB format

    # Apply the transformation to the image (resize, tensor conversion, normalization)
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

    # Perform prediction using the model
    with torch.no_grad():  # We don't need gradients for prediction
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the index of the class with the highest score

    # Map the prediction to the appropriate label (ASD or Non-ASD)
    label = 'ASD' if predicted.item() == 1 else 'Non-ASD'

    # Return the prediction result as JSON
    return jsonify({'prediction': label})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Run in debug mode for easier development
