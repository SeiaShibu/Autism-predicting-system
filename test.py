# test.py
import torch
from torchvision import transforms
from PIL import Image
from train import CustomResNet
from torchvision import models

# Load model
base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = CustomResNet(base_model)
model.load_state_dict(torch.load('asd_classifier_cnn.pth'))
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_names = ['Non-ASD', 'ASD']
    print(f"Prediction: {class_names[predicted.item()]}")

# Example usage
predict_image('data/Images/TSImages/TS001_11.png')