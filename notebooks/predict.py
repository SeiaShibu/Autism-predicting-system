# predict.py
import os
import torch
from torchvision import transforms, models
from PIL import Image

# ---------------------------
# Paths
# ---------------------------
model_path = "model/best_model.pth"   # path to your trained model
image_folder = "data/test_images"     # folder with new images to predict
os.makedirs(image_folder, exist_ok=True)  # just in case folder is missing

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Transform (same as training)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Load Model
# ---------------------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ---------------------------
# Predict function
# ---------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    label = "Non-ASD" if pred.item() == 0 else "ASD"
    return label

# ---------------------------
# Predict all images in folder
# ---------------------------
for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, img_file)
        result = predict_image(img_path)
        print(f"{img_file} --> {result}")
