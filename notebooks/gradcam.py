# gradcam.py
import torch
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ---------------------------
# Paths
# ---------------------------
model_path = "model/asd_classifier_cnn.pth"  # Your trained model path
image_path = "data/Images/TSImages/sample_image.png"  # Replace with an actual image in your dataset

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Transform
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
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ---------------------------
# Grad-CAM Class
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        outputs = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()
        loss = outputs[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(1,2))
        cam = torch.sum(weights[:, None, None] * self.activations, dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = np.nan_to_num(cam)      # Remove NaNs
        cam = np.squeeze(cam)         # Ensure 2D
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)  # Normalize 0-1
        cam = cv2.resize(cam, (224, 224))
        return cam

# ---------------------------
# Run Grad-CAM
# ---------------------------
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

grad_cam = GradCAM(model, model.layer4[-1])
cam = grad_cam(input_tensor)

# Overlay CAM on original image
img_np = np.array(img.resize((224, 224)))
cam_uint8 = np.uint8(255 * cam)
heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# Display
plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.axis('off')
plt.show()
