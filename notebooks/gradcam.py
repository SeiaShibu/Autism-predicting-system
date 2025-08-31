import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

from train import CustomResNet   # make sure train.py is in the same folder or adjust path

# -------------------------
# GradCAM Class
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.size(2), input_tensor.size(3)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class

# -------------------------
# Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = torch.hub.load('pytorch/vision', 'resnet18', weights=None)
model = CustomResNet(base_model)
model.load_state_dict(torch.load("asd_classifier_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Pick last conv block
target_layer = model.base_model[-2]
grad_cam = GradCAM(model, target_layer)

# -------------------------
# Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# Function: Run Grad-CAM
# -------------------------
def run_gradcam(image_path, output_dir="gradcam_results"):
    os.makedirs(output_dir, exist_ok=True)

    # Load & preprocess
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Grad-CAM
    cam, pred_class = grad_cam.generate_cam(input_tensor)

    # Prediction label
    label_map = {0: "Non-ASD", 1: "ASD"}
    prediction = label_map[pred_class]

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img.resize((224, 224)))[:, :, ::-1]
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Save result
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, overlay)

    # Textual reasoning
    explanation = f"""
    🔎 Prediction: {prediction}

    Reasoning:
    - The highlighted regions (red/yellow) show the image areas most responsible 
      for this classification.
    - If ASD: the model focused on gaze fixation patterns or atypical visual features.
    - If Non-ASD: the model detected more typical gaze distribution/visual cues.
    """

    print(explanation)
    print(f"✅ Saved Grad-CAM result to: {out_path}")

# -------------------------
# Run on a sample image
# -------------------------
test_image = "data/Images/TSImages/TS001_11.png"   # change this to your test folder
run_gradcam(test_image)

