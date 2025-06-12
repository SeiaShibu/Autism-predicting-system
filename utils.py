import os
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from torchvision import transforms

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_transform(size=224):
    """Return the standard image transform pipeline."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_model(model, weights_path, device='cpu'):
    """
    Instantiate `model`, load its state_dict from `weights_path`,
    move to `device`, set to eval mode, and return it.
    """
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def calculate_metrics(outputs, labels):
    """
    Compute accuracy, precision, recall, and F1 score for binary classification.
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    labs  = labels.cpu().numpy()
    return {
        'accuracy':  accuracy_score(labs, preds),
        'precision': precision_score(labs, preds, zero_division=0),
        'recall':    recall_score(labs, preds, zero_division=0),
        'f1_score':  f1_score(labs, preds, zero_division=0)
    }

def preprocess_and_predict(image_path, model, transform, device='cpu'):
    """
    Preprocess a single image file and return the model's predicted class (0 or 1).
    """
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
    return out.argmax(dim=1).item()

def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save a training checkpoint including model state, optimizer state, epoch, and loss.
    """
    ensure_dir(os.path.dirname(filepath))
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
