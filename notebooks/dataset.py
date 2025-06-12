# data.py
import os
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch

class EyeTrackingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, folder in enumerate(['TCImages', 'TSImages']):
            full_path = os.path.join(root_dir, folder)
            if os.path.exists(full_path):
                for img_file in os.listdir(full_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(full_path, img_file))
                        self.labels.append(label)
            else: 
                print(f"Warning: Folder '{folder}' not found in {root_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    
    def stratified_split(dataset, test_size=0.2):
        labels = np.array(dataset.labels)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels):
            return train_idx, val_idx