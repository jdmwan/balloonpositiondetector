import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms

class BalloonBBoxDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        #output parameters
        bbox = torch.tensor([
            row['x'],
            row['y'],
            row['width'],
            row['height']
        ], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, bbox
