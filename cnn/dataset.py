import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, split="Training"):
        df = pd.read_csv(csv_file)
        df = df[df["Usage"] == split]

        self.labels = df["emotion"].values
        self.images = df["pixels"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = np.array(self.images[idx].split(), dtype=np.float32)
        img = img.reshape(1, 48, 48) / 255.0  # (1,48,48)

        label = self.labels[idx]

        return torch.tensor(img), torch.tensor(label)