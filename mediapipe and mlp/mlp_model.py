import torch
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1434, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
