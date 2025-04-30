import torch
import torch.nn as nn

class BalloonNetCNN(nn.Module):
    def __init__(self):
        super(BalloonNetCNN, self).__init__()
        self.height = 256
        self.width = 256

        # Input: 3 x 32 x 32 (RGB image)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 3 → 16 channels, 32x32
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),                           # 16 x 16 x 16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 → 32 channels, 16x16
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),                           # 32 x 8 x 8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 8 x 8
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)                            # 64 x 4 x 4
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                  # 64 x 4 x 4 = 1024
            nn.Linear(64 * self.height * self.width, 128),
            nn.ReLU(),
            nn.Linear(128, 4)                              # Output: 2 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
