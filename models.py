import torch 
from torch import nn

class DigitModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))

        )

        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, 10)
        )

    def forward(self, images):
        x = self.cnn_block(images)
        logits = self.linear_block(x)

        return logits
