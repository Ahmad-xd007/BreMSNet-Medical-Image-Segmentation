import torch
import torch.nn as nn

class BFAModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

        self.conv = nn.Conv2d(channels * 2, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b1 = x - self.pool3(x)
        b2 = x - self.pool5(x)

        boundary = torch.cat([b1, b2], dim=1)

        weights = self.sigmoid(self.conv(boundary))

        return x * weights