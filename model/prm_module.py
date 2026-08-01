import torch
import torch.nn as nn

class PRMModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.coarse = nn.Conv2d(channels, 1, 1)

        self.attention = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x):
        coarse = self.coarse(x)

        attention = self.attention(torch.sigmoid(coarse))

        refined = x * attention

        final = self.refine(refined)

        return coarse, final