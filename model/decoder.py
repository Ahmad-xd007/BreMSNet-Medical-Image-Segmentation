import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec3 = DecoderBlock(2048, 1024, 512)
        self.dec2 = DecoderBlock(512, 512, 256)
        self.dec1 = DecoderBlock(256, 256, 128)

        self.deep1 = nn.Conv2d(512, 1, 1)
        self.deep2 = nn.Conv2d(256, 1, 1)
        self.deep3 = nn.Conv2d(128, 1, 1)

    def forward(self, x1, x2, x3, x4):
        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        deep_outputs = [
            self.deep1(d3),
            self.deep2(d2),
            self.deep3(d1)
        ]

        return d1, deep_outputs