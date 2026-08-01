import torch
import torch.nn as nn

class MHDModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        split = in_channels // 4

        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(split, split, 1)
            for _ in range(4)
        ])

        self.branch1 = nn.Conv2d(split, split, 3, padding=1)
        self.branch2 = nn.Conv2d(split, split, 3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(split, split, 3, padding=3, dilation=3)

        self.fusion = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=1)

        x1 = self.conv1x1[0](chunks[0])
        x2 = self.conv1x1[1](chunks[1])
        x3 = self.conv1x1[2](chunks[2])
        x4 = self.conv1x1[3](chunks[3])

        y1 = self.branch1(x1)
        y2 = self.branch2(x2 + y1)
        y3 = self.branch3(x3 + y2)

        out = torch.cat([y1, y2, y3, x4], dim=1)

        return self.fusion(out)