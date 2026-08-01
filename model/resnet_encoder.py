import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        self.initial = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),

            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):

        x0 = self.initial(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4