import torch
import torch.nn as nn

class AWLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        pt = probs * targets + (1 - probs) * (1 - targets)

        focal_loss = -((1 - pt) ** self.gamma) * torch.log(pt + self.eps)

        weights = focal_loss / (torch.sum(focal_loss) + self.eps)

        loss = torch.sum(weights * focal_loss)

        return loss