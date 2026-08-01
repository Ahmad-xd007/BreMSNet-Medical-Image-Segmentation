import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > 0.5).float()

    pred_sum = torch.sum(pred)
    target_sum = torch.sum(target)

    if target_sum == 0:
        return 0.0

    if pred_sum == 0:
        return 0.0

    intersection = torch.sum(pred * target)
    union = pred_sum + target_sum

    return ((2 * intersection + eps) / (union + eps)).item()


def sensitivity(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > 0.5).float()

    tp = torch.sum(pred * target)
    fn = torch.sum((1 - pred) * target)

    if torch.sum(target) == 0:
        return 0.0

    return (tp / (tp + fn + eps)).item()


def specificity(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > 0.5).float()

    tn = torch.sum((1 - pred) * (1 - target))
    fp = torch.sum(pred * (1 - target))

    return (tn / (tn + fp + eps)).item()


def hd95(pred, target, threshold=0.5):
    pred = (
        torch.sigmoid(pred) > threshold
    ).detach().cpu().numpy().astype(np.uint8)

    target = (
        target > 0.5
    ).detach().cpu().numpy().astype(np.uint8)

    pred = pred[0, 0]
    target = target[0, 0]

    if pred.sum() == 0 or target.sum() == 0:
        return 512.0

    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)

    pred_points = np.argwhere(pred_border > 0)
    target_points = np.argwhere(target_border > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return 512.0

    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)

    d1, _ = target_tree.query(pred_points, k=1)
    d2, _ = pred_tree.query(target_points, k=1)

    hd95_value = max(
        np.percentile(d1, 95),
        np.percentile(d2, 95)
    )

    return float(hd95_value)