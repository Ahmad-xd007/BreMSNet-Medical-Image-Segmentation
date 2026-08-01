import torch
import numpy as np
from metrics import dice_score, sensitivity, specificity, hd95

def validate(model, dataloader, device):
    model.eval()

    dice_list = []
    sens_list = []
    spec_list = []
    hd95_list = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            _, preds, _ = model(images)

            dice_list.append(dice_score(preds, masks))
            sens_list.append(sensitivity(preds, masks))
            spec_list.append(specificity(preds, masks))
            hd95_list.append(hd95(preds, masks))

    mean_dice = np.mean(dice_list)
    mean_sens = np.mean(sens_list)
    mean_spec = np.mean(spec_list)
    mean_hd95 = np.mean(hd95_list)

    print(f"Dice: {mean_dice:.4f}")
    print(f"Sensitivity: {mean_sens:.4f}")
    print(f"Specificity: {mean_spec:.4f}")
    print(f"HD95: {mean_hd95:.4f}")

    return mean_dice