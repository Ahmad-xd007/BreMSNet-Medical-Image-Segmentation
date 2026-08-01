import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import sys
import random
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import MammogramSegDataset
from model.bremsnet import BReMSNet
from losses import AWLoss
from metrics import dice_score


@contextmanager
def suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)

    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = (targets > 0.5).float()

        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)

        dice = (2 * intersection + self.eps) / (union + self.eps)

        return 1 - dice


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_train_val_csv():
    df = pd.read_csv("cbis_mass_pairs.csv")

    train_df = df[df["split"] == "train"].sample(
        frac=1,
        random_state=42
    )

    val_size = int(0.15 * len(train_df))

    val_df = train_df.iloc[:val_size]
    train_df = train_df.iloc[val_size:]

    train_df.to_csv("cbis_train_pairs.csv", index=False)
    val_df.to_csv("cbis_val_pairs.csv", index=False)

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))


def evaluate(model, loader, device):
    model.eval()
    dice_values = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            _, preds, _ = model(images)

            dice_values.append(
                dice_score(
                    preds,
                    masks,
                    threshold=0.05
                )
            )

    return sum(dice_values) / max(len(dice_values), 1)


def combined_loss(logits, masks, dice_loss, bce_loss, aw_loss):
    return (
        0.55 * dice_loss(logits, masks)
        + 0.35 * bce_loss(logits, masks)
        + 0.10 * aw_loss(logits, masks)
    )


def train():
    set_seed(42)

    cfg = Config()
    NUM_EPOCHS = 50

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    create_train_val_csv()

    train_dataset = MammogramSegDataset(
        "cbis_train_pairs.csv",
        cfg.IMAGE_SIZE,
        augment=False
    )

    val_dataset = MammogramSegDataset(
        "cbis_val_pairs.csv",
        cfg.IMAGE_SIZE,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if cfg.DEVICE == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if cfg.DEVICE == "cuda" else False
    )

    model = BReMSNet().to(cfg.DEVICE)

    dice_loss = DiceLoss()
    aw_loss = AWLoss(gamma=cfg.GAMMA)

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([30.0]).to(cfg.DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.98
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.DEVICE == "cuda")
    )

    best_dice = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images = images.to(cfg.DEVICE, non_blocking=True)
            masks = masks.to(cfg.DEVICE, non_blocking=True)
            masks = (masks > 0.5).float()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=(cfg.DEVICE == "cuda")
            ):
                coarse, final, deep_outputs = model(images)

                loss_final = combined_loss(
                    final,
                    masks,
                    dice_loss,
                    bce_loss,
                    aw_loss
                )

                loss_coarse = combined_loss(
                    coarse,
                    masks,
                    dice_loss,
                    bce_loss,
                    aw_loss
                )

                loss_deep = 0.0

                for out in deep_outputs:
                    loss_deep += combined_loss(
                        out,
                        masks,
                        dice_loss,
                        bce_loss,
                        aw_loss
                    )

                loss = (
                    loss_final
                    + 0.3 * loss_coarse
                    + 0.1 * loss_deep
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        val_dice = evaluate(
            model,
            val_loader,
            cfg.DEVICE
        )

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Val Dice: {val_dice:.4f} "
            f"LR: {scheduler.get_last_lr()[0]:.8f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            torch.save(
                model.state_dict(),
                cfg.SAVE_PATH
            )

            print("Saved best model")


if __name__ == "__main__":
    with suppress_stderr():
        train()