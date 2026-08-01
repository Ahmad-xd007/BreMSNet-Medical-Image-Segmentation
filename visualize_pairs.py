import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils_image import read_image, resize_pad
from config import Config


def visualize_pairs(
    csv_path="cbis_mass_pairs.csv",
    save_dir="visual_checks",
    num_samples=20,
    image_size=512
):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    print("Total samples:", len(df))

    saved = 0

    for i in range(len(df)):
        row = df.iloc[i]

        image_path = row["image"]
        mask_path = row["mask"]

        image = read_image(image_path)
        mask = read_image(mask_path)

        mask = (mask > 0.5).astype("float32")

        image = resize_pad(
            image,
            image_size,
            is_mask=False
        )

        mask = resize_pad(
            mask,
            image_size,
            is_mask=True
        )

        if mask.sum() == 0:
            continue

        image_u8 = (image * 255).astype("uint8")
        overlay_rgb = cv2.cvtColor(
            image_u8,
            cv2.COLOR_GRAY2RGB
        )

        mask_u8 = (mask * 255).astype("uint8")

        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay_rgb,
            contours,
            -1,
            (255, 0, 0),
            3
        )

        red_layer = np.zeros_like(overlay_rgb)
        red_layer[:, :, 0] = 255

        overlay_rgb = np.where(
            mask[:, :, None] > 0.5,
            (0.6 * overlay_rgb + 0.4 * red_layer).astype("uint8"),
            overlay_rgb
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"Mask | Area: {int(mask.sum())}")
        axes[1].axis("off")

        axes[2].imshow(overlay_rgb)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"sample_{saved + 1}.png"
        )

        plt.savefig(save_path, dpi=150)
        plt.close()

        print("Saved:", save_path)

        saved += 1

        if saved >= num_samples:
            break

    print("Saved total:", saved)


if __name__ == "__main__":
    cfg = Config()

    visualize_pairs(
        csv_path="cbis_mass_pairs.csv",
        save_dir="visual_checks",
        num_samples=20,
        image_size=cfg.IMAGE_SIZE
    )