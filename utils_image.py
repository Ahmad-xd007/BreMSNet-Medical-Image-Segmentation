import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import pydicom


def read_image(path, is_mask=False):
    path = str(path)

    if path.lower().endswith(".dcm") or "." not in os.path.basename(path):
        dcm = pydicom.dcmread(path)

        try:
            img = dcm.pixel_array.astype(np.float32)
        except Exception:
            img = np.zeros(
                (int(dcm.Rows), int(dcm.Columns)),
                dtype=np.float32
            )
    else:
        img = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise ValueError(f"Cannot read image: {path}")

        img = img.astype(np.float32)

    if is_mask:
        img = (img > 0).astype(np.float32)
    else:
        img = enhance_mammogram(img)

    return img.astype(np.float32)


def enhance_mammogram(img):
    img = img.astype(np.float32)

    low, high = np.percentile(
        img,
        (1, 99)
    )

    img = np.clip(
        img,
        low,
        high
    )

    img = img - img.min()

    if img.max() > 0:
        img = img / img.max()

    img_u8 = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img_u8 = clahe.apply(img_u8)

    img = img_u8.astype(np.float32) / 255.0

    return img


def resize_pad(img, size=512, is_mask=False):
    h, w = img.shape[:2]

    scale = size / max(h, w)

    nh = int(h * scale)
    nw = int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    img = cv2.resize(
        img,
        (nw, nh),
        interpolation=interpolation
    )

    canvas = np.zeros(
        (size, size),
        dtype=np.float32
    )

    y = (size - nh) // 2
    x = (size - nw) // 2

    canvas[y:y + nh, x:x + nw] = img

    if is_mask:
        canvas = (canvas > 0.5).astype(np.float32)

    return canvas