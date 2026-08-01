import os
import cv2
import sys
import numpy as np
import pandas as pd
from contextlib import contextmanager


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


def clean_path(p):
    return str(p).strip().replace("\\", "/")


def extract_uid_folder(path):
    parts = clean_path(path).split("/")
    uids = [p for p in parts if p.startswith("1.3.6")]
    return uids[-1] if len(uids) > 0 else None


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.uint8)


def image_area(path):
    img = read_gray(path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    return h * w


def white_ratio(path):
    img = read_gray(path)
    if img is None:
        return 0.0
    return float((img > 10).sum() / img.size)


def build_jpeg_index(jpeg_root):
    index = {}

    for root, _, files in os.walk(jpeg_root):
        jpgs = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(jpgs) == 0:
            continue

        uid = os.path.basename(root)

        if not uid.startswith("1.3.6"):
            continue

        index[uid] = []

        for f in jpgs:
            index[uid].append(os.path.join(root, f))

    return index


def resolve_jpeg(csv_path, jpeg_index, is_mask=False):
    uid = extract_uid_folder(csv_path)

    if uid is None or uid not in jpeg_index:
        return None

    files = jpeg_index[uid]

    if len(files) == 0:
        return None

    # ---------------- MASK SELECTION ----------------
    if is_mask:

        # First preference: file starts with 2-
        for f in files:
            name = os.path.basename(f).lower()

            if name.startswith("2-"):
                return f

        # Fallback: choose file with reasonable white mask area
        valid = []

        for f in files:
            ratio = white_ratio(f)

            if 0.00001 < ratio < 0.30:
                valid.append((ratio, f))

        if len(valid) == 0:
            return None

        valid = sorted(valid, key=lambda x: x[0])

        return valid[0][1]

    # ---------------- IMAGE SELECTION ----------------
    else:
        candidates = []

        for f in files:
            name = os.path.basename(f).lower()

            if name.startswith("2-"):
                continue

            candidates.append(f)

        if len(candidates) == 0:
            candidates = files

        candidates = sorted(
            candidates,
            key=lambda x: image_area(x),
            reverse=True
        )

        return candidates[0]


def clean_mask(mask):
    binary = (mask > 5).astype(np.uint8)

    h, w = binary.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < 5:
            continue

        # Remove wide bottom bar artifact
        if y + bh > int(0.90 * h) and bw > int(0.25 * w):
            continue

        # Remove huge wrong masks
        if area / (h * w) > 0.40:
            continue

        components.append((area, i))

    if len(components) == 0:
        return None

    components = sorted(components, reverse=True)
    largest_label = components[0][1]

    clean = (labels == largest_label).astype(np.uint8) * 255

    return clean


def is_good_image(img):
    if img is None:
        return False, "unreadable_image"

    if img.mean() < 1:
        return False, "black_image"

    if img.std() < 1:
        return False, "low_contrast_image"

    return True, "ok"


def save_debug(image, mask, clean, save_path):
    image_vis = image.copy()
    mask_vis = mask.copy()
    clean_vis = clean.copy()

    image_vis = cv2.resize(image_vis, (512, 512))
    mask_vis = cv2.resize(mask_vis, (512, 512))
    clean_vis = cv2.resize(clean_vis, (512, 512))

    combined = np.concatenate([image_vis, mask_vis, clean_vis], axis=1)
    cv2.imwrite(save_path, combined)


def build_cbis_mass_pairs(cbis_root):
    csv_dir = os.path.join(cbis_root, "csv")
    jpeg_root = os.path.join(cbis_root, "jpeg")

    clean_mask_dir = "/home/tq_ahmad/Dataset/CBIS_clean_masks"
    debug_dir = "/home/tq_ahmad/Dataset/CBIS_debug_masks"

    os.makedirs(clean_mask_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    train_csv = os.path.join(csv_dir, "mass_case_description_train_set.csv")
    test_csv = os.path.join(csv_dir, "mass_case_description_test_set.csv")

    jpeg_index = build_jpeg_index(jpeg_root)

    print("JPEG UID folders:", len(jpeg_index))

    pairs = []
    missing = 0
    removed = 0
    reasons = {}
    debug_saved = 0

    for csv_file, split in [
        (train_csv, "train"),
        (test_csv, "test")
    ]:
        df = pd.read_csv(csv_file)

        for idx, row in df.iterrows():

            image_path = resolve_jpeg(
                row["cropped image file path"],
                jpeg_index,
                is_mask=False
            )

            mask_path = resolve_jpeg(
                row["ROI mask file path"],
                jpeg_index,
                is_mask=True
            )

            if image_path is None or mask_path is None:
                missing += 1
                continue

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                missing += 1
                continue

            if image_path == mask_path:
                removed += 1
                reasons["same_image_mask"] = reasons.get("same_image_mask", 0) + 1
                continue

            image = read_gray(image_path)
            mask = read_gray(mask_path)

            image_ok, image_reason = is_good_image(image)

            if not image_ok:
                removed += 1
                reasons[image_reason] = reasons.get(image_reason, 0) + 1
                continue

            clean = clean_mask(mask)

            if clean is None:
                removed += 1
                reasons["empty_or_bad_mask"] = reasons.get("empty_or_bad_mask", 0) + 1
                continue

            mask_ratio = (clean > 0).sum() / clean.size

            if mask_ratio < 0.00001:
                removed += 1
                reasons["tiny_mask"] = reasons.get("tiny_mask", 0) + 1
                continue

            if mask_ratio > 0.30:
                removed += 1
                reasons["huge_mask"] = reasons.get("huge_mask", 0) + 1
                continue

            patient_id = row.get("patient_id", f"{split}_{idx}")

            clean_mask_path = os.path.join(
                clean_mask_dir,
                f"{split}_{patient_id}_{idx}.png"
            )

            cv2.imwrite(clean_mask_path, clean)

            if debug_saved < 50:
                debug_path = os.path.join(
                    debug_dir,
                    f"{split}_{idx}_debug.png"
                )
                save_debug(image, mask, clean, debug_path)
                debug_saved += 1

            pairs.append({
                "patient_id": patient_id,
                "image": image_path,
                "mask": clean_mask_path,
                "split": split,
                "dataset": "CBIS-DDSM",
                "image_uid": extract_uid_folder(row["cropped image file path"]),
                "mask_uid": extract_uid_folder(row["ROI mask file path"])
            })

    print("Final clean pairs:", len(pairs))
    print("Missing pairs:", missing)
    print("Removed pairs:", removed)

    print("Removal reasons:")
    for k, v in reasons.items():
        print(f"  {k}: {v}")

    print("Debug masks saved:", debug_saved)

    return pairs


if __name__ == "__main__":
    with suppress_stderr():
        pairs = build_cbis_mass_pairs("/home/tq_ahmad/Dataset/CBIS")

    pd.DataFrame(pairs).to_csv(
        "cbis_mass_pairs.csv",
        index=False
    )

    print("Saved:", len(pairs))