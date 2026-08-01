import os
import re
import cv2
import plistlib
import numpy as np
import pydicom

def parse_point(point_string):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", point_string)
    return int(float(nums[0])), int(float(nums[1]))

def dicom_shape(dicom_path):
    dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    return int(dcm.Rows), int(dcm.Columns)

def create_mask_from_xml(xml_path, image_shape, target_roi="Mass"):
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(xml_path, "rb") as f:
        data = plistlib.load(f)

    images = data.get("Images", [])

    for img_data in images:
        rois = img_data.get("ROIs", [])

        for roi in rois:
            name = roi.get("Name", "")

            if name.lower() != target_roi.lower():
                continue

            points = roi.get("Point_px", [])

            if len(points) < 3:
                continue

            polygon = np.array([parse_point(p) for p in points], dtype=np.int32)
            polygon = polygon.reshape((-1, 1, 2))

            cv2.fillPoly(mask, [polygon], 255)

    return mask

def find_dicom(dicom_dir, file_id):
    for file in os.listdir(dicom_dir):
        if str(file_id) in file:
            return os.path.join(dicom_dir, file)
    return None

def generate_inbreast_masks(inbreast_root):
    dicom_dir = os.path.join(inbreast_root, "AllDICOMs")
    xml_dir = os.path.join(inbreast_root, "AllXML")
    mask_dir = os.path.join(inbreast_root, "generated_masks")

    os.makedirs(mask_dir, exist_ok=True)

    pairs = []

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        file_id = xml_file.replace(".xml", "")
        xml_path = os.path.join(xml_dir, xml_file)
        dicom_path = find_dicom(dicom_dir, file_id)

        if dicom_path is None:
            continue

        shape = dicom_shape(dicom_path)
        mask = create_mask_from_xml(xml_path, shape, target_roi="Mass")

        mask_path = os.path.join(mask_dir, file_id + "_mask.png")
        cv2.imwrite(mask_path, mask)

        pairs.append({
            "image": dicom_path,
            "mask": mask_path,
            "dataset": "INbreast"
        })

    return pairs

if __name__ == "__main__":
    pairs = generate_inbreast_masks("data/INbreast Release 1.0")

    import pandas as pd
    pd.DataFrame(pairs).to_csv("inbreast_pairs.csv", index=False)

    print("Saved:", len(pairs))