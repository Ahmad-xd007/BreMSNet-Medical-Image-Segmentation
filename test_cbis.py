import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import sys
from contextlib import contextmanager

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import MammogramSegDataset
from model.bremsnet import BReMSNet
from validate import validate


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


def create_test_csv():
    df = pd.read_csv("cbis_mass_pairs.csv")
    test_df = df[df["split"] == "test"]

    test_df.to_csv("cbis_test_pairs.csv", index=False)

    print("Test samples:", len(test_df))


def test_cbis():
    cfg = Config()

    create_test_csv()

    test_dataset = MammogramSegDataset(
        "cbis_test_pairs.csv",
        cfg.IMAGE_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if cfg.DEVICE == "cuda" else False
    )

    model = BReMSNet().to(cfg.DEVICE)

    model.load_state_dict(
        torch.load(
            cfg.SAVE_PATH,
            map_location=cfg.DEVICE
        )
    )

    model.eval()

    print("Evaluating CBIS-DDSM test set...")

    validate(
        model,
        test_loader,
        cfg.DEVICE
    )


if __name__ == "__main__":
    with suppress_stderr():
        test_cbis()