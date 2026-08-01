import torch

class Config:
    CBIS_ROOT = "/home/tq_ahmad/Dataset/CBIS"
    INBREAST_ROOT = "/home/tq_ahmad/Dataset/INbreast Release 1.0"

    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LR = 1e-4
    GAMMA = 2.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_DIR = "checkpoints"
    SAVE_PATH = CHECKPOINT_DIR + "/bremsnet_best.pth"