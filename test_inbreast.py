import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import MammogramSegDataset
from model.bremsnet import BReMSNet
from validate import validate

def test():
    cfg = Config()

    dataset = MammogramSegDataset("inbreast_pairs.csv", cfg.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = BReMSNet().to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.SAVE_PATH, map_location=cfg.DEVICE))

    validate(model, loader, cfg.DEVICE)

if __name__ == "__main__":
    test()