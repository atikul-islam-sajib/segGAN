import os
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import config, validate_path, load
from .l1_loss import L1Loss
from .generator import Generator
from .discriminator import Discriminator


def load_dataloader():
    config_files = config()
    processed_path = config_files["path"]["processed_path"]
    processed_path = validate_path(path=processed_path)

    train_dataloader = load(
        filename=os.path.join(processed_path, "train_dataloader.pkl")
    )
    test_dataloader = load(filename=os.path.join(processed_path, "test_dataloader.pkl"))

    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}


def helpers(**kwargs):
    channels = kwargs["channels"]
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]

    netG = Generator(in_channels=channels)
    netD = Discriminator(in_channels=channels)

    if adam:
        optimizerG = optim.Adam(params=netG.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerD = optim.Adam(params=netD.parameters(), lr=lr, betas=(beta1, beta2))

    elif SGD:
        optimizerG = optim.SGD(params=netG.parameters(), lr=lr, momentum=momentum)
        optimizerD = optim.SGD(params=netD.parameters(), lr=lr, momentum=momentum)

    l1loss = L1Loss(reduction="mean")
    dataloader = load_dataloader()

    return {
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "l1loss": l1loss,
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
    }


if __name__ == "__main__":
    beta1 = config()["model"]["beta1"]
    beta2 = config()["model"]["beta2"]
    momentum = config()["model"]["momentum"]

    init = helpers(
        channels=3,
        lr=0.0002,
        adam=True,
        SGD=False,
        beta1=beta1,
        beta2=beta2,
        momentum=momentum,
    )

    print(init["netG"])
    print(init["netD"])
    print(init["optimizerG"])
    print(init["optimizerD"])
    print(init["l1loss"])
    print(init["train_dataloader"])
    print(init["test_dataloader"])
