import argparse
import torch
from torch.optim.lr_scheduler import StepLR

from .helper import helpers
from .utils import weight_init, device_init


class Trainer:
    def __init__(
        self,
        channels=3,
        lr=0.0002,
        epochs=100,
        adam=True,
        SGD=False,
        device="cuda",
        beta1=0.5,
        beta2=0.999,
        momentum=0.90,
        smooth=0.001,
        step_size=10,
        gamma=0.5,
        lr_scheduler=False,
        is_display=True,
        is_weight_init=True,
    ):

        self.channels = channels
        self.lr = lr
        self.epochs = epochs
        self.adam = adam
        self.SGD = SGD
        self.device = device
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.smooth = smooth
        self.step_size = step_size
        self.gamma = gamma
        self.lr_scheduler = lr_scheduler
        self.is_display = is_display
        self.is_weight_init = is_weight_init

        self.device = device_init(device=self.device)

        self.init = helpers(
            channels=self.channels,
            lr=self.lr,
            adam=self.adam,
            SGD=self.SGD,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            smooth=self.smooth,
        )

        self.netG = self.init["netG"]
        self.netD = self.init["netD"]

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD = self.init["optimizerD"]

        self.l1loss = self.init["l1loss"]
        self.diceloss = self.init["diceloss"]

        self.train_dataloader = self.init["train_dataloader"]
        self.test_dataloader = self.init["test_dataloader"]

        self.netG.to(self.device)
        self.netD.to(self.device)

        if self.is_weight_init:
            self.netG.apply(weight_init)
            self.netD.apply(weight_init)

        if self.lr_scheduler:
            self.schedulerG = StepLR(
                optimizer=self.optimizerG, step_size=self.step_size, gamma=self.gamma
            )
            self.schedulerD = StepLR(
                optimizer=self.optimizerD, step_size=self.step_size, gamma=self.gamma
            )

    def l2_loss(self, model):
        pass

    def elastic_loss(self, model):
        pass

    def saved_model_checkpoints(self, **kwargs):
        pass

    def saved_training_images(self, **kwargs):
        pass

    def show_progress(self, **kwargs):
        pass

    def update_train_netG(self, **kwargs):
        pass

    def update_train_netD(self, **kwargs):
        pass

    def train(self):
        pass


if __name__ == "__main__":
    trainer = Trainer()
