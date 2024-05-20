import argparse
import torch
from torch.optim.lr_scheduler import StepLR

from .helper import helpers
from .utils import weight_init, device_init
from .generator import Generator
from .discriminator import Discriminator


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

    def l1_loss(self, model):
        if isinstance(model, Generator):
            return (torch.norm(params, 1) for params in model.parameters()).mean()
        else:
            raise ValueError("The model is not a Generator".capitalize())

    def l2_loss(self, model):
        if isinstance(model, Generator):
            return (torch.norm(params, 2) for params in model.parameters()).mean()
        else:
            raise ValueError("The model is not a Generator".capitalize())

    def elastic_loss(self, model):
        if isinstance(model, Generator):
            l1 = self.l1_loss(model=model)
            l2 = self.l2_loss(model=model)

            return l1 + l2

        else:
            raise ValueError("The model is not a Generator".capitalize())

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
    parser = argparse.ArgumentParser(description="Traine code for seGAN".title())
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Define the number of channels".capitalize(),
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Define the learning rate".capitalize()
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Define the number of epochs".capitalize(),
    )
    parser.add_argument(
        "--adam", type=bool, default=True, help="Define the optimizer".capitalize()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Define the optimizer".capitalize()
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Define the device".capitalize()
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Define the beta1".capitalize()
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Define the beta2".capitalize()
    )
    parser.add_argument(
        "--momentum", type=float, default=0.90, help="Define the momentum".capitalize()
    )
    parser.add_argument(
        "--smooth", type=float, default=0.001, help="Define the smooth".capitalize()
    )
    parser.add_argument(
        "--step_size", type=int, default=10, help="Define the step size".capitalize()
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Define the gamma".capitalize()
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=False,
        help="Define the lr scheduler".capitalize(),
    )
    parser.add_argument(
        "--is_display", type=bool, default=True, help="Define the display".capitalize()
    )
    parser.add_argument(
        "--is_weight_init",
        type=bool,
        default=True,
        help="Define the weight init".capitalize(),
    )
    args = parser.parse_args()

    trainer = Trainer(
        channels=args.channels,
        lr=args.lr,
        epochs=args.epochs,
        adam=args.adam,
        SGD=args.SGD,
        device=args.device,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        smooth=args.smooth,
        step_size=args.step_size,
        gamma=args.gamma,
        lr_scheduler=args.lr_scheduler,
        is_display=args.is_display,
        is_weight_init=args.is_weight_init,
    )
