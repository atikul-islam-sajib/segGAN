import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

from .helper import helpers
from .utils import weight_init, device_init, config, validate_path, dump, load
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
        clamp=0.01,
        l1loss_value=0.1,
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
        self.clamp = clamp
        self.l1loss_value = l1loss_value
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

        self.loss = float("inf")
        self.total_netG_loss = []
        self.total_netD_loss = []
        self.history = {"netG_loss": [], "netD_loss": []}

        self.train_images_path = validate_path(
            path=config()["path"]["train_images_path"]
        )
        self.train_model = validate_path(path=config()["path"]["train_model_path"])
        self.best_model = validate_path(path=config()["path"]["best_model_path"])
        self.metrics_path = validate_path(path=config()["path"]["metrics_path"])

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
        torch.save(
            self.netG.state_dict(),
            os.path.join(self.train_model, "netG{}.pth".format(kwargs["epoch"])),
        )

        if self.loss > kwargs["netG_loss"]:
            self.loss = kwargs["netG_loss"]
            torch.save(
                {
                    "netG": self.netG.state_dict(),
                    "netD": self.netD.state_dict(),
                    "loss": kwargs["netG_loss"],
                },
                os.path.join(self.best_model, "best_model.pth"),
            )

    def saved_training_images(self, **kwargs):
        save_image(
            kwargs["predicted_mask"],
            os.path.join(
                self.train_images_path, "image{}.png".format(kwargs["epoch"] + 1)
            ),
            nrow=32,
            normalize=True,
        )

    def show_progress(self, **kwargs):
        if self.is_display:
            print(
                "Epochs - [{}/{}] - netG_loss: {:.4f} - netD_loss: {:.4f}".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    kwargs["netG_loss"],
                    kwargs["netD_loss"],
                )
            )

        else:
            print(
                "Epochs - [{}/{}] is completed".format(
                    kwargs["epoch"], kwargs["epochs"]
                )
            )

    def update_train_netG(self, **kwargs):
        self.optimizerG.zero_grad()

        images = kwargs["image"]
        masks = kwargs["mask"]

        fake_masks = self.netG(images)
        fake_masks = torch.sigmoid(fake_masks)

        fakeB = images * fake_masks
        realA = images * masks

        fake_masks_loss = self.diceloss(fake_masks, masks)

        real_predict = self.netD(realA)
        fake_predict = self.netD(fakeB.detach())

        multiscale_loss = self.l1loss(real_predict, fake_predict)

        lossG = self.l1loss_value * fake_masks_loss + multiscale_loss

        lossG.backward()
        self.optimizerG.step()

        return lossG.item()

    def update_train_netD(self, **kwargs):
        self.optimizerD.zero_grad()

        images = kwargs["image"]
        masks = kwargs["mask"]

        fake_masks = self.netG(images)
        fake_masks = torch.sigmoid(fake_masks)

        realA = images * masks
        fakeB = images * fake_masks

        real_predict = self.netD(realA)
        fake_predict = self.netD(fakeB.detach())

        lossD = 1 - self.l1loss(real_predict, fake_predict)

        lossD.backward()
        self.optimizerD.step()

        for params in self.netD.parameters():
            params.data.clamp_(-self.clamp, self.clamp)

        return lossD.item()

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            netG_loss = []
            netD_loss = []

            for index, (image, mask) in enumerate(self.train_dataloader):
                try:
                    image = image.to(self.device)
                    mask = mask.to(self.device)

                    netD_loss.append(self.update_train_netD(image=image, mask=mask))
                    netG_loss.append(self.update_train_netG(image=image, mask=mask))

                except Exception as e:
                    print(
                        f"Error during training at epoch {epoch + 1}, batch {index}: {e}"
                    )
                    continue

            try:
                self.show_progress(
                    epoch=epoch + 1,
                    epochs=self.epochs,
                    netD_loss=np.mean(netD_loss),
                    netG_loss=np.mean(netG_loss),
                )

                image, mask = next(iter(self.test_dataloader))
                image = image.to(self.device)
                predicted_mask = self.netG(image)

                self.saved_training_images(
                    image=image, mask=mask, predicted_mask=predicted_mask, epoch=epoch
                )

                self.saved_model_checkpoints(
                    netG_loss=np.mean(netG_loss),
                    epoch=epoch + 1,
                )

                if self.lr_scheduler:
                    self.schedulerD.step()
                    self.schedulerG.step()

                self.total_netG_loss.append(np.mean(netG_loss))
                self.total_netD_loss.append(np.mean(netD_loss))

            except Exception as e:
                print(f"Error during post-epoch processing at epoch {epoch + 1}: {e}")
                continue

        try:
            self.history["netG_loss"].append(self.total_netG_loss)
            self.history["netD_loss"].append(self.total_netD_loss)

            pd.DataFrame(
                {
                    "netG_loss": self.total_netG_loss,
                    "netD_loss": self.total_netD_loss,
                }
            ).to_csv(os.path.join(self.metrics_path, "model_history.csv"))

            for filename, value in [
                ("history.pkl", self.history),
                ("netG_loss.pkl", self.total_netG_loss),
                ("netD_loss.pkl", self.total_netD_loss),
            ]:
                dump(value=value, filename=os.path.join(self.metrics_path, filename))

            print(
                "Saved the model history in a csv file in {}\nSaved the model history in pickle format in {}".format(
                    self.metrics_path, self.metrics_path
                )
            )

        except Exception as e:
            print(f"Error during post-training processing: {e}")
            return

    @staticmethod
    def plot_history():
        metrics_path = config()["path"]["metrics_path"]
        metrics_path = validate_path(path=metrics_path)

        plt.figure(figsize=(20, 20))

        history = load(filename=os.path.join(metrics_path, "history.pkl"))

        for index, (title, loss) in enumerate(
            [
                ("netG_loss", history["netG_loss"]),
                ("netD_loss", history["netD_loss"]),
            ]
        ):
            plt.subplot(2 * 1, 2 * 2, 2 * index + (index + 1))

            plt.plot(loss[0], label=title)
            plt.title(f"{title}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()

        plt.tight_layout()
        save_image(os.path.join(metrics_path, "model_history.jpeg"))
        plt.show()


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

    trainer.train()

    trainer.plot_history()
