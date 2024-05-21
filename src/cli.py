import argparse
import os

from .utils import config, validate_path
from .dataloader import Loader
from .trainer import Trainer
from .test import TestModel


def execute(kwargs):
    if kwargs["train"]:
        loader = Loader(
            image_path=kwargs["image_path"],
            batch_size=kwargs["batch_size"],
            channels=kwargs["channels"],
            image_size=kwargs["image_size"],
            split_size=kwargs["split_size"],
        )

        loader.create_dataloader()
        loader.plot_images()
        loader.dataset_details()

        trainer = Trainer(
            channels=kwargs["channels"],
            lr=kwargs["lr"],
            epochs=kwargs["epochs"],
            adam=kwargs["adam"],
            SGD=kwargs["SGD"],
            device=kwargs["device"],
            beta1=kwargs["beta1"],
            beta2=kwargs["beta2"],
            momentum=kwargs["momentum"],
            smooth=kwargs["smooth"],
            step_size=kwargs["step_size"],
            clamp=kwargs["clamp_value"],
            l1loss_value=kwargs["l1loss_value"],
            gamma=kwargs["gamma"],
            lr_scheduler=kwargs["lr_scheduler"],
            is_display=kwargs["display"],
            is_weight_init=kwargs["weight_init"],
        )

        trainer.train()
        trainer.plot_history()

    elif kwargs["test"]:
        test = TestModel(device=kwargs["device"])
        test.test()

    else:
        raise ValueError("Please, define either train or test flag".capitalize())


def get_config_args(args):
    config_files = config()

    return {
        "image_path": config_files["dataloader"]["image_path"],
        "batch_size": config_files["dataloader"]["batch_size"],
        "channels": config_files["dataloader"]["channels"],
        "image_size": config_files["dataloader"]["image_size"],
        "split_size": config_files["dataloader"]["split_size"],
        "beta1": config_files["model"]["beta1"],
        "beta2": config_files["model"]["beta2"],
        "lr": config_files["model"]["lr"],
        "epochs": config_files["model"]["epochs"],
        "momentum": config_files["model"]["momentum"],
        "step_size": config_files["model"]["step_size"],
        "gamma": config_files["model"]["gamma"],
        "clamp_value": config_files["model"]["clamp_value"],
        "l1loss_value": config_files["model"]["l1loss_value"],
        "smooth": config_files["model"]["smooth"],
        "device": config_files["model"]["device"],
        "adam": config_files["model"]["adam"],
        "SGD": config_files["model"]["SGD"],
        "lr_scheduler": config_files["model"]["lr_scheduler"],
        "display": config_files["model"]["display"],
        "weight_init": config_files["model"]["weight_init"],
        "train": args.train,
        "test": args.test,
    }


def cli():
    parser = argparse.ArgumentParser(description="CLI for seGAN".title())
    parser.add_argument(
        "--image_path",
        type=str,
        default="./datasets.zip",
        help="Define the image path".capitalize(),
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Define the batch size".capitalize()
    )
    parser.add_argument(
        "--channels", type=int, default=3, help="Define the image channels".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Define the image size".capitalize()
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=0.8,
        help="Define the split size".capitalize(),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Define the config file path".capitalize(),
    )
    parser.add_argument(
        "--l1loss_value",
        type=float,
        default=0.1,
        help="Define the l1loss value".capitalize(),
    )
    parser.add_argument(
        "--clamp_value",
        type=float,
        default=0.01,
        help="Define the clamp value".capitalize(),
    )
    parser.add_argument(
        "--smooth", type=float, default=1.0, help="Define the smooth value".capitalize()
    )
    parser.add_argument(
        "--step_size", type=int, default=100, help="Define the step size".capitalize()
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Define the gamma value".capitalize()
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Define the beta1 value".capitalize()
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Define the beta2 value".capitalize()
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Define the learning rate value".capitalize(),
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Define the epochs value".capitalize()
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="Define the momentum value".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Define the device".capitalize()
    )
    parser.add_argument(
        "--adam", type=bool, default=True, help="Define the adam value".capitalize()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Define the SGD value".capitalize()
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=True,
        help="Define the lr scheduler value".capitalize(),
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=True,
        help="Define the display value".capitalize(),
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",
        help="Define the weight init value".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.config:
        kwargs = get_config_args(args)
    else:
        kwargs = vars(args)

    execute(kwargs)


if __name__ == "__main__":
    cli()
