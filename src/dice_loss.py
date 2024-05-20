import argparse
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.001):
        super(DiceLoss, self).__init__()

        self.loss_name = "Dice Loss".capitalize()

        self.smooth = smooth

    def forward(self, predicted, actual):
        if isinstance(predicted, torch.Tensor) and isinstance(actual, torch.Tensor):
            actual = actual.view(-1)
            predicted = predicted.view(-1)

        intersection = (predicted * actual).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        return 1.0 - dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Loss for seGAN".capitalize())
    parser.add_argument(
        "--smooth", type=float, default=0.001, help="smoothing factor".capitalize()
    )

    args = parser.parse_args()

    diceloss = DiceLoss(smooth=args.smooth)

    actual = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    predicted = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])

    print("The loss is {}".format(diceloss(predicted, actual)))
