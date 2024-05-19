import torch
import argparse
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(L1Loss, self).__init__()

        self.loss_name = "L1Loss".title()

        self.reduction = reduction

        self.l1_loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, actual, predicted):
        if isinstance(actual, torch.Tensor) and isinstance(predicted, torch.Tensor):
            return self.l1_loss(actual, predicted)

        else:
            raise ValueError(
                "Actual and Predicted should be in the format of tensor".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the L1 loss for seGAN".title())
    parser.add_argument(
        "--reduction", type=str, default="mean", help="Define the reduction method"
    )
    args = parser.parse_args()

    l1loss = L1Loss(reduction=args.reduction)

    actual = torch.tensor([1.0, 0.0, 1.0])
    predicted = torch.tensor([1.0, 0.0, 1.0])

    print("Total loss is # {}".format(l1loss(actual, predicted)))
