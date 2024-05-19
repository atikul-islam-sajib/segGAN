import os
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

from .utils import config, validate_path
from .discriminator_block import DiscriminatorBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.out_channels = 64

        layers = []

        for _ in range(3):
            layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels, out_channels=self.out_channels
                )
            )

            self.in_channels = self.out_channels
            self.out_channels *= 2

        for idx in range(2):
            layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels,
                    out_channels=1 if (idx == 1) else self.out_channels,
                    stride_size=1,
                    last_layer=(idx == 1),
                )
            )
            self.in_channels = self.out_channels
            self.out_channels *= 2

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x1 = self.model[0](x)
            x2 = self.model[1](x1)
            x3 = self.model[2](x2)
            x4 = self.model[3](x3)
            x5 = self.model[4](x4)

            return torch.cat(
                (
                    x1.view(x1.size(0), -1),
                    x2.view(x2.size(0), -1),
                    x3.view(x3.size(0), -1),
                    x4.view(x4.size(0), -1),
                    x5.view(x5.size(0), -1),
                ),
                dim=1,
            )

        else:
            raise ValueError("X should be in the format of tenor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, Discriminator):
            return sum(
                params.numel() for params in model.parameters() if params.requires_grad
            )

        else:
            raise ValueError(
                "Model should be in the format of Discriminator".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator code for seGAN".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the channels of the image".capitalize(),
    )

    args = parser.parse_args()

    config_files = config()
    files_path = validate_path(path=config_files["path"]["files_path"])

    in_channels = args.in_channels

    netD = Discriminator(in_channels=in_channels)

    draw_graph(
        model=netD, input_data=torch.randn(1, in_channels, 256, 256)
    ).visual_graph.render(filename=os.path.join(files_path, "netD"), format="jpeg")

    """
    To check:

        print(summary(model=netD, input_size=(in_channels, 256, 256)))
    
        assert Discriminator.total_params(model=netD) == 2766657

        assert netD(torch.randn(1, 3, 256, 256)).size() == (1, 2327940)
    """
