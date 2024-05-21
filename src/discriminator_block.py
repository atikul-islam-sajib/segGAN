import torch
import argparse
import torch.nn as nn
from collections import OrderedDict


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        kernel_size=4,
        stride_size=2,
        padding_size=1,
        last_layer=False,
    ):
        super(DiscriminatorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_layer = last_layer

        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding_size = padding_size

        self.discriminator_block = self.block()

    def block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
        )

        if self.last_layer:
            pass

        else:
            layers["leaky_ReLU"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.discriminator_block(x)

        else:
            raise ValueError("Input should be a tensor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, torch.nn.modules.container.Sequential):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            raise ValueError("Model should be a Sequential model".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator for seGAN".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Number of output channels".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels

    layers = []

    for _ in range(3):
        layers.append(
            DiscriminatorBlock(in_channels=in_channels, out_channels=out_channels)
        )

        in_channels = out_channels
        out_channels *= 2

    for idx in range(2):
        layers.append(
            DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=1 if (idx == 1) else out_channels,
                stride_size=1,
                last_layer=(idx == 1),
            )
        )
        in_channels = out_channels
        out_channels *= 2

    model = nn.Sequential(*layers)

    assert DiscriminatorBlock.total_params(model=model) == 2766657
    assert model(torch.randn(1, 3, 256, 256)).size() == (1, 1, 30, 30)
