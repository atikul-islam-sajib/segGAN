import torch
import argparse
import torch.nn as nn
from collections import OrderedDict


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=64, use_leaky_relu=True, use_batch_norm=False
    ):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaky_relu = use_leaky_relu
        self.batch_norm = use_batch_norm

        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = 1

        self.encoder = self.block()

    def block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=False,
        )

        if self.leaky_relu:
            layers["leaky_ReLU"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.batch_norm:
            layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder(x)

        else:
            raise ValueError("X should be in the format of tensor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, torch.nn.modules.container.Sequential):
            return sum(params.numel() for params in model.parameters())

        else:
            raise ValueError("Model should be in the format of Sequential".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Encoder Block for seGAN".capitalize()
    )
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

    encoder1 = EncoderBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        use_leaky_relu=True,
        use_batch_norm=False,
    )
    encoder2 = EncoderBlock(
        in_channels=out_channels,
        out_channels=out_channels * 2,
        use_leaky_relu=True,
        use_batch_norm=True,
    )
    encoder3 = EncoderBlock(
        in_channels=out_channels * 2,
        out_channels=out_channels * 4,
        use_leaky_relu=True,
        use_batch_norm=True,
    )
    encoder4 = EncoderBlock(
        in_channels=out_channels * 4,
        out_channels=out_channels * 8,
        use_leaky_relu=True,
        use_batch_norm=True,
    )
    encoder5 = EncoderBlock(
        in_channels=out_channels * 8,
        out_channels=out_channels * 8,
        use_leaky_relu=True,
        use_batch_norm=True,
    )
    encoder6 = EncoderBlock(
        in_channels=out_channels * 8,
        out_channels=out_channels * 8,
        use_leaky_relu=True,
        use_batch_norm=True,
    )
    encoder7 = EncoderBlock(
        in_channels=out_channels * 8,
        out_channels=out_channels * 8,
        use_leaky_relu=False,
        use_batch_norm=False,
    )

    for block in [encoder1, encoder2, encoder3, encoder4, encoder5, encoder6, encoder7]:
        layers.append(block)

    model = nn.Sequential(*layers)

    assert EncoderBlock.total_params(model=model) == 15342336

    print(model(torch.randn(1, 3, 256, 256)).size())
