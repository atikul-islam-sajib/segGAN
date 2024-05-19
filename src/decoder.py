import torch
import argparse
import torch.nn as nn
from collections import OrderedDict


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, last_layer=False):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_layer = last_layer

        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = 1

        self.decoder = self.block()

    def block(self):
        layers = OrderedDict()

        layers["convTranspose"] = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=False,
        )
        if self.last_layer:
            layers["Tanh"] = nn.Tanh()
        else:
            layers["ReLU"] = nn.ReLU(inplace=True)
            layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)

        return nn.Sequential(layers)

    def forward(self, x, skip_info=None):
        if isinstance(x, torch.Tensor) and isinstance(skip_info, torch.Tensor):
            x = self.decoder(x)
            return torch.cat((x, skip_info), dim=1)

        else:
            if isinstance(x, torch.Tensor) and skip_info is None:
                return self.decoder(x)

    @staticmethod
    def total_params(model):
        if isinstance(model, torch.nn.modules.container.Sequential):
            return sum(params.numel() for params in model.parameters())

        else:
            raise ValueError("Model should be in the format of Sequential".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder block for netG".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=512,
        help="Number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=512,
        help="Number of output channels".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels

    model = DecoderBlock(in_channels=in_channels, out_channels=out_channels)

    print(model)
