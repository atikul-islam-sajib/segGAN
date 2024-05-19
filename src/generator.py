import os
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from collections import OrderedDict

from .encoder import EncoderBlock
from .decoder import DecoderBlock
from .utils import config, validate_path


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 64

        self.encoder1 = EncoderBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            use_leaky_relu=True,
            use_batch_norm=False,
        )
        self.encoder2 = EncoderBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels * 2,
            use_leaky_relu=True,
            use_batch_norm=True,
        )
        self.encoder3 = EncoderBlock(
            in_channels=self.out_channels * 2,
            out_channels=self.out_channels * 4,
            use_leaky_relu=True,
            use_batch_norm=True,
        )
        self.encoder4 = EncoderBlock(
            in_channels=self.out_channels * 4,
            out_channels=self.out_channels * 8,
            use_leaky_relu=True,
            use_batch_norm=True,
        )
        self.encoder5 = EncoderBlock(
            in_channels=self.out_channels * 8,
            out_channels=self.out_channels * 8,
            use_leaky_relu=True,
            use_batch_norm=True,
        )
        self.encoder6 = EncoderBlock(
            in_channels=self.out_channels * 8,
            out_channels=self.out_channels * 8,
            use_leaky_relu=True,
            use_batch_norm=True,
        )
        self.encoder7 = EncoderBlock(
            in_channels=self.out_channels * 8,
            out_channels=self.out_channels * 8,
            use_leaky_relu=False,
            use_batch_norm=False,
        )

        self.decoder1 = DecoderBlock(
            in_channels=self.out_channels * 8,
            out_channels=self.out_channels * 8,
        )
        self.decoder2 = DecoderBlock(
            in_channels=self.out_channels * 8 * 2,
            out_channels=self.out_channels * 8,
        )
        self.decoder3 = DecoderBlock(
            in_channels=self.out_channels * 8 * 2,
            out_channels=self.out_channels * 8,
        )
        self.decoder4 = DecoderBlock(
            in_channels=self.out_channels * 8 * 2,
            out_channels=self.out_channels * 4,
        )
        self.decoder5 = DecoderBlock(
            in_channels=self.out_channels * 4 * 2,
            out_channels=self.out_channels * 2,
        )
        self.decoder6 = DecoderBlock(
            in_channels=self.out_channels * 2 * 2,
            out_channels=self.out_channels,
        )
        self.decoder7 = DecoderBlock(
            in_channels=self.out_channels * 2,
            out_channels=in_channels,
            last_layer=True,
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            encoder1 = self.encoder1(x)
            encoder2 = self.encoder2(encoder1)
            encoder3 = self.encoder3(encoder2)
            encoder4 = self.encoder4(encoder3)
            encoder5 = self.encoder5(encoder4)
            encoder6 = self.encoder6(encoder5)
            encoder7 = self.encoder7(encoder6)

            decoder1 = self.decoder1(encoder7, encoder6)
            decoder2 = self.decoder2(decoder1, encoder5)
            decoder3 = self.decoder3(decoder2, encoder4)
            decoder4 = self.decoder4(decoder3, encoder3)
            decoder5 = self.decoder5(decoder4, encoder2)
            decoder6 = self.decoder6(decoder5, encoder1)
            output = self.decoder7(decoder6)

            return output

        else:
            raise ValueError("X should be in the format of tensor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, Generator):
            return sum(p.numel() for p in model.parameters())

        else:
            raise ValueError("Model should be in the Generator".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("netG for the seGAN".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="The number of channels in the input".capitalize(),
    )
    args = parser.parse_args()

    config_files = config()
    files_path = validate_path(config_files["path"]["files_path"])

    in_channels = args.in_channels

    netG = Generator(in_channels=in_channels)

    draw_graph(model=netG, input_data=torch.randn(1, 3, 256, 256)).visual_graph.render(
        filename=os.path.join(files_path, "netG"), format="jpeg"
    )

    """
    To check:
    
        print(summary(model=netG, input_size=(3, 256, 256)))
    
        assert netG(torch.randn(1, 3, 256, 256)).size() == (1, 3, 256, 256)

        assert netG.total_params(netG) == 41828992
    """
