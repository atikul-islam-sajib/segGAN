import os
import torch
import unittest

from src.utils import config, load, validate_path
from src.dataloader import Loader
from src.generator import Generator
from src.discriminator import Discriminator


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.config_files = config()

        self.processed_path = validate_path(
            path=self.config_files["path"]["processed_path"]
        )

        self.train_dataloader = load(
            filename=os.path.join(self.processed_path, "train_dataloader.pkl")
        )
        self.test_dataloader = load(
            filename=os.path.join(self.processed_path, "test_dataloader.pkl")
        )

        self.netG = Generator(in_channels=3)
        self.netD = Discriminator(in_channels=3)

    def test_train_data_quantity(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.train_dataloader), 8)

    def test_test_data_quantity(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.test_dataloader), 2)

    def test_total_data_quantity(self):
        self.total_data = sum(X.size(0) for X, _ in self.train_dataloader) + sum(
            X.size(0) for X, _ in self.test_dataloader
        )

        self.assertEqual(self.total_data, 10)

    def test_train_batch_size(self):
        self.assertEqual(self.train_dataloader.batch_size, 1)

    def test_test_batch_size(self):
        self.assertEqual(self.test_dataloader.batch_size, 8)

    def test_train_channels(self):
        X, y = next(iter(self.train_dataloader))
        self.assertEqual(X.size(1), 3)

    def test_test_channels(self):
        X, y = next(iter(self.test_dataloader))
        self.assertEqual(X.size(1), 3)

    def test_train_dataloader_type(self):
        self.assertEqual(type(self.train_dataloader), torch.utils.data.DataLoader)

    def test_test_dataloader_type(self):
        self.assertEqual(type(self.test_dataloader), torch.utils.data.DataLoader)

    def test_netG_size(self):
        self.assertEqual(
            self.netG(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 3, 256, 256])
        )

    def test_netG_total_params(self):
        self.assertEqual(sum(p.numel() for p in self.netG.parameters()), 41828992)

    def test_netD_size(self):
        self.assertEqual(
            self.netD(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 2327940])
        )

    def test_netD_total_params(self):
        self.assertEqual(sum(p.numel() for p in self.netD.parameters()), 2766657)


if __name__ == "__main__":
    unittest.main()
