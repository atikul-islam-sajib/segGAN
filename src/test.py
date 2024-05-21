import os
import torch
import argparse
import matplotlib.pyplot as plt

from .utils import validate_path, config, device_init, load
from .generator import Generator


class TestModel:
    def __init__(self, device="cuda"):
        try:
            self.device = device_init(device=device)
            self.test_dataloader = validate_path(config()["path"]["processed_path"])
            self.best_model = validate_path(
                path=os.path.join(config()["path"]["best_model_path"])
            )
            self.test_image_path = validate_path(
                path=config()["path"]["test_image_path"]
            )
            self.netG = Generator().to(self.device)
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def load_dataset(self):
        try:
            return load(
                filename=os.path.join(self.test_dataloader, "test_dataloader.pkl")
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def load_best_model(self):
        try:
            self.netG.load_state_dict(
                torch.load(os.path.join(self.best_model, "best_model.pth"))["netG"]
            )
        except Exception as e:
            print(f"Error loading the best model: {e}")
            raise

    def plot(self):
        try:
            plt.figure(figsize=(20, 10))

            image, mask = next(iter(self.load_dataset()))
            predict = self.netG(image.to(self.device))

            size = image.size(0)
            if size == 1:
                num_row, num_columns = 1, 1
            else:
                num_row = size // 2
                num_columns = size // num_row

            for index, image in enumerate(predict):
                plt.subplot(2 * num_row, 2 * num_columns, 2 * index + 1)
                image = image.squeeze().cpu().permute(1, 2, 0).detach().numpy()
                masks = mask[index].squeeze().cpu().permute(1, 2, 0).detach().numpy()

                image = (image - image.min()) / (image.max() - image.min())
                masks = (masks - masks.min()) / (masks.max() - masks.min())

                plt.imshow(image, cmap="gray")
                plt.axis("off")
                plt.title("Generated Image")

                plt.subplot(2 * num_row, 2 * num_columns, 2 * index + 2)
                plt.imshow(masks)
                plt.title("Mask")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(self.test_image_path, "test.png"))
            plt.show()

            print("Test image is saved in the path: ", self.test_image_path)
        except Exception as e:
            print(f"Error during plotting: {e}")
            raise

    def test(self):
        try:
            self.load_best_model()
            self.plot()
        except Exception as e:
            print(f"Error during testing: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model for seGAN".title())
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="device for running the model".capitalize(),
    )
    args = parser.parse_args()

    try:
        test = TestModel(device=args.device)
        test.test()
    except Exception as e:
        print(f"An error occurred: {e}")
