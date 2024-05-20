import argparse
import torch

from .helper import helpers


class Trainer:
    def __init__(
        self,
        channels=3,
        lr=0.0002,
        epochs=100,
        adam=True,
        SGD=False,
        beta1=0.5,
        beta2=0.999,
        momentum=0.90,
        smooth=0.001,
        lr_scheduler=False,
        is_display=True,
    ):

        self.channels = channels
        self.lr = lr
        self.epochs = epochs
        self.adam = adam
        self.SGD = SGD
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.smooth = smooth
        self.lr_scheduler = lr_scheduler
        self.is_display = is_display

    def l2_loss(self, model):
        pass

    def elastic_loss(self, model):
        pass

    def saved_model_checkpoints(self, **kwargs):
        pass

    def saved_training_images(self, **kwargs):
        pass

    def show_progress(self, **kwargs):
        pass

    def update_train_netG(self, **kwargs):
        pass

    def update_train_netD(self, **kwargs):
        pass

    def train(self):
        pass
