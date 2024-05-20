import os
import yaml
import torch
import joblib
import traceback


def config():
    with open("./config.yml", "r") as file:
        config_files = yaml.safe_load(file)

    return config_files


class PathException(Exception):
    def __init__(self, message):
        super(PathException, self).__init__(message)
        self.message = message


def validate_path(path):
    if os.path.exists(path):
        return path
    else:
        traceback.print_exc()
        raise PathException("{} Path does not exist".capitalize().format(path))


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value, filename)

    else:
        raise PathException("{} Path does not exist".capitalize().format(filename))


def load(filename=None):
    if filename is not None:
        return joblib.load(filename)

    else:
        raise PathException("{} Path does not exist".capitalize().format(filename))


def device_init(device="cuda"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        return torch.device("cpu")
