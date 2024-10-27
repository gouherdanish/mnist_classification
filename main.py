import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

if __name__=='__main__':
    train_ds = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=False
    )