import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import DataConstants, LeNetModelParams

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.image_pixels = DataConstants.IMAGE_SIZE[0]*DataConstants.IMAGE_SIZE[1]
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5)
        self.fc1 = nn.Linear(
            in_features=16*4*4,
            out_features=120
        )
        self.fc2 = nn.Linear(
            in_features=120,
            out_features=84
        )
        self.fc3 = nn.Linear(
            in_features=84,
            out_features=DataConstants.OUTPUT_CLASSES
        )
    
    def forward(self, x):
        x = F.avg_pool2d(F.tanh(self.conv1(x)),kernel_size=2)
        x = F.avg_pool2d(F.tanh(self.conv2(x)),kernel_size=2)
        x = torch.flatten(x,start_dim=1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)