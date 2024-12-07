import torch.nn as nn
import torch.nn.functional as F

from constants import DataConstants, MLPModelParams

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.model_name = 'mlp'
        self.image_pixels = DataConstants.IMAGE_SIZE[0]*DataConstants.IMAGE_SIZE[1]
        self.fc1 = nn.Linear(self.image_pixels, MLPModelParams.NEURONS_FC1)
        self.fc2 = nn.Linear(MLPModelParams.NEURONS_FC1, DataConstants.OUTPUT_CLASSES)
    
    def forward(self, x):
        x = x.view(-1,self.image_pixels)
        x = F.relu(self.fc1(x))
        return self.fc2(x)