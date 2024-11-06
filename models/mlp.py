import torch.nn as nn

from constants import Constants, ModelParams

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.image_pixels = Constants.IMAGE_SIZE[0]*Constants.IMAGE_SIZE[1]
        self.fc1 = nn.Linear(self.image_pixels, ModelParams.FC1_NEURONS)
        self.fc2 = nn.Linear(ModelParams.FC1_NEURONS, ModelParams.FC2_NEURONS)
        self.relu = nn.ReLU()
        self.output = nn.Linear(ModelParams.FC2_NEURONS, ModelParams.OUTPUT_CLASSES)
    
    def forward(self, x):
        x = x.view(-1,self.image_pixels)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)