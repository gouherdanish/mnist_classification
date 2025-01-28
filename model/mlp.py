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

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP,self).__init__()
#         self.model_name = 'mlp'
#         self.image_pixels = DataConstants.IMAGE_SIZE[0]*DataConstants.IMAGE_SIZE[1]
#         self.fc1 = nn.Linear(self.image_pixels, MLPModelParams.NEURONS_FC1)
#         self.fc2 = nn.Linear(MLPModelParams.NEURONS_FC1, MLPModelParams.NEURONS_FC2)
#         self.fc3 = nn.Linear(MLPModelParams.NEURONS_FC2, DataConstants.OUTPUT_CLASSES)
    
#     def forward(self, x):
#         x = x.view(-1,self.image_pixels)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
    
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP,self).__init__()
#         self.model_name = 'mlp'
#         self.image_pixels = DataConstants.IMAGE_SIZE[0]*DataConstants.IMAGE_SIZE[1]
#         self.fc1 = nn.Linear(self.image_pixels, MLPModelParams.NEURONS_FC1)
#         self.fc2 = nn.Linear(MLPModelParams.NEURONS_FC1), MLPModelParams.NEURONS_FC2)
#         self.fc3 = nn.Linear(MLPModelParams.NEURONS_FC2), MLPModelParams.NEURONS_FC3)
#         self.fc4 = nn.Linear(MLPModelParams.NEURONS_FC3), DataConstants.OUTPUT_CLASSES)
    
#     def forward(self, x):
#         x = x.view(-1,self.image_pixels)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return self.fc4(x)