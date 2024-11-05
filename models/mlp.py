import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)