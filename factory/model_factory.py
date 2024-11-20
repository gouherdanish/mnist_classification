import torch.nn as nn
from model.mlp import MLP
from model.lenet import LeNet

class ModelFactory:
    model_registry = {
        'mlp': MLP,
        'lenet': LeNet
    }

    def select(self,model: str) -> nn.Module:
        if model not in self.model_registry:
            raise ValueError(f'Unsupported model: {model}')
        return self.model_registry[model]()