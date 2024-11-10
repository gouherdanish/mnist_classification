import torch.nn as nn
from arch.mlp import MLP

class ModelFactory:
    model_registry = {
        'mlp': MLP
    }

    def select(self,model: str) -> nn.Module:
        if model not in self.model_registry:
            raise ValueError(f'Unsupported model: {model}')
        return self.model_registry[model]()