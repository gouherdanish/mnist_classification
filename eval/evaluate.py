import torch.nn as nn

class ModelEvaluator:
    def __init__(self,model:nn.Module) -> None:
        self.model = model
        self.result = {}

    def count_params(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])
    
    def count_flops(self):
        total_flops = 0
        for layer in self.model.children():
            if isinstance(layer,nn.Linear):
                flops_matmul = layer.in_features * layer.out_features
                flops_bias = layer.out_features
                total_flops += flops_matmul + flops_bias
        return total_flops

    
    def evaluate(self):
        self.result['param_count'] = self.count_params()
