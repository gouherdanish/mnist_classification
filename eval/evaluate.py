import torch.nn as nn

class ModelEvaluator:
    @staticmethod
    def count_params(model):
        return sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    @staticmethod
    def count_flops(model):
        total_flops = 0
        for layer in model.children():
            if isinstance(layer,nn.Linear):
                flops_matmul = layer.in_features * layer.out_features
                flops_bias = layer.out_features
                total_flops += flops_matmul + flops_bias
        return total_flops

    def evaluate(self,model:nn.Module):
        result = {}
        result['param_count'] = self.count_params(model)
        result['flops'] = self.count_flops(model)
        return result
