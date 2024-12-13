import torch.nn as nn

class ModelEvaluator:
    @staticmethod
    def count_params(model):
        return sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    @staticmethod
    def count_flops(model,input_size):
        total_flops = 0
        prev_size = input_size
        for layer in model.children():
            if isinstance(layer,nn.Conv2d):
                # Convolutional FLOPs
                C_in, H_in, W_in = prev_size
                C_out = layer.out_channels
                kernel_h, kernel_w = layer.kernel_size
                stride_h, stride_w = layer.stride
                padding_h, padding_w = layer.padding

                flops_mul = kernel_h * kernel_w * C_in
                flops_add = flops_mul - 1
                flops_per_output_pixel = flops_mul + flops_add

                H_out = (H_in - kernel_h + 2 * padding_h) // stride_h + 1
                W_out = (W_in - kernel_w + 2 * padding_w) // stride_w + 1
                total_output_pixels = H_out * W_out * C_out
                
                flops_conv = flops_per_output_pixel * total_output_pixels
                bias_add = total_output_pixels
                total_flops += flops_conv + bias_add
                prev_size = (C_out, H_out, W_out)
            elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                # Pooling FLOPs
                C_in, H_in, W_in = prev_size
                kernel_h, kernel_w = layer.kernel_size
                stride_h, stride_w = layer.stride
                
                flops_mul = kernel_h * kernel_w * C_in
                flops_add = flops_mul - 1
                flops_per_output_pixel = flops_mul + flops_add

                H_out = (H_in - kernel_h) // stride_h + 1
                W_out = (W_in - kernel_w) // stride_w + 1
                total_output_pixels = H_out * W_out
                
                flops_pool = flops_per_output_pixel * total_output_pixels
                bias_add = total_output_pixels
                total_flops += flops_pool + bias_add
                prev_size = (C_in, H_out, W_out)
            elif isinstance(layer,nn.Linear):
                # Fully connected FLOPs
                flops_mul = layer.in_features
                flops_add = flops_mul - 1
                flops_per_output_neuron = flops_mul + flops_add

                total_output_neurons = layer.out_features
                flops_fc = flops_per_output_neuron * total_output_neurons
                bias_add = total_output_neurons
                total_flops += flops_fc + bias_add
                prev_size = (layer.out_features,)
        return total_flops

    def evaluate(
            self,
            model:nn.Module,
            input_size:tuple):
        result = {}
        result['param_count'] = self.count_params(model)
        result['flops'] = self.count_flops(model,input_size=input_size)
        return result
