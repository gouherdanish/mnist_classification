import time, os
import torch
import torch.nn as nn

from constants import DataConstants, PathConstants
from factory.inference_factory import InferenceFactory

class ModelEvaluator:
    @staticmethod
    def model_size(model):
        size_bytes = os.path.getsize(PathConstants.MODEL_PATH(model.model_name))
        return f"{size_bytes / 1e6:.1f} MB"

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

    @staticmethod
    def compute_energy(runtime):
        factor = 713        # For India, Units: gCO2eq/kWh
        ram_power = 3       # 3 Watts for 8 GB https://mlco2.github.io/codecarbon/methodology.html#
        cpu_power = 13      # 13 Watts for MAC M1 https://versus.com/en/apple-m1/cpu-tdp
        gpu_power = 0
        return (ram_power + cpu_power + gpu_power) * runtime * factor / 1000
        
    def inference(self,model,dataloader):
        latency_list, acc_list, energy_list = [], [], []
        inferencing = InferenceFactory.get(strategy='batch',model=model)
        for batch_X, batch_y in dataloader:
            start_time = time.time()
            batch_acc = inferencing._infer_batch(batch_X, batch_y)
            end_time = time.time()
            runtime = end_time - start_time
            latency = runtime / len(batch_y)
            latency_list.append(latency)
            energy_list.append(self.compute_energy(latency))
            acc_list.append(batch_acc)
        return f"{(1e6 * sum(latency_list) / len(latency_list)):.1f} \u03BCs", f"{(100* sum(acc_list) / len(acc_list)):.1f}%", f"{(1e6* sum(energy_list) / len(energy_list)):.1f} \u03BCgCO2eq"

    def evaluate(
            self,
            model:nn.Module,
            dataloader:torch.utils.data.DataLoader):
        result = {}
        result['size'] = self.model_size(model)
        result['params'] = self.count_params(model)
        result['flops'] = self.count_flops(model,input_size=(DataConstants.IN_CHANNELS,*DataConstants.IMAGE_SIZE))
        result['latency'], result['accuracy'], result['CO2eq'] = self.inference(model,dataloader)
        return result
