import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

from constants import PathConstants

class ModelCheckpoint:

    @staticmethod
    def save(
            epoch:int,
            model:nn.Module,
            optimizer:torch.optim,
            checkpoint_path:Union[str,Path]) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(checkpoint,checkpoint_path)

    @staticmethod
    def load(
            checkpoint_path:str, 
            model:nn.Module, 
            optimizer:torch.optim) -> nn.Module:
        last_epoch = 0
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path) 
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                if 'epoch' in checkpoint.keys():
                    last_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['model_state'])
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
            return last_epoch, model, optimizer
        return last_epoch, model, optimizer