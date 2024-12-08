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
    def load(checkpoint_path):
        if checkpoint_path and Path(checkpoint_path).exists():
            return torch.load(checkpoint_path, weights_only=True)