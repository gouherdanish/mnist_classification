from typing import Union
import torch
import torchvision

from data.model_data import ModelData
from data.batch_data import BatchData
from data.incremental_data import IncrementalDataset
from constants import DataConstants, HyperParams

class CustomDataBuilder:
    def __init__(
            self,
            data_params:ModelData,
            sample_image_path:Union[str,None]) -> None:
        self.train = data_params.train_flag()
        self.shuffle = data_params.shuffle_flag()
        self.download = data_params.download_flag()
        self.sample_image_path = sample_image_path
        self.is_incremental = True if sample_image_path != '' else False
        self.batch_size = 1 if self.is_incremental else HyperParams.BATCH_SIZE 

    def load_data(self):
        if self.is_incremental:
            data = IncrementalDataset(image_path=self.sample_image_path)
        else:
            data = BatchData(train=self.train,download=self.download)
        return data.load()
        
    def prepare(self):
        dataset = self.load_data()
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=self.shuffle, 
            batch_size=self.batch_size)
        return dataloader