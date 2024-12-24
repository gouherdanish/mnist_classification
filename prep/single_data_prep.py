from typing import Union
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision 

from prep.base_prepare import DataPreparation
from data.single_data import SingleDataset
from custom.transforms import InvertIntensity

class SingleDataPreparation(DataPreparation):
    def __init__(
            self,
            img:Union[str,Path,np.ndarray],
            label:int=-1) -> None:
        self.img = img
        self.label = label
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            InvertIntensity(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])

    def _load_data(self):
        """
        Function to load data into a PyTorch Dataset object
        - if path string is provided, it uses cv2 to read the image from the path into a numpy array
        - if numpy array is provided directly, it is used as is
        """
        if isinstance(self.img,str):
            self.img = Path(self.img)
        if isinstance(self.img,Path):
            self.img = cv2.imread(self.img,cv2.IMREAD_GRAYSCALE)
        return SingleDataset(
            img=self.img,
            label=self.label,
            transform=self.transform
        )

    def prepare(self):
        dataset = self._load_data()
        loader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False, 
            batch_size=1)
        return loader