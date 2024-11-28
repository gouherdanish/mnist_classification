from abc import ABC, abstractmethod

class ModelData(ABC):
    @abstractmethod
    def train_flag(self):
        pass 

    @abstractmethod
    def shuffle_flag(self):
        pass

    @abstractmethod
    def download_flag(self):
        pass