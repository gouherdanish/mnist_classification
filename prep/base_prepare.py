from abc import ABC, abstractmethod

class DataPreparation(ABC):
    @abstractmethod
    def prepare(self):
        pass