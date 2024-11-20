import torchvision 

class BatchData:
    def __init__(self,train=True,transform=None) -> None:
        self.train=train
        self.transform=transform

    def load(self):
        if self.train:
            return torchvision.datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=self.transform
            )
        else:
            return torchvision.datasets.MNIST(
                root="data",
                train=False,
                download=False,
                transform=self.transform
            )