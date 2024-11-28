import torchvision 

class BatchData:
    def __init__(self,train=True,download=False) -> None:
        self.train = train
        self.download = download
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5),(0.5))
        ])

    def load(self):
        return torchvision.datasets.MNIST(
            root="data",
            train=self.train,
            download=self.download,
            transform=self.transform
        )