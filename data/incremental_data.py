import torchvision
from torch.utils.data import Dataset
from PIL import Image

from custom.transforms import InvertIntensity

class IncrementalDataset(Dataset):
    def __init__(self,image_path) -> None:
        self.image_path = image_path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            InvertIntensity(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        img = Image.open(self.image_path).convert("L")  # convert to grayscale; 'L' mode stores just the Luminance value
        if self.transform: 
            img = self.transform(img)
        return img