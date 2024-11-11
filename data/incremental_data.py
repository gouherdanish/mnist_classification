
from torch.utils.data import Dataset
from PIL import Image

class IncrementalDataset(Dataset):
    def __init__(self,image_path,transform=None) -> None:
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        img = Image.open(self.image_path).convert("L")
        if self.transform: 
            img = self.transform(img)
        return img