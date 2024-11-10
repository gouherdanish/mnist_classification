import cv2
import torch 
import torch.nn as nn
from utils import Utils

class IncrementalInference:
    def __init__(
            self,
            model
        ):
        self.model = model

    @Utils.timeit
    def _infer_batch(self, X, y):
        self.model.eval() # set to evaluation mode
        with torch.no_grad():
            out = self.model(X)
            pred = out.argmax(dim=1)
            acc = (y==pred).sum()
        return acc.item()/len(y)

    def infer(self,test_loader)->dict:
        res = {
            'test_acc':[],
        }
        self.model.eval()
        with torch.no_grad():
            for image_tensor in test_loader:
                out = self.model(image_tensor)
                pred = out.argmax(dim=1)
        return pred, 
