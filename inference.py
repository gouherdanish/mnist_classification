import cv2
import torch 
import torch.nn as nn
from utils import Utils

class ModelInference:
    def __init__(
            self,
            model,
            test_loader
        ):
        self.model = model
        self.test_loader = test_loader

    @Utils.timeit
    def _infer_batch(self, X, y):
        self.model.eval() # set to evaluation mode
        with torch.no_grad():
            out = self.model(X)
            pred = out.argmax(dim=1)
            acc = (y==pred).sum()
        return acc.item()/len(y)

    def infer(self):
        res = {
            'test_acc':[],
        }
        for batch_X, batch_y in self.test_loader:
            batch_acc = self._infer_batch(batch_X, batch_y)
            res['test_acc'].append(batch_acc)
        return res
