import cv2
import torch 
import torch.nn as nn
from utils import Utils

class BatchInference:
    def __init__(
            self,
            model
        ):
        self.model = model

    def _infer_batch(self, X, y):
        self.model.eval() # set to evaluation mode
        with torch.no_grad():
            out = self.model(X)
            pred = out.argmax(dim=1)
            acc = (y==pred).sum()
        return pred, acc.item()/len(y)

    def infer(self,test_loader)->dict:
        res = {
            'test_acc':[],
        }
        for batch_X, batch_y in test_loader:
            pred, batch_acc = self._infer_batch(batch_X, batch_y)
            res['test_acc'].append(batch_acc)
        return res
