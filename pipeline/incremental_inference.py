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

    def infer(self,test_loader)->dict:
        self.model.eval()
        with torch.no_grad():
            image_tensor = next(iter(test_loader))
            out = self.model(image_tensor)
            probs = nn.functional.softmax(out)
            prob, pred = probs.max(1)
        return prob, pred
