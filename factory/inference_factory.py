from pathlib import Path
import cv2
import torch 
import torch.nn as nn
from abc import ABC, abstractmethod

from checkpoint.model_checkpoint import ModelCheckpoint
from utils import Utils
from constants import PathConstants

class InferenceFactory:
    registry = {}

    @classmethod
    def register(cls, strategy):
        def inner(wrapped_cls):
            cls.registry[strategy] = wrapped_cls
            return wrapped_cls
        return inner
    
    @classmethod
    def get(cls, strategy, **kwargs):
        return cls.registry[strategy](**kwargs)
    

class InferenceStrategy(ABC):
    def __init__(self,model):
        self.model = model
        self.checkpoint_path = PathConstants.MODEL_PATH(model.model_name)
        assert Path(self.checkpoint_path).exists(), f"Model Path Not Found: {self.checkpoint_path}"
        self.checkpoint = ModelCheckpoint.load(checkpoint_path=self.checkpoint_path)
        self.model.load_state_dict(self.checkpoint['model_state'])
        print(self.checkpoint['epoch'])

    @abstractmethod
    def infer(self)->dict:
        pass

@InferenceFactory.register('batch')
class BatchInferenceStrategy(InferenceStrategy):
    def _infer_batch(self, X, y):
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            pred = out.argmax(dim=1)
            acc = (y==pred).sum()
        return acc.item()/len(y)

    def infer(self,test_loader)->dict:
        res = {
            'test_acc':[],
        }
        for batch_X, batch_y in test_loader:
            batch_acc = self._infer_batch(batch_X, batch_y)
            res['test_acc'].append(batch_acc)
        return res
    
@InferenceFactory.register('single')
class SingleInferenceStrategy(InferenceStrategy):
    def infer(self,test_loader)->dict:
        res = {}
        self.model.eval()
        with torch.no_grad():
            image_tensor, _ = next(iter(test_loader))
            out = self.model(image_tensor)
            probs = nn.functional.softmax(out)
            prob, pred = probs.max(1)
            res['confidence'] = prob
            res['pred_label'] = pred
        return res