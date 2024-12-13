from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from constants import HyperParams, PathConstants
from utils import Utils
from checkpoint.model_checkpoint import ModelCheckpoint

class TrainingFactory:
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
    

class ModelTraining(ABC):
    def __init__(
            self,
            model:nn.Module,
            optimizer=None,
            lr=HyperParams.LEARNING_RATE,
            loss_fn=nn.CrossEntropyLoss()
        ):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = loss_fn
        self.last_epoch = 0
        self.checkpoint_path = PathConstants.MODEL_PATH(model.model_name)
        self.checkpoint = ModelCheckpoint.load(checkpoint_path=self.checkpoint_path)
        if self.checkpoint:
            self.last_epoch = self.checkpoint['epoch']
            self.model.load_state_dict(self.checkpoint['model_state'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state'])

    @abstractmethod
    def train(self):
        pass

@TrainingFactory.register('batch')
class BatchTraining(ModelTraining):
    def train_loop(self,train_loader):
        self.model.train()
        total_loss, acc, count = 0,0,0
        for batch, (X,y) in enumerate(train_loader):
            pred = self.model(X)
            loss = self.loss_fn(pred,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            acc += (y==pred.argmax(1)).sum()
            count += len(y)
            # if batch % 100 == 0:
            #     print(f'Batch {batch} : size={len(y)} loss={total_loss.item()/count} acc={acc.item()/count}')
        return total_loss.item()/count, acc.item()/count
    
    def val_loop(self,val_loader):
        self.model.eval()
        total_loss, acc, count = 0,0,0
        with torch.no_grad():
            for X,y in val_loader:
                out = self.model(X)
                loss = self.loss_fn(out,y)
                _,pred = torch.max(out,1)
                acc += (pred==y).sum()
                total_loss += loss
                count+=len(y)
        return total_loss.item()/count, acc.item()/count
    
    @Utils.timeit
    def train(
            self,
            train_loader,
            val_loader=None,
            epochs=10):
        res = {
            'train_loss':[],
            'val_loss':[],
            'train_acc':[],
            'val_acc':[]
        }
        self.with_validation = val_loader and len(val_loader) != 0
        for epoch in range(self.last_epoch+1,epochs+1):
            tl,ta = self.train_loop(train_loader)
            res['train_loss'].append(tl)
            res['train_acc'].append(ta)
            if self.with_validation:
                vl,va = self.val_loop(val_loader)
                res['val_loss'].append(vl)
                res['val_acc'].append(va)
            if epoch == 1 or epoch == epochs or epoch%5 == 0:
                if self.with_validation:
                    print(f"Epoch {epoch:02}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
                else:
                    print(f"Epoch {epoch:02}, Train acc={ta:.3f}, Train loss={tl:.3f}")
                ModelCheckpoint.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    checkpoint_path=self.checkpoint_path
                )
        return res
    
@TrainingFactory.register('single')
class SingleTraining(ModelTraining):
    def _train_loop(self,train_loader):
        self.model.train()
        X,y = next(iter(train_loader))
        pred = self.model(X)
        print(pred,y)
        loss = self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @Utils.timeit
    def train(self,train_loader):
        print(f'LAST EPOCH : {self.last_epoch}')
        self._train_loop(train_loader)
        ModelCheckpoint.save(
            epoch=self.last_epoch+1,
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=self.checkpoint_path
        )
    