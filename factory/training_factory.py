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
    @abstractmethod
    def train(self):
        pass

@TrainingFactory.register('batch')
class BatchTraining(ModelTraining):
    def __init__(
            self,
            model:nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
            optimizer=None,
            lr=HyperParams.LEARNING_RATE,
            loss_fn=nn.CrossEntropyLoss(),
            checkpoint_path=None
        ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = loss_fn
        self.with_validation = len(val_loader) != 0
        self.checkpoint_path = checkpoint_path

    def train_loop(self,model,optimizer):
        model.train()
        total_loss, acc, count = 0,0,0
        for batch, (X,y) in enumerate(self.train_loader):
            pred = model(X)
            loss = self.loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            acc += (y==pred.argmax(1)).sum()
            count += len(y)
            # if batch % 100 == 0:
            #     print(f'Batch {batch} : size={len(y)} loss={total_loss.item()/count} acc={acc.item()/count}')
        return total_loss.item()/count, acc.item()/count
    
    def val_loop(self,model):
        model.eval()
        total_loss, acc, count = 0,0,0
        with torch.no_grad():
            for X,y in self.val_loader:
                out = model(X)
                loss = self.loss_fn(out,y)
                _,pred = torch.max(out,1)
                acc += (pred==y).sum()
                total_loss += loss
                count+=len(y)
        return total_loss.item()/count, acc.item()/count

    @Utils.timeit
    def train(self,epochs):
        res = {
            'train_loss':[],
            'val_loss':[],
            'train_acc':[],
            'val_acc':[]
        }
        last_epoch,model,optimizer = ModelCheckpoint.load(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            optimizer=self.optimizer
        )
        for epoch in range(last_epoch+1,epochs+1):
            tl,ta = self.train_loop(model,optimizer)
            res['train_loss'].append(tl)
            res['train_acc'].append(ta)
            if self.with_validation:
                vl,va = self.val_loop(model)
                res['val_loss'].append(vl)
                res['val_acc'].append(va)
            if epoch == 1 or epoch == epochs or epoch%5 == 0:
                if self.with_validation:
                    print(f"Epoch {epoch:02}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
                else:
                    print(f"Epoch {epoch:02}, Train acc={ta:.3f}, Train loss={tl:.3f}")
                ModelCheckpoint.save(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    checkpoint_path=self.checkpoint_path
                )
        return res
    
@TrainingFactory.register('single')
class SingleTraining(ModelTraining):
    def __init__(
            self,
            model:nn.Module,
            train_loader:DataLoader,
            optimizer=None,
            lr=HyperParams.LEARNING_RATE,
            loss_fn=nn.CrossEntropyLoss(),
            checkpoint_path=''
        ):
        self.model = model
        self.train_loader = train_loader
        self.lr = lr
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = loss_fn
        self.checkpoint_path=checkpoint_path

    def _train_loop(self,model,optimizer):
        model.train()
        X,y = next(iter(self.train_loader))
        pred = model(X)
        print(pred,y)
        loss = self.loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self):
        print(self.checkpoint_path)
        last_epoch,model,optimizer = ModelCheckpoint.load(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            optimizer=self.optimizer
        )
        print(f'LAST EPOCH : {last_epoch}')
        self._train_loop(model,optimizer)
        ModelCheckpoint.save(
            epoch=last_epoch+1,
            model=model,
            optimizer=optimizer,
            checkpoint_path=self.checkpoint_path
        )
    