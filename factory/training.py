import torch
import torch.nn as nn
from constants import Constants, HyperParams
from utils import Utils

class ModelTraining:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer=None,
            lr=HyperParams.LEARNING_RATE,
            loss_fn=nn.CrossEntropyLoss()
        ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = loss_fn

    def train_loop(self):
        self.model.train()
        total_loss, acc, count = 0,0,0
        for batch, (X,y) in enumerate(self.train_loader):
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
    
    def val_loop(self):
        self.model.eval()
        total_loss, acc, count = 0,0,0
        with torch.no_grad():
            for X,y in self.val_loader:
                out = self.model(X)
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
        for ep in range(epochs):
            tl,ta = self.train_loop()
            vl,va = self.val_loop()
            print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
            res['train_loss'].append(tl)
            res['train_acc'].append(ta)
            res['val_loss'].append(vl)
            res['val_acc'].append(va)
        return res
    