from pathlib import Path
import argparse
import torch

from constants import PathConstants
from prep.batch_data_prep_training import BatchTrainingDataPreparation
from factory.model_factory import ModelFactory
from factory.training_factory import TrainingFactory
from checkpoint.model_checkpoint import ModelCheckpoint


def run(args):
    model_name = args.model_name
    epochs = args.epochs

    data_prep = BatchTrainingDataPreparation()
    train_loader, val_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_name)

    training = TrainingFactory.get(strategy='batch',model=model)
    hist = training.train(train_loader=train_loader,val_loader=val_loader,epochs=epochs)
    # print(hist)
    train_acc = 100*sum(hist['train_acc'])/len(hist['train_acc'])
    val_acc = 100*sum(hist['val_acc'])/len(hist['val_acc'])
    print(f"Train Accuracy : {train_acc:.1f}%")
    print(f"Val Accuracy : {val_acc:.1f}%")

    # torch.save(model.state_dict(),PathConstants.MODEL_PATH(model_name))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs to train for')
    args = parser.parse_args()

    run(args)

