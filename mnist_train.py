import argparse
import torch
import torch.nn as nn

from constants import Constants, HyperParams, PathConstants
from data.training_data import TrainingDataPreparation
from factory.model_factory import ModelFactory
from training import ModelTraining

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs to train for')
    args = parser.parse_args()

    model_type = args.model_type
    epochs = args.epochs

    data_prep = TrainingDataPreparation()
    train_loader, val_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_type)

    training = ModelTraining(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader)
    hist = training.train(epochs=epochs)
    print(hist)

    torch.save(model.state_dict(),PathConstants.MODEL_PATH)

