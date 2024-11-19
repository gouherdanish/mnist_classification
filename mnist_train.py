import argparse
import torch

from constants import PathConstants
from prep.training_data_prep_batch import BatchTrainingDataPreparation
from factory.model_factory import ModelFactory
from factory.training import ModelTraining

def run(args):
    model_type = args.model_type
    epochs = args.epochs

    data_prep = BatchTrainingDataPreparation()
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs to train for')
    args = parser.parse_args()

    run(args)

