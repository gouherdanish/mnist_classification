import argparse
import torch

from constants import PathConstants
from ml.ml_projects.mnist_classification.prep.batch_data_prep_training import BatchTrainingDataPreparation
from factory.model_factory import ModelFactory
from factory.training import ModelTraining


def run(args):
    model_name = args.model_name
    epochs = args.epochs

    data_prep = BatchTrainingDataPreparation()
    train_loader, val_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_name)

    training = ModelTraining(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader)
    hist = training.train(epochs=epochs)

    print(hist)

    torch.save(model.state_dict(),PathConstants.MODEL_PATH(model_name))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs to train for')
    args = parser.parse_args()

    run(args)

