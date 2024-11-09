import argparse
import torch
import torch.nn as nn

from constants import Constants, HyperParams, PathConstants
from data.test_data import TestDataPreparation
from factory.model_factory import ModelFactory
from inference import ModelInference

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    args = parser.parse_args()

    model_type = args.model_type

    data_prep = TestDataPreparation()
    test_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_type)
    state_dict = torch.load(PathConstants.MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)

    inferencing = ModelInference(model=model,test_loader=test_loader)
    hist = inferencing.infer()
    print(hist)
    print(sum(hist['test_acc'])/len(hist['test_acc']))

