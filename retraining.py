from typing import Union
from pathlib import Path
import numpy as np
import cv2
import argparse
import torch

from constants import PathConstants
from prep.single_data_prep import SingleDataPreparation
from factory.model_factory import ModelFactory
from factory.training_factory import TrainingFactory

def run(img: Union[str,Path,np.ndarray],label: int,model_name:str='lenet'):
    data_prep = SingleDataPreparation(img=img,label=label)
    train_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_name)

    training = TrainingFactory.get(strategy='single',model=model)
    training.train(train_loader=train_loader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',type=str,required=True,help='path of test image in case of incremental inference')
    parser.add_argument('--label',type=int,required=True,help='path of test image in case of incremental inference')
    parser.add_argument('--model_name',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    args = parser.parse_args()

    run(**vars(args))

