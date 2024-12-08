from pathlib import Path
import argparse
import torch

from constants import PathConstants
from factory.model_factory import ModelFactory
from factory.inference_factory import InferenceFactory
from prep.single_data_prep import SingleDataPreparation
from eval.evaluate import ModelEvaluator
from utils import Utils

@Utils.timeit
def run(img,model_name='lenet'):
    inference_strategy = 'single'

    data_prep = SingleDataPreparation(img=img)
    test_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_name)

    inferencing = InferenceFactory.get(strategy=inference_strategy,model=model)
    res = inferencing.infer(test_loader=test_loader)
    print(res)
    return res['confidence'], res['pred_label']

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',type=str,required=True,help='path of test image in case of incremental inference')
    parser.add_argument('--model_name',type=str,default='lenet',choices=['mlp','lenet'],help='type of model to run on')
    args = parser.parse_args()

    # run(**vars(args))
    confidence, pred_label = run(**vars(args))

