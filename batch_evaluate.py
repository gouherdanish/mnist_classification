import argparse

from factory.model_factory import ModelFactory
from factory.inference_factory import InferenceFactory
from prep.batch_data_prep_test import BatchTestDataPreparation
from eval.evaluate import ModelEvaluator
from constants import DataConstants
from utils import Utils

@Utils.timeit
def run(args):
    model_name = args.model_name
    inference_strategy = 'batch'

    data_prep = BatchTestDataPreparation()
    test_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_name)

    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(model,dataloader=test_loader)
    print(eval_result)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='lenet',choices=['mlp','lenet'],help='type of model to run on')
    args = parser.parse_args()

    run(args)

