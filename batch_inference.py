from pathlib import Path
import argparse
import torch

from constants import PathConstants
from factory.model_factory import ModelFactory
from factory.inference_factory import InferenceFactory
from prep.batch_data_prep_test import BatchTestDataPreparation
from eval.evaluate import ModelEvaluator
from utils import Utils

@Utils.timeit
def run(args):
    model_name = args.model_name
    inference_strategy = 'batch'

    checkpoint_path = PathConstants.MODEL_PATH(model_name)
    assert Path(checkpoint_path).exists(), f"Path Not Found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model_factory = ModelFactory()
    model = model_factory.select(model_name)
    model.load_state_dict(checkpoint['model_state'])

    data_prep = BatchTestDataPreparation()
    test_loader = data_prep.prepare()

    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(model)
    print(eval_result)

    inferencing = InferenceFactory.get(strategy=inference_strategy,model=model)
    hist = inferencing.infer(test_loader=test_loader)
    print(hist)
    print(sum(hist['test_acc'])/len(hist['test_acc']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='lenet',choices=['mlp','lenet'],help='type of model to run on')
    args = parser.parse_args()

    run(args)

