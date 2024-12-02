import argparse
import torch
from torchsummary import summary

from constants import PathConstants
from factory.data_factory import DataFactory
from factory.model_factory import ModelFactory
from factory.inference_factory import InferenceFactory
from prep.batch_data_prep_test import BatchTestDataPreparation
from prep.incremental_data_prep import IncrementalDataPreparation
from eval.evaluate import ModelEvaluator
from utils import Utils

@Utils.timeit
def run(args):
    model_name = args.model_name
    test_image_path = args.test_image_path
    inference_strategy = 'incremental'

    model_factory = ModelFactory()
    model = model_factory.select(model_name)
    state_dict = torch.load(PathConstants.MODEL_PATH(model_name), weights_only=True)
    model.load_state_dict(state_dict)

    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(model)
    print(eval_result)

    inferencing = InferenceFactory.get(inference_strategy,model=model)
    data_prep = IncrementalDataPreparation(test_image_path=test_image_path)
    test_loader = data_prep.prepare()
    confidence, pred_label = inferencing.infer(test_loader=test_loader)
    print(confidence, pred_label)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--test_image_path',type=str,default='',help='path of test image in case of incremental inference')
    args = parser.parse_args()

    run(args)
