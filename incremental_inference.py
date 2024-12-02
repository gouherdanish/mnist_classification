import argparse
import torch

from constants import PathConstants
from factory.model_factory import ModelFactory
from factory.inference_factory import InferenceFactory
from prep.incremental_data_prep import IncrementalDataPreparation
from eval.evaluate import ModelEvaluator
from utils import Utils

@Utils.timeit
def run_for_image_path(test_image_path,model_name='lenet'):
    inference_strategy = 'incremental'

    data_prep = IncrementalDataPreparation(test_image_path=test_image_path)
    test_loader = data_prep.prepare()
    
    model_factory = ModelFactory()
    model = model_factory.select(model_name)
    state_dict = torch.load(PathConstants.MODEL_PATH(model_name), weights_only=True)
    model.load_state_dict(state_dict)

    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(model)
    print(eval_result)

    inferencing = InferenceFactory.get(inference_strategy,model=model)
    confidence, pred_label = inferencing.infer(test_loader=test_loader)
    print(confidence, pred_label)
    return confidence, pred_label

@Utils.timeit
def run_for_uploaded_img(pil_img,model_name='lenet'):
    inference_strategy = 'incremental'

    model_factory = ModelFactory()
    model = model_factory.select(model_name)
    state_dict = torch.load(PathConstants.MODEL_PATH(model_name), weights_only=True)
    model.load_state_dict(state_dict)

    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(model)
    print(eval_result)

    inferencing = InferenceFactory.get(inference_strategy,model=model)
    data_prep = IncrementalDataPreparation(pil_img=pil_img)
    test_loader = data_prep.prepare()
    confidence, pred_label = inferencing.infer(test_loader=test_loader)
    print(confidence, pred_label)
    return confidence, pred_label

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='lenet',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--test_image_path',type=str,default='',help='path of test image in case of incremental inference')
    args = parser.parse_args()

    confidence, pred_label = run_for_image_path(vars(**args))

