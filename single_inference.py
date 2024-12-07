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

    checkpoint_path = PathConstants.MODEL_PATH(model_name)
    assert Path(checkpoint_path).exists(), f"Path Not Found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    data_prep = SingleDataPreparation(img=img)
    test_loader = data_prep.prepare()

    model_factory = ModelFactory()
    model = model_factory.select(model_name)
    model.load_state_dict(checkpoint['model_state'])

    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(model)
    print(eval_result)

    inferencing = InferenceFactory.get(inference_strategy,model=model)
    confidence, pred_label = inferencing.infer(test_loader=test_loader)
    print(confidence, pred_label)
    return confidence, pred_label

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',type=str,required=True,help='path of test image in case of incremental inference')
    parser.add_argument('--model_name',type=str,default='lenet',choices=['mlp','lenet'],help='type of model to run on')
    args = parser.parse_args()

    # run(**vars(args))
    confidence, pred_label = run(**vars(args))

