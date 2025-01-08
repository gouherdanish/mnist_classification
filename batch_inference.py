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
    eval_result = evaluator.evaluate(model,input_size=(DataConstants.IN_CHANNELS,*DataConstants.IMAGE_SIZE))
    print(eval_result)

    inferencing = InferenceFactory.get(strategy=inference_strategy,model=model)
    hist = inferencing.infer(test_loader=test_loader)
    # print(hist)
    test_acc = 100*sum(hist['test_acc'])/len(hist['test_acc'])
    print(f"Test Accuracy : {test_acc:.1f}%")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='lenet',choices=['mlp','lenet'],help='type of model to run on')
    args = parser.parse_args()

    run(args)

