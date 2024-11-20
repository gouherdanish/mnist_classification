import argparse
import torch
from torchsummary import summary

from constants import PathConstants
from factory.data_factory import DataFactory
from factory.model_factory import ModelFactory
from factory.inference_factory import InferenceFactory
from prep.test_data_prep_batch import BatchTestDataPreparation
from prep.test_data_prep_incremental import IncrementalTestDataPreparation
from utils import Utils

@Utils.timeit
def run(args):
    print(args)
    model_name = args.model_name
    test_image_path = args.test_image_path
    inference_strategy = 'incremental' if test_image_path != '' else 'batch'

    model_factory = ModelFactory()
    model = model_factory.select(model_name)
    state_dict = torch.load(PathConstants.MODEL_PATH(model_name), weights_only=True)
    model.load_state_dict(state_dict)

    inferencing = InferenceFactory.get(inference_strategy,model=model)
    if inference_strategy == 'incremental':
        data_prep = IncrementalTestDataPreparation(test_image_path=test_image_path)
        test_loader = data_prep.prepare()
        confidence, pred_label = inferencing.infer(test_loader=test_loader)
        print(confidence, pred_label)
    else: 
        data_prep = BatchTestDataPreparation()
        test_loader = data_prep.prepare()
        hist = inferencing.infer(test_loader=test_loader)
        print(hist)
        print(sum(hist['test_acc'])/len(hist['test_acc']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--test_image_path',type=str,default='',help='path of test image in case of incremental inference')
    args = parser.parse_args()

    run(args)

