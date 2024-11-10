import argparse
import torch
from torchsummary import summary

from constants import PathConstants
from factory.model_factory import ModelFactory
from data.test_data_prep_batch import BatchTestDataPreparation
from data.test_data_prep_incremental import IncrementalTestDataPreparation
from model.batch_inference import BatchInference
from model.incremental_inference import IncrementalInference
from utils import Utils

@Utils.timeit
def run_pipeline(args):
    model_type = args.model_type
    inference_strategy = args.inference_strategy
    test_image_path = args.test_image_path

    model_factory = ModelFactory()
    model = model_factory.select(model_type)
    state_dict = torch.load(PathConstants.MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)

    if inference_strategy == 'incremental_inference' or test_image_path != '':
        data_prep = IncrementalTestDataPreparation(test_image_path=test_image_path)  
        inferencing = IncrementalInference(model=model)
    else: 
        data_prep = BatchTestDataPreparation()
        inferencing = BatchInference(model=model)
    test_loader = data_prep.prepare()
    print(len(test_loader))

    hist = inferencing.infer(test_loader=test_loader)
    print(hist)
    print(sum(hist['test_acc'])/len(hist['test_acc']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',type=str,default='mlp',choices=['mlp','lenet'],help='type of model to run on')
    parser.add_argument('--inference_strategy',type=str,default='batch_inference',choices=['batch_inference','incremental_inference'],help='how to run inference - per image or per batch of image')
    parser.add_argument('--test_image_path',type=str,default='',help='path of test image in case of incremental inference')
    args = parser.parse_args()

    run_pipeline(args)

