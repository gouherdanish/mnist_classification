import os

class DataConstants:
    IMAGE_SIZE = (28,28)            # size of input image
    IN_CHANNELS = 1                 # grayscale
    VALIDATION_SPLIT_FRAC = 0.1     # 10% validation set out of training set
    OUTPUT_CLASSES = 10             # Labels 0 to 9

class PathConstants:
    CUR_DIR = os.path.dirname(__file__)

    def MODEL_PATH(model):
        return os.path.join(PathConstants.CUR_DIR,'out',f'mnist_{model}_weights.pth')

class HyperParams:
    LEARNING_RATE = 0.001           # Constant Learning rate
    BATCH_SIZE = 64                # One batch has 128 images

class MLPModelParams:
    NEURONS_FC1 = 512               
    NEURONS_FC2 = 256                  
    NEURONS_FC3 = 120                  
    NEURONS_FC4 = 84               

class LeNetModelParams:
    NUM_FILTERS_CONVLAYER1 = 6
    NUM_FILTERS_CONVLAYER2 = 16
    KERNEL_SIZE_CONVLAYER = 5
    KERNEL_SIZE_POOLINGLAYER = 2
    NEURONS_FC1 = 120
    NEURONS_FC2 = 84
