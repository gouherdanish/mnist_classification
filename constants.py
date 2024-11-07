import os

class Constants:
    IMAGE_SIZE = (28,28)
    VALIDATION_SPLIT_FRAC = 0.1

class PathConstants:
    CUR_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(CUR_DIR,'out','mnist_mlp_weights.pth')

class HyperParams:
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128

class ModelParams:
    FC1_NEURONS = 512
    OUTPUT_CLASSES = 10