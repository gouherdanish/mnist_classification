import torch.nn as nn
from prep.training_data_prep_batch import BatchTrainingDataPreparation
from prep.test_data_prep_batch import BatchTestDataPreparation
from prep.test_data_prep_incremental import IncrementalTestDataPreparation

class DataFactory:
    data_prep = {
        'batch_train': BatchTrainingDataPreparation,
        'batch_inference': BatchTestDataPreparation,
        'incremental_inference': IncrementalTestDataPreparation
    }

    def select(self,strategy: str) -> nn.Module:
        if strategy not in self.data_prep:
            raise ValueError(f'Unsupported strategy: {strategy}')
        return self.data_prep[strategy]()
    
