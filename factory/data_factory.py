import torch.nn as nn
from prep.batch_data_prep_training import BatchTrainingDataPreparation
from prep.batch_data_prep_test import BatchTestDataPreparation
from prep.incremental_data_prep import IncrementalDataPreparation

class DataFactory:
    data_prep = {
        'batch_train': BatchTrainingDataPreparation,
        'batch_inference': BatchTestDataPreparation,
        'incremental_inference': IncrementalDataPreparation
    }

    def select(self,strategy: str) -> nn.Module:
        if strategy not in self.data_prep:
            raise ValueError(f'Unsupported strategy: {strategy}')
        return self.data_prep[strategy]()
    
