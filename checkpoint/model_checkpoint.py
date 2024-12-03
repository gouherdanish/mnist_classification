import torch

class ModelCheckpoint:
    def __init__(self,epoch,model,optimizer) -> None:
        self.checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }

    def save(self,checkpoint_path):
        torch.save(self.checkpoint,checkpoint_path)
