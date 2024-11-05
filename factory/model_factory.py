from models.mlp import MLP

class ModelFactory:
    model_registry = {
        'mlp': MLP()
    }

    @classmethod
    def select(cls,model):
        return cls.model_registry[model]