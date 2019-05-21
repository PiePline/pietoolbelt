import torch
from torch import nn
from torch.nn import Module

__all__ = ['ClassificationModel', 'ModelWithActivation', 'Activation', 'calc_model_params_num']


class Activation(Module):
    """
    Activation layer with option
    """

    def __init__(self, activation: str = None):
        super().__init__()
        self._activation = lambda x: x
        if activation is not None:
            if activation == 'sigmoid':
                self._activation = torch.nn.Sigmoid()
            else:
                raise NotImplementedError("Activation '{}' not implemented".format(activation))

    def forward(self, input):
        return self._activation(input)


class ModelWithActivation(Module):
    def __init__(self, base_model: Module, activation: str):
        super().__init__()
        self._base_model = base_model
        self._activation = Activation(activation)

    def forward(self, data):
        res = self._base_model(data)
        return self._activation(res)


def calc_model_params_num(model: Module):
    return sum(p.numel() for p in model.parameters())


class ClassificationModel(Module):
    def __init__(self, encoder: Module, in_features: int, classes_num: int):
        super().__init__()
        self._encoder = encoder
        self.fc = nn.Linear(in_features, classes_num)

    def forward(self, data):
        data = self._encoder(data)
        return self.fc(data)
