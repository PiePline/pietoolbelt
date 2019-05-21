import torch
from torch.nn import Module


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
