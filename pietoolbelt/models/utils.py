from typing import List

import torch
from torch import nn
from torch.nn import Module

__all__ = ['ClassificationModel', 'ModelWithActivation', 'Activation', 'ModelsContainer', 'calc_model_params_num']


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
    def __init__(self, encoder: Module, in_features: int, classes_num: int, pool=None):
        super().__init__()
        self._encoder = encoder
        self.fc = nn.Linear(in_features, classes_num)

        if pool is not None:
            self._pool = pool
            self._forward = self._forward_with_pool
        else:
            self._forward = self._forward_simple

    def forward(self, data):
        return self._forward(data)

    def _forward_simple(self, data):
        data = self._encoder(data)
        data = data.view(data.size(0), -1)
        return self.fc(data)

    def _forward_with_pool(self, data):
        data = self._encoder(data)
        data = self._pool(data)
        data = data.view(data.size(0), -1)
        return self.fc(data)


class ModelsContainer(Module):
    def __init__(self, models: List[Module], reduction: callable):
        super().__init__()

        self._models_names = []
        for i, model in enumerate(models):
            attr_name = "model_{}".format(i)
            setattr(self, attr_name, model)
            self._models_names.append(attr_name)

        self._reduction = reduction

    def forward(self, *args, **kwargs):
        results = [getattr(self, attr_name)(*args, **kwargs) for attr_name in self._models_names]
        return self._reduction(torch.stack(results, dim=0))
