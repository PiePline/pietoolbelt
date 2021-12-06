from typing import Callable, List

from torch import Module

__all__ = ['ModelRegistry', 'ModelFoldsRegistry']


class ModelRegistry:
    def __init__(self):
        self._models = dict()

    def add_model(self, name: str, init_model: Callable[[...], Module]):
        if name in self._models:
            raise IndexError("Model [{}] already in registry".format(name))
        self._models[name] = init_model

    def get_models(self) -> List[str]:
        return list(self._models.keys())

    def init_model(self, name: str, **kwargs) -> Module:
        return self._models[name](**kwargs)


class ModelFoldsRegistry(ModelRegistry):
    def __init__(self, folds: List[str]):
        super().__init__()
        self._folds = folds

    def get_folds(self) -> List[str]:
        return self._folds
