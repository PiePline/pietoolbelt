import os

import torch
from torch.nn import Module

__all__ = ['ModelsWeightsStorage']


class ModelsWeightsStorage:
    def __init__(self, path: str = None):
        if path is None:
            if 'WEIGHTS_STORAGE' not in os.environ:
                raise Exception("Can't get weights storage path. Please define 'WEIGHTS_STORAGE' environment variable")
            self._path = os.environ['WEIGHTS_STORAGE']
        else:
            self._path = path

    def load(self, model: Module, dataset: str = None, params: {} = None):
        weights_file = os.path.join(self._path, self._compile_model_name(model, dataset, params))

        print("Model inited by file:", weights_file, end='; ')
        pretrained_weights = torch.load(weights_file)
        print("dict len before:", len(pretrained_weights), end='; ')

        processed = {}
        model_state_dict = model.state_dict()
        for k, v in pretrained_weights.items():
            if k.split('.')[0] == 'module' and not isinstance(model, torch.nn.DataParallel):
                k = '.'.join(k.split('.')[1:])
            elif isinstance(model, torch.nn.DataParallel) and k.split('.')[0] != 'module':
                k = 'module.' + k
            if k in model_state_dict:
                if v.device != model_state_dict[k].device:
                    v.to(model_state_dict[k].device)
                processed[k] = v

        model.load_state_dict(processed)

        print("dict len after:", len(processed))

    @staticmethod
    def _compile_model_name(model: Module, dataset: str, params: {}):
        dataset_name = "__{}".format(str("any" if dataset is None else dataset))
        params_name = "" if params is None else ("__" + "__".join(['{}_{}'.format(k, v) for k, v in params.items()]))
        return str(type(model).__name__) + dataset_name + params_name + ".pth"

    def save(self, model: Module, dataset: str = None, params: {} = None):
        weights_file = os.path.join(self._path, self._compile_model_name(model, dataset, params))

        if os.path.exists(weights_file):
            raise Exception("File '" + weights_file + "' also exists in storage")

        torch.save(model.state_dict(), weights_file)
