import io
import os
from abc import ABCMeta, abstractmethod

import torch
from torch.nn import Module

import boto3

__all__ = ['ModelsWeightsStorage']


class AbstractWeightsStorage(metaclass=ABCMeta):
    def __init__(self, path: str = None):
        if path is None:
            if 'WEIGHTS_STORAGE' not in os.environ:
                raise Exception("Can't get weights storage path. Please define 'WEIGHTS_STORAGE' environment variable")
            self._path = os.environ['WEIGHTS_STORAGE']
        else:
            self._path = path

    @abstractmethod
    def _load(self, weights_file_name: str):
        """
        Internal method for loading weights
        """

    @abstractmethod
    def _save(self, model: Module, weights_file_name: str):
        """
        Internal method for saving weights
        """

    def load(self, model: Module, dataset: str = None, params: {} = None):
        weights_file_name = self._compile_weights_name(model=model, dataset=dataset, params=params)
        pretrained_weights = self._load(weights_file_name)
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

    def save(self, model: Module, dataset: str = None, params: {} = None):
        weights_file_name = os.path.join(self._path, self._compile_weights_name(model, dataset, params))
        self._save(model, weights_file_name)

    @staticmethod
    def _compile_weights_name(model: Module, dataset: str = None, params: {} = None):
        dataset_name = "__{}".format(str("any" if dataset is None else dataset))
        params_name = "" if params is None else ("__" + "__".join(['{}_{}'.format(k, v) for k, v in params.items()]))
        return str(type(model).__name__) + dataset_name + params_name + ".pth"


class ModelsWeightsStorage(AbstractWeightsStorage):
    def _load(self, weights_file_name: str):
        weights_file = os.path.join(self._path, weights_file_name)
        return torch.load(weights_file)

    def _save(self, model: Module, weights_file_name: str):
        weights_file = os.path.join(self._path, weights_file_name)

        if os.path.exists(weights_file):
            raise Exception("File '" + weights_file + "' also exists in storage")

        torch.save(model.state_dict(), weights_file)


class S3ModelWeightsStorage(AbstractWeightsStorage):
    def __init__(self, url: str, port: int, bucket: str, username: str, password: str):
        super().__init__(path=url)
        self._bucket = bucket
        self._client = boto3.client('s3', endpoint_url='http://{}:{}'.format(self._path, str(port)),
                                    aws_access_key_id=username, aws_secret_access_key=password)

    def _load(self, weights_file_name: str):
        content = self._client.get_object(Bucket=self._bucket, Key=weights_file_name)
        with io.BytesIO(content['Body'].read()) as in_file:
            return torch.load(in_file)

    def _save(self, model: Module, weights_file_name: str):
        with io.BytesIO() as out_file:
            torch.save(model.state_dict(), out_file)
            self._client.put_object(Bucket=self._bucket, Key=weights_file_name, Body=out_file)
