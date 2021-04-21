import json
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Dict

from piepline.data_producer import AbstractDataset
from piepline.predict import Predictor
from piepline.utils.fsm import FileStructManager

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult, AbstractStepResult
from pietoolbelt.pipeline.train.folded_train import FoldedTrainResult

__all__ = ['AbstractPredictResult', 'AbstractPredict', 'FoldedPredict']


class AbstractPredictResult(AbstractStepDirResult):
    @abstractmethod
    def get_predict(self, index: str) -> Any:
        """
        Get predict by index

        Args:
            index (str): index of predict

        Returns:
             predict object
        """


class AbstractPredict(metaclass=ABCMeta):
    def __init__(self, predictor: Predictor, result: AbstractPredictResult):
        self._predictor = predictor
        self._result = result

    def predictor(self) -> Predictor:
        return self._predictor

    @abstractmethod
    def run(self, dataset: AbstractDataset, batch_size: int = 1, workers_num: int = 0) -> None:
        """
        Run prediction operation
        """


class FoldedPredictResult(AbstractStepDirResult):
    def __init__(self, path: str, init_result: Callable[[str], AbstractPredictResult]):
        super().__init__(path)

        self._init_result = init_result
        self._meta_file = os.path.join(self._path, 'meta.json')
        self._folds = dict()

        if os.path.exists(self._meta_file):
            with open(self._meta_file, 'r') as meta_file:
                self._folds = json.load(meta_file)

    def add_fold(self, fold_name: str):
        path = os.path.join(self._path, fold_name)
        self._folds[fold_name] = path

        with open(self._meta_file, 'w') as meta_file:
            json.dump(self._folds, meta_file, indent=4)

        return self._init_result(path)

    def get_all_results(self) -> Dict[str, AbstractPredictResult]:
        return {f: self._init_result(p) for f, p in self._folds.items()}


class FoldedPredict:
    def __init__(self, init_predictor: Callable[[FileStructManager, str], AbstractPredict], folded_train_result: FoldedTrainResult,
                 result: FoldedPredictResult):
        self._init_predictor = init_predictor
        self._train_result = folded_train_result
        self._result = result

    def run(self, dataset: AbstractDataset, batch_size: int = 1, workers_num: int = 0):
        for fold_name, fold_data in self._train_result.get_folds_meta().items():
            result_path = os.path.relpath(fold_data['path'], self._train_result.get_root_dir())

            fsm = FileStructManager(base_dir=fold_data['path'], is_continue=True)
            result = self._result.add_fold(fold_name)
            predictor = self._init_predictor(fsm, result)
            predictor.run(dataset=dataset, batch_size=batch_size, workers_num=workers_num)
