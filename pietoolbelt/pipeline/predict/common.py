import os
from abc import ABCMeta, abstractmethod
from typing import Any, Callable

from piepline.data_producer import AbstractDataset
from piepline.predict import Predictor
from piepline.utils.fsm import FileStructManager

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult
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


class FoldedPredict:
    def __init__(self, init_predictor: Callable[[FileStructManager, str], AbstractPredict], folded_train_result: FoldedTrainResult):
        self._init_predictor = init_predictor
        self._train_result = folded_train_result

    def run(self, dataset: AbstractDataset, batch_size: int = 1, workers_num: int = 0):
        for fold_path in self._train_result.get_output_paths():
            result_path = os.path.relpath(self._train_result.get_root_dir(), fold_path)

            fsm = FileStructManager(base_dir=fold_path, is_continue=True)

            predictor = self._init_predictor(fsm, result_path)
            predictor.run(dataset=dataset, batch_size=batch_size, workers_num=workers_num)
