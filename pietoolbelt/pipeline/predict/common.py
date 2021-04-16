from abc import ABCMeta, abstractmethod
from typing import Any

from piepline.data_producer import AbstractDataset
from piepline.predict import Predictor

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult


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
