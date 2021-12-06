import os
from abc import abstractmethod, ABCMeta
from typing import List

__all__ = ['AbstractStepResult', 'AbstractStep', 'DatasetInPipeline', 'AbstractStepDirResult']

from piepline.data_producer import BasicDataset


class AbstractStepResult(metaclass=ABCMeta):
    @abstractmethod
    def get_output_paths(self) -> List[str]:
        """
        Get list of output paths
        """


class AbstractStepDirResult(AbstractStepResult, metaclass=ABCMeta):
    def __init__(self, path: str):
        self._path = path

        if not os.path.exists(path):
            os.makedirs(path)

    def get_output_paths(self) -> List[str]:
        return [self._path]


class ResultsContainer(AbstractStepResult):
    def __init__(self, results: List[AbstractStepResult]):
        self._results = results

    def get_output_paths(self) -> List[str]:
        out = []
        for res in self._results:
            out.extend(res.get_output_paths())
        return out


class AbstractStep(metaclass=ABCMeta):
    def __init__(self, output_res: AbstractStepResult, input_results: List[AbstractStepResult] = None):
        self._input_results = input_results
        self._output_res = output_res

    @abstractmethod
    def run(self):
        """
        Run step
        """

    def get_output_res(self) -> AbstractStepResult:
        return self._output_res

    def get_input_results(self) -> List[AbstractStepResult] or None:
        return self._input_results


class DatasetInPipeline(AbstractStepResult, BasicDataset, metaclass=ABCMeta):
    """
    Interface for dataset in pipeline
    """
