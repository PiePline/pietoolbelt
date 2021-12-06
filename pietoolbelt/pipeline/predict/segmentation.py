import os
from typing import Any, List

import numpy as np
from piepline.predict import Predictor

from pietoolbelt.pipeline.abstract_step import AbstractStep
from pietoolbelt.pipeline.predict.common import AbstractPredict, AbstractPredictResult
from pietoolbelt.pipeline.train.common import TrainResult

__all__ = ['SegmentationPredict', 'SegmentationPredictResult', 'PipelineSegmentationPredict']


class SegmentationPredictResult(AbstractPredictResult):
    def __init__(self, path: str):
        super().__init__(path)
        self._predicts_dir = os.path.join(path, 'predicts')
        self._meta_file = os.path.join(path, 'meta.json')

        if not os.path.exists(self._predicts_dir):
            os.makedirs(self._predicts_dir)

        self._index_to_file_path = lambda index: os.path.join(self._predicts_dir, '{}.npy'.format(index))

    def add_predict(self, index: str, predict: np.ndarray):
        with open(self._index_to_file_path(index), 'wb') as predict_file:
            np.save(predict_file, predict.astype(np.float32))

    def get_predict(self, index: str) -> Any:
        with open(self._index_to_file_path(index), 'rb') as predict_file:
            return np.load(predict_file)


class SegmentationPredict(AbstractPredict):
    def __init__(self, predictor: Predictor, result: SegmentationPredictResult):
        super().__init__(predictor, result=result)

    def _predict_to_list(self, predict: Any) -> List[Any]:
        predict = predict.detach().cpu().numpy()
        return [np.squeeze(p) for p in predict]


class PipelineSegmentationPredict(SegmentationPredict, AbstractStep):
    def __init__(self, predictor: Predictor, train_result: TrainResult, result: SegmentationPredictResult):
        SegmentationPredict.__init__(self, predictor, result)
        AbstractStep.__init__(self, output_res=result, input_results=[train_result])
