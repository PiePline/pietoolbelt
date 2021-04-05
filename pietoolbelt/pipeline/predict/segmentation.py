import os

import numpy as np
from piepline.data_producer import DataProducer, AbstractDataset
from piepline.predict import Predictor

__all__ = ['SegmentationPredict', 'SegmentationPredictResult']

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult


class SegmentationPredictResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path)
        self._predicts_dir = os.path.join(path, 'predicts')
        self._meta_file = os.path.join(path, 'meta.json')

        if not os.path.exists(self._predicts_dir):
            os.makedirs(self._predicts_dir)

    def add_predict(self, index: str, predict: np.ndarray):
        with open(os.path.join(self._predicts_dir, '{}.npy'.format(index)), 'wb') as predict_file:
            np.save(predict_file, predict.astype(np.float32))


class SegmentationPredict:
    def __init__(self, predictor: Predictor, result: SegmentationPredictResult):
        self._predictor = predictor
        self._result = result

    def run(self, dataset: AbstractDataset, batch_size: int = 1, workers_num: int = 0):
        dp = DataProducer(dataset, batch_size=batch_size, num_workers=workers_num).global_shuffle(False). \
            pass_indices(need_pass=True).get_loader()

        for dat in dp:
            predict = self._predictor.predict(dat)
            predict = predict.detach().cpu().numpy()

            for cur_predict, index in zip(predict, dat['data_idx']):
                self._result.add_predict(index=index, predict=predict)
