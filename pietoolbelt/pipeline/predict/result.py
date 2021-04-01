import os

import numpy as np

__all__ = ['PredictResult']


class PredictResult:
    def __init__(self, path: str):
        self._predicts_dir = os.path.join(path, 'predicts')
        self._meta_file = os.path.join(path, 'meta.json')

        if not os.path.exists(self._predicts_dir):
            os.makedirs(self._predicts_dir)

    def add_predict(self, index: str, predict: np.ndarray):
        with open(os.path.join(self._predicts_dir, '{}.npy'.format(index)), 'wb') as predict_file:
            np.save(predict_file, predict.astype(np.float32))
