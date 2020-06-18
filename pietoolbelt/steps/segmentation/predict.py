import os

import numpy as np
from piepline import Predictor, AbstractDataset


class PredictStep:
    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def run(self, dataset: AbstractDataset, out_path: str):
        if os.path.exists(out_path):
            os.makedirs(out_path)

        for i, dat in enumerate(dataset):
            predict = self._predictor.predict(dat)
            predict = np.squeeze(predict.detach().cpu().numpy())

            with open(os.path.join(out_path, '{}.npy'.format(i)), 'w') as predict_file:
                np.save(predict_file, predict.astype(np.float32))
