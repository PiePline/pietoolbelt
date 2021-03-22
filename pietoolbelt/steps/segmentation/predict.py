import os

import numpy as np
from piepline.data_producer import DataProducer
from piepline.predict import Predictor


class PredictStep:
    def __init__(self, predictor: Predictor):
        self._predictor = predictor
        self._predictor.data_processor()

    def run(self, dataset: DataProducer, out_path: str, batch_size: int = 1, workers_num: int = 0):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        dp = DataProducer(dataset, batch_size=batch_size, num_workers=workers_num).global_shuffle(False).\
            pass_indices(need_pass=True).get_loader()

        for dat in dp:
            predict = self._predictor.predict(dat)
            predict = predict.detach().cpu().numpy()

            for cur_predict, index in zip(predict, dat['data_idx']):
                with open(os.path.join(out_path, '{}.npy'.format(index)), 'wb') as predict_file:
                    np.save(predict_file, cur_predict.astype(np.float32))
