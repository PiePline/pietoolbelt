import json
import os
from typing import List, Tuple, Callable, Any

import numpy as np
from piepline.data_producer import BasicDataset, DataProducer

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult
from pietoolbelt.pipeline.predict.common import AbstractPredictResult


class ThresholdsSearchResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path)
        self._meta_file_path = os.path.join(path, 'threshold.json')

        self._thresholds = {'result': None, 'values': []}

    def add_cmb(self, thresh: float, err: float):
        self._thresholds['values'].append({'thresh': thresh, 'err': err})

    def set_res(self, thresh: float, err: float):
        self._thresholds['result'].append({'thresh': thresh, 'err': err})

    def _dump_data(self):
        with open(self._meta_file_path, 'w') as meta_file:
            json.dump(self._thresholds, meta_file, indent=4)


class ThresholdsSearch:
    def __init__(self, predict_result: AbstractPredictResult, dataset: BasicDataset, calc_error: Callable[[Any, Any], List[float]],
                 reduce: Callable[[List[float]], float]):
        self._predict_result = predict_result
        self._dataset = dataset
        self._calc_error = calc_error
        self._reduce = reduce

    def calc_accuracy_on_thresh(self, data_producer: DataProducer, threshold: float) -> float:
        loader = data_producer.get_loader()

        errors_results = []
        for data in loader:
            cur_predicts = []
            for idx in data['data_idx']:
                pred = self._predict_result.get_predict(idx)
                cur_predicts.append(np.where(pred >= threshold, 1, 0).astype(np.float32))
            errors_results.extend(self._calc_error(np.concatenate(cur_predicts, axis=0), data['target'].numpy()))

        return float(self._reduce(errors_results))

    def run(self, thresholds: List[float], batch_size: int, workers_num: int) -> Tuple[float, float]:
        dp = DataProducer(self._dataset, batch_size=batch_size, num_workers=workers_num).pass_indices(need_pass=True)
        best_accuracy, best_idx = None, None
        for idx, thresh in enumerate(thresholds):
            cur_accuracy = self.calc_accuracy_on_thresh(dp, thresh)
            if best_accuracy is None or best_accuracy < cur_accuracy:
                best_accuracy, best_idx = cur_accuracy, idx

        return thresholds[best_idx], best_accuracy
