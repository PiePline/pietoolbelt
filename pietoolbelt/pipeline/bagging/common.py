import itertools
import json
import os
from abc import ABCMeta
from multiprocessing import Pool
from typing import Dict, Callable, List, Any

import numpy as np
from piepline.data_producer import BasicDataset, DataProducer
from tqdm import tqdm

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult, ResultsContainer
from pietoolbelt.pipeline.predict.common import AbstractPredictResult
from pietoolbelt.utils.step_meta import StepMeta

__all__ = ['AbstractBagging']


class BaggingResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path)

    def set_config(self, config: dict):
        raise NotImplementedError()


class AbstractBagging(metaclass=ABCMeta):
    def __init__(self, predicts_result: List[AbstractPredictResult], calc_error: Callable[[float, float], float],
                 dataset: BasicDataset, reduce: Callable[[List[float]], float]):
        self._predicts_results = predicts_result
        self._dataset = dataset

        self._calc_error = calc_error
        self._reduce = reduce

    def run(self, max_cmb_len: int = None) -> List[AbstractPredictResult]:
        all_combinations = self._generate_combinations(max_cmb_len)

        best_cmb, best_err = None, None
        for cmb in all_combinations:
            cur_err = self._calc_err_by_cmb(cmb)
            if best_err is None or cur_err < best_err:
                best_cmb, best_err = cmb, cur_err
        return best_cmb

    def _calc_err_by_cmb(self, combination: List[AbstractPredictResult]) -> float:
        errors = []

        dp = DataProducer(self._dataset, batch_size=1, num_workers=0).global_shuffle(False). \
            pass_indices(need_pass=True).get_loader()

        for dat in dp:
            predict = self._calc_predict(dat['data_idx'], combination)
            err = self._calc_error(predict, dat['target'])
            errors.append(err)

        return np.mean(errors)

    def _calc_predict(self, index: str, results: List[AbstractPredictResult]) -> Any:
        return self._reduce([res.get_predict(index) for res in results])

    def _generate_combinations(self, max_cmb_len: int = None) -> List[List[AbstractPredictResult]]:
        result = []
        max_cmb_len = len(self._predicts_results) if max_cmb_len is None else max_cmb_len
        for cmb_len in range(1, max_cmb_len):
            for cmb in itertools.combinations(self._predicts_results, cmb_len):
                result.append(list(cmb))
        return result

    @staticmethod
    def _merge_predicts(predicts: []) -> dict:
        res_predict = np.median([p['predicts'] for p in predicts], axis=0)
        res_target = predicts[0]['targets']
        return {'predicts': res_predict, 'targets': res_target, 'model': [p['model'][0] for p in predicts]}

    @staticmethod
    def _merge_predicts_by_indices(data: []) -> dict:
        predicts, indices, metric = data
        cur_predict = AbstractBagging._merge_predicts(predicts)
        return dict({'metric': AbstractBagging._calc_metric_by_predict(cur_predict, metric), 'indices': indices},
                    **{k: v for k, v in cur_predict.items() if k not in ['predicts', 'targets']})

    @staticmethod
    def _calc_metric_by_predict(predict: {}, metric: callable) -> float:
        return metric(predict['predicts'], predict['targets'])
