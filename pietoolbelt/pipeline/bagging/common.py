import itertools
from typing import Callable, List, Any, Dict

import numpy as np
from piepline.data_producer import BasicDataset, DataProducer

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult
from pietoolbelt.pipeline.predict.common import AbstractPredictResult

__all__ = ['BasicBagging', 'BaggingResult']


class BaggingResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path)

    def set_config(self, config: dict):
        raise NotImplementedError()


class BasicBagging:
    def __init__(self, predicts_result: Dict[str, AbstractPredictResult], calc_error: Callable[[float, float], float],
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
        return [self._predicts_results[c] for c in best_cmb]

    def _calc_err_by_cmb(self, combination: List[str]) -> float:
        errors = []

        dp = DataProducer(self._dataset, batch_size=1, num_workers=0).global_shuffle(False). \
            pass_indices(need_pass=True).get_loader()

        for dat in dp:
            predict = self._calc_predict(dat['data_idx'], [self._predicts_results[c] for c in combination])
            err = self._calc_error(predict, dat['target'])
            errors.append(err)

        return np.mean(errors)

    def _calc_predict(self, index: str, results: List[AbstractPredictResult]) -> Any:
        return self._reduce([res.get_predict(index) for res in results])

    def _generate_combinations(self, max_cmb_len: int = None) -> List[List[str]]:
        result = []
        max_cmb_len = len(self._predicts_results) if max_cmb_len is None else max_cmb_len
        for cmb_len in range(1, max_cmb_len):
            for cmb in itertools.combinations(self._predicts_results.keys(), cmb_len):
                result.append(list(cmb))
        return result
