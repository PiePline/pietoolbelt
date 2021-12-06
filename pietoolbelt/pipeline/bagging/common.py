import itertools
import json
import os
from typing import Callable, List, Any, Dict

import numpy as np
from piepline.data_producer import DataProducer, AbstractDataset
from piepline.predict import Predictor
from piepline.utils.fsm import FileStructManager

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult
from pietoolbelt.pipeline.model_registry import ModelRegistry
from pietoolbelt.pipeline.predict.common import AbstractPredictResult, AbstractPredict

__all__ = ['BasicBagging', 'BaggingResult']


class BaggingResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path)
        self._res_file = os.path.join(path, 'meta.json')

        self._combinations = self._load_prev_data()

    def add_cmb(self, combination: List[str], err: float):
        if 'all_cmb' in self._combinations:
            for res in self._combinations['all_cmb']:
                if combination == res['cmb']:
                    raise IndexError("Combination already in result file. Combination: {}".format(combination))

        self._combinations['all_cmb'].append({'cmb': combination, 'err': err})
        with open(self._res_file, 'w') as res_file:
            json.dump(self._combinations, res_file, indent=4)

    def set_result(self, combination: List[str], err: float):
        if 'res' in self._combinations and self._combinations['res'] is not None:
            raise Exception("Result already specified")

        self._combinations['res'] = {'cmb': combination, 'err': err}

        with open(self._res_file, 'w') as res_file:
            json.dump(self._combinations, res_file, indent=4)

    def get_combinations(self) -> Dict[str, List[Dict[str, List[str] or float]]]:
        return self._combinations['all_cmb']

    def get_result(self) -> Dict[str, List[str] or float] or None:
        return self._combinations['res']

    def _load_prev_data(self) -> dict:
        if os.path.exists(self._res_file):
            with open(self._res_file, 'r+') as res_file:
                data = json.load(res_file)
        else:
            data = dict()

        if 'all_cmb' not in data:
            data['all_cmb'] = []
        if 'res' not in data:
            data['res'] = None

        return data


class BasicBagging:
    def __init__(self, predicts_result: Dict[str, AbstractPredictResult], calc_error: Callable[[Any, Any], float],
                 dataset: AbstractDataset, reduce: Callable[[List[float]], float], result: BaggingResult):
        if len(predicts_result) < 2:
            raise Exception("Predicts results number must be >= 1")

        self._predicts_results = predicts_result
        self._dataset = dataset
        self._result = result

        self._calc_error = calc_error
        self._reduce = reduce

        self._pick_target = lambda x: x['target'].numpy()

    def set_pick_target(self, pick: Callable[[Any], Any]) -> 'BasicBagging':
        self._pick_target = pick
        return self

    def run(self, max_cmb_len: int = None) -> Dict[str, AbstractPredictResult]:
        all_combinations = self._generate_combinations(max_cmb_len)

        best_cmb, best_err = None, None
        for cmb in all_combinations:
            cur_err = self._calc_err_by_cmb(cmb)

            self._result.add_cmb(combination=cmb, err=cur_err)

            if best_err is None or cur_err < best_err:
                best_cmb, best_err = cmb, cur_err

        self._result.set_result(combination=best_cmb, err=best_err)

        return {c: self._predicts_results[c] for c in best_cmb}

    def _calc_err_by_cmb(self, combination: List[str]) -> float:
        errors = []

        dp = DataProducer(self._dataset, batch_size=1, num_workers=0).global_shuffle(False). \
            pass_indices(need_pass=True).get_loader()

        for dat in dp:
            predict = self._calc_predict(dat['data_idx'][0], [self._predicts_results[c] for c in combination])
            err = self._calc_error(predict, self._pick_target(dat))
            errors.append(err)

        return np.mean(errors)

    def _calc_predict(self, index: str, results: List[AbstractPredictResult]) -> Any:
        return self._reduce([res.get_predict(index) for res in results])

    def _generate_combinations(self, max_cmb_len: int = None) -> List[List[str]]:
        result = []
        max_cmb_len = len(self._predicts_results) if max_cmb_len is None else max_cmb_len
        for cmb_len in range(1, max_cmb_len + 1):
            for cmb in itertools.combinations(self._predicts_results.keys(), cmb_len):
                result.append(list(cmb))
        return result


class BaggingResultModelsLoader:
    def __init__(self, bagging_result: BaggingResult, model_registry: ModelRegistry):
        self._result = bagging_result
        self._model_registry = model_registry

    def run(self, init_predictor: Callable[[str], Predictor]):
        for fold_name in self._result.get_result()['cmb']:
            fsm = FileStructManager(base_dir=fold_data['path'], is_continue=True)
