import json
from typing import Callable, Any, List, Dict

from piepline.data_producer import AbstractDataset, DataProducer

from pietoolbelt.pipeline.abstract_step import AbstractStepDirResult
from pietoolbelt.pipeline.predict.common import AbstractPredictResult

import numpy as np
import os

__all__ = ['MetricsCalcResult', 'MetricsCalculation']


class MetricsCalcResult(AbstractStepDirResult):
    def __init__(self, path: str):
        super().__init__(path)
        self._metrics_file = os.path.join(self._path, 'metrics.json')

        self._metrics = dict()
        if os.path.exists(self._metrics_file):
            with open(self._metrics_file, 'r') as metrics_file:
                self._metrics = json.load(metrics_file)

    def set_metric(self, name: str, value: float):
        if name in self._metrics:
            raise IndexError("Metric '{}' also registered".format(name))
        self._metrics[name] = value

        with open(self._metrics_file, 'w') as out:
            json.dump(self._metrics, out)

    def get_metrics(self) -> Dict[str, float]:
        return self._metrics

    def get_output_paths(self) -> List[str]:
        return [self._metrics_file]


class MetricsCalculation:
    def __init__(self, predict_res: AbstractPredictResult, result: MetricsCalcResult):
        self._predict_res = predict_res
        self._result = result

        self._pick_target = lambda x: x['target']
        self._metrics = dict()

    def add_metric(self, name: str, calc_metric: Callable[[Any, Any], float],
                   reduce: Callable[[List[float]], float] = None) -> 'MetricsCalculation':
        if name in self._metrics:
            raise IndexError("Metric [{}] already registered".format(name))
        self._metrics[name] = {'calc': calc_metric, 'reduce': np.mean if reduce is None else reduce}
        return self

    def run(self, dataset: AbstractDataset):
        dp = DataProducer(dataset, batch_size=1, num_workers=0).global_shuffle(False).pass_indices(need_pass=True).get_loader()

        result = dict()
        for item in dp:
            index = item['data_idx'][0]
            predict = self._predict_res.get_predict(index)
            target = self._pick_target(item)

            for metric_name, params in self._metrics.items():
                if metric_name not in result:
                    result[metric_name] = []
                result[metric_name].append(params['calc'](predict, target))

        for metric_name, values in result.items():
            value = self._metrics[metric_name]['reduce'](values)
            self._result.set_metric(name=metric_name, value=value)

    def set_pick_target(self, pick: Callable[[Any], Any]) -> 'MetricsCalculation':
        self._pick_target = pick
        return self
