import json
import os
import shutil
import unittest
from typing import Any, Dict

from piepline.data_producer import AbstractDataset, BasicDataset

from pietoolbelt.pipeline.metrics_calc.common import MetricsCalculation, MetricsCalcResult
from pietoolbelt.pipeline.predict.common import AbstractPredictResult


class _PredictResMock(AbstractPredictResult):
    def __init__(self, predicts: Dict[str, Any]):
        self._predicts = predicts

    def get_predict(self, index: str) -> Any:
        return self._predicts[index]


class _BaseTest(unittest.TestCase):
    RESULT_DIR = 'metric_calc_res'

    def tearDown(self):
        if os.path.exists(MetricsCalcTest.RESULT_DIR):
            shutil.rmtree(MetricsCalcTest.RESULT_DIR, ignore_errors=True)


class _DatasetMock(BasicDataset):
    def __init__(self, items: list):
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return item


class PredictResultTest(_BaseTest):
    def test_init(self):
        try:
            MetricsCalcResult(path=PredictResultTest.RESULT_DIR)
        except Exception as err:
            self.fail("Initialisation failed with error: [{}]".format(err))

    def test_results_write(self):
        res = MetricsCalcResult(path=PredictResultTest.RESULT_DIR)
        metrics = {'a': 0.5, 'b': 23}
        res.set_metrics(metrics)

        metrics_file = os.path.join(PredictResultTest.RESULT_DIR, 'metrics.json')
        self.assertTrue(os.path.exists(metrics_file))
        self.assertEqual(res.get_output_paths(), [metrics_file])


class MetricsCalcTest(_BaseTest):
    def test_init(self):
        try:
            MetricsCalculation(predict_res=_PredictResMock({'a': 1, 'b': 2}),
                               result=MetricsCalcResult(path=MetricsCalcTest.RESULT_DIR))
        except Exception as err:
            self.fail("Initialisation failed with error: [{}]".format(err))

    def test_calc(self):
        result = MetricsCalcResult(path=MetricsCalcTest.RESULT_DIR)
        calc = MetricsCalculation(predict_res=_PredictResMock({'0': 1, '1': 0}), result=result)\
            .add_metric('metric1', lambda p, t: p - float(t))\
            .add_metric('metric2', lambda p, t: p * float(t), reduce=lambda v: 2 * sum(v))\
            .run(_DatasetMock([{'target': 1}, {'target': 0}]))

        with open(result.get_output_paths()[0], 'r') as metrics_file:
            self.assertEqual({'metric1': 0, 'metric2': 2}, json.load(metrics_file))
