import os
import shutil
import unittest
from typing import Any, Dict

import numpy as np
from piepline.data_producer import BasicDataset

from pietoolbelt.pipeline.bagging.common import BasicBagging, BaggingResult
from pietoolbelt.pipeline.predict.common import AbstractPredictResult


class _PredictResultMock(AbstractPredictResult):
    def __init__(self, predicts: Dict[str, float]):
        self._predicts = predicts

    def get_predict(self, index: str) -> Any:
        return self._predicts[index]


class _DatasetMock(BasicDataset):
    def __init__(self, items: list):
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return {'target': item}


class TestBasicBagging(unittest.TestCase):
    RES_DIR = 'test_bagging_res'

    def tearDown(self):
        if os.path.exists(TestBasicBagging.RES_DIR):
            shutil.rmtree(TestBasicBagging.RES_DIR, ignore_errors=True)

    def test_init_with_incorrect_args(self):
        with self.assertRaises(Exception):
            BasicBagging(predicts_result={'r1': _PredictResultMock({'0': 1, '1': 0})}, calc_error=lambda x, y: x - y,
                         dataset=_DatasetMock([1, 0]), reduce=np.mean, result=BaggingResult(path=TestBasicBagging.RES_DIR))

    def test_init(self):
        try:
            BasicBagging(predicts_result={'r1': _PredictResultMock({'a': 1, 'b': 0}), 'r2': _PredictResultMock({'a': 2})},
                         calc_error=lambda x, y: x - y, dataset=_DatasetMock([1, 0]), reduce=np.mean,
                         result=BaggingResult(path=TestBasicBagging.RES_DIR))
        except Exception as err:
            self.fail("Fail instantiate BasicBagging. Error: [{}]".format(err))

    def test_bagging(self):
        bagging = BasicBagging(
            predicts_result={'r1': _PredictResultMock({'0': 1, '1': 0}), 'r2': _PredictResultMock({'0': 2, '1': 0})},
            calc_error=lambda x, y: x - y, dataset=_DatasetMock([1, 0]), reduce=np.mean,
            result=BaggingResult(path=TestBasicBagging.RES_DIR))

        res = bagging.run()
        self.assertEqual(list(res.keys()), ['r1'])

        shutil.rmtree(TestBasicBagging.RES_DIR, ignore_errors=True)

        bagging = BasicBagging(
            predicts_result={'r1': _PredictResultMock({'0': 1, '1': 2}), 'r2': _PredictResultMock({'0': 2, '1': 0}),
                             'r3': _PredictResultMock({'0': 2, '1': 0})},
            calc_error=lambda x, y: x - y, dataset=_DatasetMock([1, 0]), reduce=np.mean,
            result=BaggingResult(path=TestBasicBagging.RES_DIR))

        res = bagging.run()
        self.assertEqual(list(res.keys()), ['r2'])

        shutil.rmtree(TestBasicBagging.RES_DIR, ignore_errors=True)

        bagging = BasicBagging(
            predicts_result={'r1': _PredictResultMock({'0': 1, '1': 2}), 'r2': _PredictResultMock({'0': 2, '1': 0}),
                             'r3': _PredictResultMock({'0': 0, '1': 0})},
            calc_error=lambda x, y: abs(x - y), dataset=_DatasetMock([1, 0]), reduce=np.mean,
            result=BaggingResult(path=TestBasicBagging.RES_DIR))

        res = bagging.run()
        self.assertEqual(list(res.keys()), ['r2', 'r3'])
