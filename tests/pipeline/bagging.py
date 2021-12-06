import itertools
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

    def add_predict(self, index: str, predict: Any):
        pass

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

    def test_predict_correctness(self):
        result = BaggingResult(path=TestBasicBagging.RES_DIR)
        bagging = BasicBagging(
            predicts_result={'r1': _PredictResultMock({'0': 1, '1': 2}), 'r2': _PredictResultMock({'0': 2, '1': 0}),
                             'r3': _PredictResultMock({'0': 0, '1': 0})},
            calc_error=lambda x, y: abs(x - y), dataset=_DatasetMock([1, 0]), reduce=np.mean,
            result=result)

        res = bagging.run()
        self.assertEqual(list(res.keys()), ['r2', 'r3'])
        self.assertEqual(result.get_result()['cmb'], ['r2', 'r3'])

    def test_combinations(self):
        result = BaggingResult(path=TestBasicBagging.RES_DIR)
        bagging = BasicBagging(
            predicts_result={'r1': _PredictResultMock({'0': 1, '1': 2}), 'r2': _PredictResultMock({'0': 2, '1': 0}),
                             'r3': _PredictResultMock({'0': 0, '1': 0})},
            calc_error=lambda x, y: abs(x - y), dataset=_DatasetMock([1, 0]), reduce=np.mean,
            result=result)

        res = bagging.run()
        self.assertEqual(list(res.keys()), ['r2', 'r3'])
        self.assertEqual(result.get_result()['cmb'], ['r2', 'r3'])

        all_items = ['r1', 'r2', 'r3']
        all_combinations = []
        for cmb_len in range(1, len(all_items) + 1):
            for cmb in itertools.combinations(all_items, cmb_len):
                all_combinations.append(list(cmb))

        for cmb1 in all_combinations:
            was_found = False
            for cmb2 in result.get_combinations():
                if cmb1 == cmb2['cmb']:
                    was_found = True

            if not was_found:
                self.fail("Combination {} doesn't exists".format(cmb1))

        shutil.rmtree(TestBasicBagging.RES_DIR, ignore_errors=True)

        result = BaggingResult(path=TestBasicBagging.RES_DIR)
        bagging = BasicBagging(
            predicts_result={'r1': _PredictResultMock({'0': 1, '1': 2}), 'r2': _PredictResultMock({'0': 2, '1': 0}),
                             'r3': _PredictResultMock({'0': 0, '1': 0})},
            calc_error=lambda x, y: abs(x - y), dataset=_DatasetMock([1, 0]), reduce=np.mean,
            result=result)

        res = bagging.run(max_cmb_len=2)
        self.assertEqual(list(res.keys()), ['r2', 'r3'])
        self.assertEqual(result.get_result()['cmb'], ['r2', 'r3'])

        all_items = ['r1', 'r2', 'r3']
        all_combinations = []
        for cmb_len in range(1, 3):
            for cmb in itertools.combinations(all_items, cmb_len):
                all_combinations.append(list(cmb))

        for cmb1 in all_combinations:
            was_found = False
            for cmb2 in result.get_combinations():
                if cmb1 == cmb2['cmb']:
                    was_found = True

            if not was_found:
                self.fail("Combination {} doesn't exists".format(cmb1))
